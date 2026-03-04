"""
brain.py — Cerveau de Jarvis : boucle de réflexion et appel LLM.

Architecture :
  1. Reçoit le message utilisateur
  2. Construit le contexte (soul + mémoire + faits)
  3. Demande au LLM s'il a besoin d'un outil (réflexion)
  4. Si oui → exécute l'outil, injecte le résultat
  5. Génère la réponse finale
  6. Extrait et sauvegarde les faits importants
"""

import json
import logging
import base64
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

import config
import memory
import scraper

logger = logging.getLogger("jarvis.brain")


def _build_llm_client() -> tuple[OpenAI, str]:
    """
    Crée le client LLM selon le provider configuré.
    Retourne (client, model_name).
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        logger.info("LLM provider : Groq (%s)", config.GROQ_MODEL)
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
        return client, config.GROQ_MODEL

    else:  # openai par défaut
        logger.info("LLM provider : OpenAI (%s)", config.OPENAI_MODEL)
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        return client, config.OPENAI_MODEL


_client, _model = _build_llm_client()

# Client Gemini (fallback gratuit quand Groq est limite)
try:
    _gemini_client = OpenAI(
        api_key=config.GEMINI_API_KEY,
        base_url=config.GEMINI_BASE_URL,
    ) if config.GEMINI_API_KEY else None
    if _gemini_client:
        logger.info("Gemini fallback configure : %s", config.GEMINI_MODEL)
    else:
        logger.warning("GEMINI_API_KEY absente — pas de fallback Gemini")
except Exception as e:
    logger.error("Erreur init Gemini : %s", e)
    _gemini_client = None

# Client Groq separe pour Whisper (toujours Groq, meme si le LLM est OpenAI)
_groq_client = OpenAI(
    api_key=config.GROQ_API_KEY,
    base_url=config.GROQ_BASE_URL,
) if config.GROQ_API_KEY else None


# ── Transcription audio (Whisper via Groq) ──────────────

def transcribe_audio(file_path: str) -> str:
    """Transcrit un fichier audio avec Whisper via Groq (gratuit)."""
    if not _groq_client:
        logger.warning("Pas de GROQ_API_KEY, transcription impossible.")
        return ""
    try:
        with open(file_path, "rb") as f:
            transcription = _groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="fr",
            )
        return transcription.text.strip()
    except Exception as e:
        logger.error("Erreur transcription audio : %s", e)
        return ""


# ── Analyse d'image ──────────────────────────────────────

def analyze_image(user_id: int, image_path: str, caption: str) -> str:
    """Analyse une image en l'envoyant au LLM avec le caption."""
    from datetime import datetime

    # Encoder l'image en base64
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error("Erreur lecture image : %s", e)
        return "Erreur lors de la lecture de l'image."

    # Construire le prompt
    system_prompt = _build_system_prompt(user_id).replace(
        "{datetime}", datetime.now().strftime("%A %d %B %Y, %H:%M")
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
            }},
        ]},
    ]

    # Chaîne de fallback vision : Groq vision → Gemini → modèle principal
    vision_chain: list[tuple[OpenAI, str]] = []
    if config.GROQ_API_KEY and config.GROQ_VISION_MODEL:
        vision_chain.append((_groq_client or _client, config.GROQ_VISION_MODEL))
    if _gemini_client:
        vision_chain.append((_gemini_client, config.GEMINI_MODEL))
    vision_chain.append((_client, _model))  # Dernier recours

    for i, (client, model) in enumerate(vision_chain):
        try:
            logger.info("Vision essai #%d : %s", i, model)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            result = response.choices[0].message.content.strip()
            memory.save_message(user_id, "user", f"[Photo] {caption}")
            memory.save_message(user_id, "assistant", result)
            return result
        except Exception as e:
            logger.warning("Vision echouee sur %s : %s", model, str(e)[:200])
            continue

    return "J'ai pas reussi a analyser cette image."


# ── Chargement de l'identité ────────────────────────────

def load_soul() -> str:
    """Charge le fichier soul.md et le retourne comme texte."""
    path = Path(__file__).parent / config.SOUL_FILE
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("soul.md introuvable, utilisation d'une identité par défaut.")
    return "Tu es Jarvis, un assistant personnel intelligent et serviable."


# ── Outils disponibles ──────────────────────────────────

TOOLS_DESCRIPTION = """
Tu disposes des outils suivants que tu peux décider d'utiliser AVANT de répondre :

1. **web_search** : Rechercher des informations sur internet.
   Usage : {"tool": "web_search", "query": "ta recherche ici"}

2. **scrape_url** : Extraire le contenu d'une page web spécifique.
   Usage : {"tool": "scrape_url", "url": "https://example.com"}

3. **save_fact** : Mémoriser un fait important sur l'utilisateur.
   Usage : {"tool": "save_fact", "category": "exam", "key": "math_exam", "value": "15 mars 2026"}

4. **delete_fact** : Oublier un fait précédemment mémorisé.
   Usage : {"tool": "delete_fact", "key": "math_exam"}

5. **remind_me** : Programmer un message a envoyer plus tard (rappel/timer).
   Usage : {"tool": "remind_me", "delay_seconds": 300, "message": "Eh c'est l'heure de bosser !"}
   L'utilisateur peut dire "dans 5 min", "dans 1h", "dans 30 secondes", etc.

Si tu as besoin d'un outil, réponds UNIQUEMENT avec un JSON valide contenant "tool" et ses paramètres.
Si tu n'as besoin d'aucun outil, réponds avec : {"tool": "none"}
"""


def _execute_tool(tool_call: dict, user_id: int) -> str:
    """Exécute un outil et retourne le résultat sous forme de texte."""
    tool_name = tool_call.get("tool", "none")

    if tool_name == "web_search":
        query = tool_call.get("query", "")
        logger.info("🔍 Outil web_search : %s", query)
        return scraper.search_and_summarize(query)

    elif tool_name == "scrape_url":
        url = tool_call.get("url", "")
        logger.info("🌐 Outil scrape_url : %s", url)
        return scraper.scrape_url(url)

    elif tool_name == "save_fact":
        category = tool_call.get("category", "general")
        key = tool_call.get("key", "")
        value = tool_call.get("value", "")
        memory.save_fact(user_id, category, key, value)
        logger.info("💾 Fait sauvegardé : [%s] %s = %s", category, key, value)
        return f"Fait mémorisé : [{category}] {key} = {value}"

    elif tool_name == "delete_fact":
        key = tool_call.get("key", "")
        deleted = memory.delete_fact(user_id, key)
        logger.info("🗑️ Fait supprimé : %s (trouvé=%s)", key, deleted)
        return f"Fait '{key}' {'supprimé' if deleted else 'non trouvé'}."

    elif tool_name == "remind_me":
        delay = tool_call.get("delay_seconds", 300)
        msg = tool_call.get("message", "Hey, c'est ton rappel !")
        logger.info("Rappel programme dans %ds : %s", delay, msg[:80])
        # Marqueur spécial que telegram_bot.py va détecter
        return f"__REMINDER_SET__|{delay}|{msg}"

    return ""


def _build_system_prompt(user_id: int) -> str:
    """Construit le prompt système complet avec identité + faits mémorisés."""
    soul = load_soul()
    facts_summary = memory.get_facts_summary(user_id)

    return f"""{soul}

---

## Faits memorises sur l'utilisateur

{facts_summary}

---

## Date et heure actuelles

{{datetime}}

---

## Format de reponse

Quand tu veux envoyer plusieurs messages separes (comme un vrai humain sur Telegram), utilise ||| pour separer les bulles.
Exemple : "Salut !|||Ca va ? J'ai regarde ton truc|||En gros c'est ca..."
Le separateur ||| cree un nouveau message avec un petit delai de frappe entre chaque — ca fait naturel.
N'abuse pas : 1-3 bulles max en general. Parfois une seule bulle suffit.

Si tu dois programmer un rappel, inclus le marqueur [REMIND:Xs:message] dans ta reponse.
Exemple : "Ok je te rappelle dans 5 min [REMIND:300s:Eh, c'est l'heure de bosser sur ton projet !]"

Si tu veux reagir au message de l'utilisateur avec un emoji (parce que c'est drole, impressionnant, etc.),
ajoute [REACT:emoji] dans ta reponse. Exemples : [REACT:😂] [REACT:🔥] [REACT:💀] [REACT:❤️]
Fais-le de temps en temps quand c'est motive, pas a chaque message.
"""


def _call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """
    Appelle l'API LLM avec cha\u00eene de fallback :
      1. Mod\u00e8le principal (Groq 70B)
      2. Mod\u00e8le l\u00e9ger Groq (8B)
      3. Google Gemini (fallback gratuit, limites \u00e9normes)
    """
    # Construire la cha\u00eene : [(client, model), ...]
    chain: list[tuple[OpenAI, str]] = [(_client, _model)]

    # Fallback Groq (mod\u00e8le l\u00e9ger)
    if config.LLM_PROVIDER.lower() == "groq" and config.GROQ_FALLBACK_MODEL != _model:
        chain.append((_client, config.GROQ_FALLBACK_MODEL))

    # Fallback Gemini (toujours en dernier recours)
    if _gemini_client:
        chain.append((_gemini_client, config.GEMINI_MODEL))

    for i, (client, model) in enumerate(chain):
        try:
            logger.info("Essai LLM #%d : %s", i, model)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
            )
            if i > 0:
                logger.info("Reponse via fallback #%d : %s", i, model)
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            logger.warning("Rate limit sur %s : %s", model, str(e)[:200])
            continue

        except Exception as e:
            logger.error("Erreur LLM sur %s : %s", model, str(e)[:200])
            continue

    # Tous les fournisseurs ont \u00e9chou\u00e9
    return (
        "Tous mes fournisseurs d'IA sont temporairement indisponibles. "
        "Reessaie dans quelques minutes !"
    )


# ── Boucle de réflexion principale ──────────────────────

def think_and_respond(user_id: int, user_message: str) -> str:
    """
    Pipeline complet :
      1. Sauvegarde le message utilisateur
      2. Phase de réflexion : le LLM décide s'il a besoin d'un outil
      3. Si outil → exécution + injection du résultat
      4. Phase de réponse : génération de la réponse finale
      5. Sauvegarde la réponse
    """
    from datetime import datetime

    # 1. Sauvegarder le message utilisateur
    memory.save_message(user_id, "user", user_message)

    # 2. Récupérer l'historique
    history = memory.get_recent_history(user_id, limit=20)

    # 3. Construire le prompt système
    system_prompt = _build_system_prompt(user_id).replace(
        "{datetime}", datetime.now().strftime("%A %d %B %Y, %H:%M")
    )

    # ── Phase de RÉFLEXION (skip si rate limited)────────
    # On tente la réflexion mais si ça échoue, on passe directement à la réponse
    tool_result = ""
    tool_used = None  # Nom de l'outil utilisé (ou None)
    try:
        reflection_messages = [
            {"role": "system", "content": (
                "Tu es l'agent de reflexion de Jarvis. "
                "Analyse le message de l'utilisateur et decide si tu dois utiliser un outil.\n\n"
                "IMPORTANT : N'utilise web_search QUE si l'utilisateur demande une info "
                "que tu ne connais PAS avec certitude (actualites, paroles de chansons, "
                "prix actuels, resultats sportifs, etc.). "
                "Si tu peux repondre de memoire avec certitude, utilise {\"tool\": \"none\"}.\n\n"
                "IMPORTANT SUR save_fact : N'utilise save_fact QUE si l'utilisateur dit EXPLICITEMENT "
                "une info factuelle dans le message actuel (ex: 'j'ai un exam de maths le 15'). "
                "Ne deduis PAS de faits a partir du contexte. Ne memorise PAS d'infos inventees. "
                "En cas de doute, ne sauvegarde PAS.\n\n"
                f"{TOOLS_DESCRIPTION}\n\n"
                f"Faits memorises :\n{memory.get_facts_summary(user_id)}"
            )},
            *history[-6:],
        ]

        logger.info("Phase de réflexion...")
        reflection_raw = _call_llm(reflection_messages, temperature=0.3)

        # Essayer de parser la décision
        json_match = None
        for line in reflection_raw.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_match = line
                break
        if not json_match:
            import re
            match = re.search(r'\{[^{}]+\}', reflection_raw)
            if match:
                json_match = match.group()

        if json_match:
            tool_call = json.loads(json_match)
            if tool_call.get("tool") and tool_call["tool"] != "none":
                tool_used = tool_call["tool"]
                tool_result = _execute_tool(tool_call, user_id)
                logger.info("Outil %s → resultat (%d caracteres)", tool_used, len(tool_result))
    except Exception as e:
        logger.warning("Phase de reflexion echouee, passage direct a la reponse : %s", e)

    # ── Phase de RÉPONSE ────────────────────────────────
    # Gérer le cas spécial d'un rappel programmé via outil
    reminder_info = None
    if tool_result and tool_result.startswith("__REMINDER_SET__"):
        _, delay_str, remind_msg = tool_result.split("|", 2)
        reminder_info = (int(delay_str), remind_msg)
        tool_result = f"Rappel programme dans {int(delay_str)//60} minute(s) : \"{remind_msg}\". Confirme a l'utilisateur."

    response_messages = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    if tool_result:
        response_messages.append({
            "role": "system",
            "content": (
                f"[OUTIL UTILISE : {tool_used}]\n\n"
                f"Voici les VRAIS resultats obtenus :\n\n{tool_result}\n\n"
                "Base ta reponse UNIQUEMENT sur ces resultats. "
                "Si les resultats sont vides ou insuffisants, dis-le honnetement. "
                "Ne complete PAS avec des informations inventees."
            ),
        })
    else:
        response_messages.append({
            "role": "system",
            "content": (
                "[AUCUN OUTIL UTILISE]\n\n"
                "Tu n'as PAS fait de recherche web pour cette reponse. "
                "Ne dis PAS que tu as cherche sur internet ou que tu as trouve quelque chose en ligne. "
                "Reponds uniquement avec ce que tu sais de memoire. "
                "Si tu n'es pas sur d'une info, dis clairement que tu ne sais pas "
                "et propose de faire une recherche web."
            ),
        })

    logger.info("Generation de la reponse...")
    response = _call_llm(response_messages)

    # Injecter le marqueur de rappel pour que telegram_bot.py le détecte
    if reminder_info:
        delay, msg = reminder_info
        response += f"\n[REMIND:{delay}s:{msg}]"

    # 5. Sauvegarder la réponse (seulement si c'est une vraie reponse)
    if not response.startswith("Tous mes fournisseurs"):
        memory.save_message(user_id, "assistant", response)

    return response


# ── Commandes spéciales ──────────────────────────────────

def handle_command(user_id: int, command: str) -> str | None:
    """
    Gère les commandes spéciales (/start, /clear, /facts, /search).
    Retourne une réponse ou None si ce n'est pas une commande.
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/start":
        return (
            "Bonjour ! Je suis Jarvis, ton assistant personnel.\n\n"
            "Tu peux me parler naturellement ou utiliser ces commandes :\n"
            "/clear - Effacer l'historique\n"
            "/facts - Voir ce que j'ai mémorisé\n"
            "/search <requête> - Rechercher sur le web\n"
            "/forget <clé> - Oublier un fait\n"
        )

    elif cmd == "/clear":
        n = memory.clear_history(user_id)
        return f"Historique effacé ({n} messages supprimés)."

    elif cmd == "/facts":
        summary = memory.get_facts_summary(user_id)
        return f"Faits mémorisés :\n\n{summary}"

    elif cmd == "/search" and arg:
        result = scraper.search_and_summarize(arg)
        memory.save_message(user_id, "user", f"/search {arg}")
        memory.save_message(user_id, "assistant", result)
        return result

    elif cmd == "/forget" and arg:
        deleted = memory.delete_fact(user_id, arg)
        return f"{'✅ Fait oublié.' if deleted else '❌ Fait non trouvé.'}"

    return None  # Pas une commande connue
