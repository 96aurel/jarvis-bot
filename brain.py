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
import re
import base64
import time
from pathlib import Path

from openai import OpenAI, RateLimitError, APIStatusError

import config
import memory
import scraper

logger = logging.getLogger("jarvis.brain")


def _extract_json(text: str) -> str | None:
    """
    Extrait le premier objet JSON valide d'un texte, meme avec des objets imbriques.
    Utilise le comptage de accolades pour trouver les limites exactes.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                # Verifier que c'est du JSON valide
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    return None
    return None


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

# Client Ollama (local, illimite, prioritaire si active)
_ollama_client = None
try:
    if config.OLLAMA_ENABLED:
        _ollama_client = OpenAI(
            api_key="ollama",  # Ollama n'a pas besoin de cle mais le SDK en veut une
            base_url=config.OLLAMA_BASE_URL,
        )
        # Test rapide pour verifier que Ollama tourne
        _ollama_client.models.list()
        logger.info("Ollama connecte : %s @ %s", config.OLLAMA_MODEL, config.OLLAMA_BASE_URL)
except Exception as e:
    logger.warning("Ollama non disponible (%s) — sera ignore dans la chaine", str(e)[:100])
    _ollama_client = None

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

# Verification critique au demarrage
if not _gemini_client:
    logger.error(
        "!!! GEMINI_API_KEY MANQUANTE !!! "
        "Sans Gemini, le bot n'a AUCUN fallback quand Groq rate-limit. "
        "Images et fichiers ne fonctionneront pas non plus. "
        "Ajoute GEMINI_API_KEY dans tes variables d'environnement."
    )


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

    # Chaîne de fallback vision : Gemini (meilleur + pas de rate limit) → Groq → principal
    vision_chain: list[tuple[OpenAI, str]] = []
    if _gemini_client:
        vision_chain.append((_gemini_client, config.GEMINI_MODEL))
    if config.GROQ_API_KEY and config.GROQ_VISION_MODEL:
        vision_chain.append((_groq_client or _client, config.GROQ_VISION_MODEL))
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
        except (RateLimitError, APIStatusError) as e:
            logger.warning("Vision rate limit sur %s : %s", model, str(e)[:200])
            continue
        except Exception as e:
            logger.warning("Vision echouee sur %s : %s", model, str(e)[:200])
            continue

    logger.error("Toute la vision chain a echoue. Chain : %s", [m for _, m in vision_chain])
    return "J'arrive pas a analyser l'image la. Verifie que GEMINI_API_KEY est configure."


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
Outils disponibles :
1. web_search : {"tool": "web_search", "query": "ta recherche"}
2. scrape_url : {"tool": "scrape_url", "url": "https://..."}
3. delete_fact : {"tool": "delete_fact", "key": "nom_du_fait"}
4. remind_me : {"tool": "remind_me", "delay_seconds": 300, "message": "Ton rappel"}

Reponds UNIQUEMENT avec un JSON valide.
Ajoute TOUJOURS un champ "facts" listant les faits personnels EXPLICITES du message.

Exemples :
{"tool": "none", "facts": []}
{"tool": "web_search", "query": "score PSG", "facts": []}
{"tool": "none", "facts": [{"category": "exam", "key": "maths", "value": "vendredi 15"}]}
{"tool": "web_search", "query": "...", "facts": [{"category": "perso", "key": "prenom", "value": "Thomas"}]}
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
Faits memorises : {facts_summary}
Date/heure : {{datetime}}

## Format
- Par DEFAUT envoie UN SEUL message. Le ||| pour separer en bulles est RARE (1 fois sur 5 max). La majorite de tes reponses = une seule bulle.
- Rappel : [REMIND:Xs:message] ex: [REMIND:300s:C'est l'heure !]
- Reaction emoji (rare, quand c'est motive) : [REACT:emoji] ex: [REACT:😂]
"""


def _call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """
    Appelle l'API LLM avec chaine de fallback :
      1. Ollama (local, illimite) — si actif
      2. Groq 70B (rapide, gratuit mais limite)
      3. Google Gemini (fallback gratuit, limites enormes)
    """
    # Construire la chaine : [(client, model), ...]
    chain: list[tuple[OpenAI, str]] = []

    # Ollama en premier si disponible
    if _ollama_client:
        chain.append((_ollama_client, config.OLLAMA_MODEL))

    # Provider principal (Groq ou OpenAI)
    chain.append((_client, _model))

    # Fallback Gemini (dernier recours)
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

        except (RateLimitError, APIStatusError) as e:
            status = getattr(e, 'status_code', 429)
            logger.warning("Rate limit/API error (%s) sur %s : %s", status, model, str(e)[:200])
            continue

        except Exception as e:
            logger.error("Erreur LLM sur %s : %s", model, str(e)[:200])
            continue

    # Tous les fournisseurs ont echoue
    logger.error("AUCUN fournisseur LLM disponible. Chain testee : %s",
                 [m for _, m in chain])
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

    # ── Phase de RÉFLEXION (outil + extraction de faits en 1 appel) ────
    tool_result = ""
    tool_used = None
    try:
        reflection_messages = [
            {"role": "system", "content": (
                "Tu es l'agent de reflexion de Jarvis. "
                "Analyse le message et fais 2 choses :\n"
                "1. Decide si un outil est necessaire\n"
                "2. Extrais les faits personnels EXPLICITES du message "
                "(exam, deadline, prenom, formation, etc.)\n\n"
                "N'utilise web_search QUE pour des infos que tu ne connais PAS "
                "(actualites, prix, resultats, etc.).\n\n"
                f"{TOOLS_DESCRIPTION}\n\n"
                f"Faits deja memorises :\n{memory.get_facts_summary(user_id)}"
            )},
            *history[-4:],
        ]

        logger.info("Phase de reflexion...")
        reflection_raw = _call_llm(reflection_messages, temperature=0.3)
        logger.info("Reflexion brute : %s", reflection_raw[:300])

        # Parser le JSON avec extraction robuste (gere les objets imbriques)
        json_str = _extract_json(reflection_raw)

        if json_str:
            tool_call = json.loads(json_str)
            logger.info("JSON parse : tool=%s, facts=%d",
                        tool_call.get("tool", "none"), len(tool_call.get("facts", [])))

            # Sauvegarder les faits detectes
            for fact in tool_call.get("facts", []):
                if isinstance(fact, dict) and fact.get("key") and fact.get("value"):
                    cat = fact.get("category", "general")
                    memory.save_fact(user_id, cat, fact["key"], fact["value"])
                    logger.info("Fait sauvegarde : [%s] %s = %s", cat, fact["key"], fact["value"])

            # Executer l'outil si besoin
            if tool_call.get("tool") and tool_call["tool"] != "none":
                tool_used = tool_call["tool"]
                tool_result = _execute_tool(tool_call, user_id)
                logger.info("Outil %s -> resultat (%d chars)", tool_used, len(tool_result))
        else:
            logger.warning("Aucun JSON valide trouve dans la reflexion")
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
