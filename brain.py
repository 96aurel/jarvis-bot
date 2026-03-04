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
from pathlib import Path

from openai import OpenAI

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

    return ""


def _build_system_prompt(user_id: int) -> str:
    """Construit le prompt système complet avec identité + faits mémorisés."""
    soul = load_soul()
    facts_summary = memory.get_facts_summary(user_id)

    return f"""{soul}

---

## Faits mémorisés sur l'utilisateur

{facts_summary}

---

## Date et heure actuelles

{{datetime}}
"""


def _call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """Appelle l'API OpenAI et retourne le contenu de la réponse."""
    response = _client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=temperature,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


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

    # ── Phase de RÉFLEXION ──────────────────────────────
    reflection_messages = [
        {"role": "system", "content": (
            "Tu es l'agent de réflexion de Jarvis. "
            "Analyse le message de l'utilisateur et décide si tu dois utiliser un outil.\n\n"
            f"{TOOLS_DESCRIPTION}\n\n"
            f"Faits mémorisés :\n{memory.get_facts_summary(user_id)}"
        )},
        *history[-6:],  # Contexte récent pour la réflexion
    ]

    logger.info("Phase de réflexion…")
    reflection_raw = _call_llm(reflection_messages, temperature=0.3)

    # Essayer de parser la décision
    tool_result = ""
    try:
        # Extraire le JSON de la réponse (même si entouré de texte)
        json_match = None
        for line in reflection_raw.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_match = line
                break
        if not json_match:
            # Tenter sur tout le texte
            import re
            match = re.search(r'\{[^{}]+\}', reflection_raw)
            if match:
                json_match = match.group()

        if json_match:
            tool_call = json.loads(json_match)
            if tool_call.get("tool") and tool_call["tool"] != "none":
                tool_result = _execute_tool(tool_call, user_id)
                logger.info("Résultat de l'outil (%d caractères)", len(tool_result))
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Impossible de parser la réflexion : %s", e)

    # ── Phase de RÉPONSE ────────────────────────────────
    response_messages = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    if tool_result:
        response_messages.append({
            "role": "system",
            "content": (
                f"[Résultat de recherche / outil]\n\n{tool_result}\n\n"
                "Utilise ces informations pour enrichir ta réponse."
            ),
        })

    logger.info("Génération de la réponse…")
    response = _call_llm(response_messages)

    # 5. Sauvegarder la réponse
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
            "👋 Bonjour ! Je suis **Jarvis**, ton assistant personnel.\n\n"
            "Tu peux me parler naturellement ou utiliser ces commandes :\n"
            "• `/clear` — Effacer l'historique\n"
            "• `/facts` — Voir ce que j'ai mémorisé\n"
            "• `/search <requête>` — Rechercher sur le web\n"
            "• `/forget <clé>` — Oublier un fait\n"
        )

    elif cmd == "/clear":
        n = memory.clear_history(user_id)
        return f"🧹 Historique effacé ({n} messages supprimés)."

    elif cmd == "/facts":
        summary = memory.get_facts_summary(user_id)
        return f"📋 **Faits mémorisés :**\n\n{summary}"

    elif cmd == "/search" and arg:
        result = scraper.search_and_summarize(arg)
        memory.save_message(user_id, "user", f"/search {arg}")
        memory.save_message(user_id, "assistant", result)
        return result

    elif cmd == "/forget" and arg:
        deleted = memory.delete_fact(user_id, arg)
        return f"{'✅ Fait oublié.' if deleted else '❌ Fait non trouvé.'}"

    return None  # Pas une commande connue
