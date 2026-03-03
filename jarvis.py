#!/usr/bin/env python3
"""
jarvis.py — Point d'entrée principal de l'agent Jarvis.

Usage :
    python jarvis.py
"""

import logging
import sys
from pathlib import Path

# S'assurer que le dossier courant est dans le path
sys.path.insert(0, str(Path(__file__).parent))

import config
import memory
import telegram_bot


def setup_logging() -> None:
    """Configure le logging pour la console et un fichier."""
    log_format = "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("jarvis.log", encoding="utf-8"),
        ],
    )
    # Réduire le bruit des bibliothèques externes
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


def check_config() -> bool:
    """Vérifie que la configuration minimale est renseignée."""
    ok = True
    if not config.TELEGRAM_BOT_TOKEN or config.TELEGRAM_BOT_TOKEN == "TON_TOKEN_TELEGRAM_ICI":
        logging.error("❌ TELEGRAM_BOT_TOKEN non configuré dans config.py")
        ok = False
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "TA_CLE_OPENAI_ICI":
        logging.error("❌ OPENAI_API_KEY non configurée dans config.py")
        ok = False
    return ok


def main() -> None:
    """Lance Jarvis."""
    setup_logging()
    logger = logging.getLogger("jarvis")

    logger.info("=" * 50)
    logger.info("  🤖 Démarrage de JARVIS")
    logger.info("=" * 50)

    # Vérifier la config
    if not check_config():
        logger.error("Configure config.py avant de lancer Jarvis.")
        sys.exit(1)

    # Initialiser la base de données
    logger.info("Initialisation de la mémoire (SQLite)…")
    memory.init_db()

    # Vérifier que soul.md existe
    soul_path = Path(__file__).parent / config.SOUL_FILE
    if soul_path.exists():
        logger.info("Identité chargée depuis %s", soul_path)
    else:
        logger.warning("⚠️ %s introuvable — identité par défaut utilisée", config.SOUL_FILE)

    # Lancer le bot Telegram
    telegram_bot.run_bot()


if __name__ == "__main__":
    main()
