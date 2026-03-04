"""
Configuration centrale de Jarvis.

Les secrets sont lus depuis les variables d'environnement.
En local : crée un fichier .env (copie .env.example).
En production (Render, Railway…) : configure-les dans le dashboard.
"""

import os
from pathlib import Path

# Charge le .env local s'il existe (utile en dev, ignoré en prod)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv est optionnel en production

# ── Telegram ────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Liste des user IDs Telegram autorisés (séparés par des virgules)
_raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: list[int] = [
    int(uid.strip()) for uid in _raw_ids.split(",") if uid.strip().isdigit()
]

# ── LLM Provider ─────────────────────────────────────────
# Choix du fournisseur : "groq" (gratuit & rapide) ou "openai"
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "groq")

# ── Groq (recommandé — gratuit) ─────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL: str = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_WHISPER_MODEL: str = "whisper-large-v3"  # Pour la transcription vocale (gratuit)
GROQ_VISION_MODEL: str = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")

# ── OpenAI (optionnel) ──────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ── Google Gemini (fallback gratuit — très généreux) ─────
# Clé API : https://aistudio.google.com/app/apikey
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ── Chemins ──────────────────────────────────────────────
SOUL_FILE = "soul.md"
DATABASE_FILE = "jarvis_memory.db"

# ── Scraping ─────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
SCRAPE_MAX_CHARS = 4000
