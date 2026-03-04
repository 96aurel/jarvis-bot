"""
telegram_bot.py — Interface Telegram pour Jarvis.

Utilise python-telegram-bot v20+ (async).
Gère les messages texte et les commandes.
"""

import logging

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    Defaults,
    filters,
)

import config
import brain

logger = logging.getLogger("jarvis.telegram")


# ── Vérification d'accès ────────────────────────────────

def _is_authorized(user_id: int) -> bool:
    """Vérifie si l'utilisateur est autorisé. Si la liste est vide, tout le monde est autorisé."""
    if not config.ALLOWED_USER_IDS:
        return True
    return user_id in config.ALLOWED_USER_IDS


# ── Handlers ─────────────────────────────────────────────

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les commandes /start, /clear, /facts, /search, /forget."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        await update.message.reply_text("⛔ Accès non autorisé.")
        return

    text = update.message.text
    response = brain.handle_command(user_id, text)

    if response:
        await _safe_reply(update.message, response)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les messages texte normaux (conversation avec Jarvis)."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        await update.message.reply_text("⛔ Accès non autorisé.")
        return

    user_message = update.message.text
    if not user_message:
        return

    logger.info("Message de %s : %s", update.effective_user.first_name, user_message[:80])

    # Indicateur de frappe pendant que Jarvis réfléchit
    await update.message.chat.send_action("typing")

    try:
        response = brain.think_and_respond(user_id, user_message)
    except Exception as e:
        logger.exception("Erreur lors du traitement : %s", e)
        response = "⚠️ Désolé, une erreur s'est produite. Réessaie dans un instant."

    await _safe_reply(update.message, response)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les erreurs non attrapées."""
    logger.error("Erreur Telegram : %s", context.error, exc_info=context.error)


# ── Utilitaires ──────────────────────────────────────────

def _escape_markdown(text: str) -> str:
    """Supprime les caractères Markdown problématiques pour un envoi en texte brut."""
    # Remplace les marqueurs Markdown par leur version lisible
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** → bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)         # *italic* → italic
    text = re.sub(r'__(.+?)__', r'\1', text)          # __underline__
    text = re.sub(r'_(.+?)_', r'\1', text)            # _italic_
    text = re.sub(r'`{3}[\s\S]*?`{3}', '', text)     # ```code blocks```
    text = re.sub(r'`(.+?)`', r'\1', text)            # `inline code`
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [link](url) → link
    return text


async def _safe_reply(message, text: str) -> None:
    """Envoie un message en texte brut (pas de parse_mode = pas d'erreur Markdown)."""
    for chunk in _split_message(text):
        await message.reply_text(chunk)


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Découpe un message trop long pour Telegram."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Essayer de couper à un saut de ligne
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# ── Lancement du bot ─────────────────────────────────────

def run_bot() -> None:
    """Configure et lance le bot Telegram."""
    logger.info("Démarrage du bot Telegram…")

    defaults = Defaults(parse_mode=None)
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).defaults(defaults).build()

    # Commandes
    for cmd in ["start", "clear", "facts", "search", "forget"]:
        app.add_handler(CommandHandler(cmd, handle_command))

    # Messages texte
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Erreurs
    app.add_error_handler(error_handler)

    # Lancement en polling (simple, pas besoin de webhook)
    logger.info("Bot prêt ! En attente de messages…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
