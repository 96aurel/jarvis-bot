"""
telegram_bot.py — Interface Telegram pour Jarvis.

Utilise python-telegram-bot v20+ (async).
Fonctionnalités :
  - Messages texte (avec reply intelligent)
  - Messages vocaux (transcription Whisper)
  - Photos (description via contexte)
  - Réactions emoji pendant la réflexion
  - Indicateur de réflexion (message édité)
  - Contexte des messages cités (reply-to)
"""

import asyncio
import logging
import random
import tempfile
from pathlib import Path

from telegram import Update, ReactionTypeEmoji
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

# Phrases d'attente aléatoires pour l'indicateur de réflexion
_THINKING_PHRASES = [
    "Hmm, laisse-moi réfléchir...",
    "Je regarde ça...",
    "Un instant, je réfléchis...",
    "Bonne question, je cherche...",
    "Je me penche dessus...",
]


# ── Vérification d'accès ────────────────────────────────

def _is_authorized(user_id: int) -> bool:
    if not config.ALLOWED_USER_IDS:
        return True
    return user_id in config.ALLOWED_USER_IDS


# ── Réactions emoji ──────────────────────────────────────

async def _react(message, emoji: str) -> None:
    """Ajoute une réaction emoji sur un message. Silencieux si ça échoue."""
    try:
        await message.set_reaction([ReactionTypeEmoji(emoji=emoji)])
    except Exception:
        pass  # Pas grave si les réactions ne sont pas supportées


async def _clear_reaction(message) -> None:
    """Retire les réactions."""
    try:
        await message.set_reaction([])
    except Exception:
        pass


# ── Handlers ─────────────────────────────────────────────

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les commandes /start, /clear, /facts, /search, /forget."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        await update.message.reply_text("Accès non autorisé.")
        return

    text = update.message.text
    response = brain.handle_command(user_id, text)

    if response:
        # Commandes : toujours reply au message de l'utilisateur
        await _safe_reply(update.message, response, quote=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les messages texte normaux (conversation avec Jarvis)."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    user_message = update.message.text
    if not user_message:
        return

    # Récupérer le contexte du message cité (reply-to)
    quoted_text = _get_quoted_context(update)
    if quoted_text:
        user_message = f"[En réponse à : \"{quoted_text}\"]\n\n{user_message}"

    logger.info("Message de %s : %s", update.effective_user.first_name, user_message[:80])

    # Réaction 👀 = "j'ai vu, je réfléchis"
    await _react(update.message, "👀")

    # Envoyer un message de réflexion qu'on éditera ensuite
    thinking_msg = await update.message.reply_text(
        random.choice(_THINKING_PHRASES),
        reply_to_message_id=update.message.message_id,
    )

    # Maintenir le "typing" en arrière-plan
    typing_task = asyncio.create_task(_keep_typing(update.message.chat))

    try:
        response = brain.think_and_respond(user_id, user_message)
    except Exception as e:
        logger.exception("Erreur lors du traitement : %s", e)
        response = "Désolé, une erreur s'est produite. Réessaie dans un instant."
    finally:
        typing_task.cancel()

    # Retirer la réaction 👀, mettre ✅
    await _clear_reaction(update.message)
    await _react(update.message, "✅")

    # Éditer le message de réflexion avec la vraie réponse
    chunks = _split_message(response)
    try:
        await thinking_msg.edit_text(chunks[0])
    except Exception:
        # Si l'édition échoue, envoyer un nouveau message
        await _safe_reply(update.message, chunks[0], quote=False)

    # S'il y a des chunks supplémentaires (réponse longue), les envoyer à la suite
    for chunk in chunks[1:]:
        await _safe_reply(update.message, chunk, quote=False)

    # Retirer ✅ après 5 secondes
    await asyncio.sleep(5)
    await _clear_reaction(update.message)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les messages vocaux — transcription automatique + réponse."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    await _react(update.message, "🎧")

    # Télécharger le fichier vocal
    try:
        file = await voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        # Transcrire avec Whisper via Groq
        transcript = brain.transcribe_audio(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)  # Nettoyer

        if not transcript:
            await update.message.reply_text("Je n'ai pas réussi à comprendre l'audio.")
            return

        logger.info("Transcription vocale : %s", transcript[:80])

        # Informer de la transcription
        thinking_msg = await update.message.reply_text(
            f"J'ai entendu : \"{transcript[:200]}\"\n\nJe réfléchis...",
            reply_to_message_id=update.message.message_id,
        )

        typing_task = asyncio.create_task(_keep_typing(update.message.chat))
        try:
            response = brain.think_and_respond(user_id, transcript)
        except Exception as e:
            logger.exception("Erreur : %s", e)
            response = "Désolé, une erreur s'est produite."
        finally:
            typing_task.cancel()

        await _clear_reaction(update.message)
        await _react(update.message, "✅")

        try:
            await thinking_msg.edit_text(response[:4000])
        except Exception:
            await _safe_reply(update.message, response, quote=True)

    except Exception as e:
        logger.exception("Erreur traitement vocal : %s", e)
        await update.message.reply_text("Erreur lors du traitement du message vocal.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les photos — description + réponse au caption éventuel."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    await _react(update.message, "👀")

    caption = update.message.caption or "Que vois-tu sur cette image ?"

    # Télécharger la photo (meilleure résolution)
    try:
        photo = update.message.photo[-1]  # Plus grande taille
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        thinking_msg = await update.message.reply_text(
            "Je regarde l'image...",
            reply_to_message_id=update.message.message_id,
        )

        response = brain.analyze_image(user_id, tmp_path, caption)
        Path(tmp_path).unlink(missing_ok=True)

        await _clear_reaction(update.message)
        await _react(update.message, "✅")

        try:
            await thinking_msg.edit_text(response[:4000])
        except Exception:
            await _safe_reply(update.message, response, quote=True)

    except Exception as e:
        logger.exception("Erreur traitement photo : %s", e)
        await update.message.reply_text("Erreur lors de l'analyse de l'image.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les erreurs non attrapées."""
    logger.error("Erreur Telegram : %s", context.error, exc_info=context.error)


# ── Utilitaires ──────────────────────────────────────────

def _get_quoted_context(update: Update) -> str | None:
    """Extrait le texte du message auquel l'utilisateur répond (reply-to)."""
    reply = update.message.reply_to_message
    if reply and reply.text:
        # Tronquer si trop long
        return reply.text[:500]
    return None


async def _keep_typing(chat, interval: float = 4.0) -> None:
    """Envoie l'action 'typing' en boucle jusqu'à annulation."""
    try:
        while True:
            await chat.send_action("typing")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _safe_reply(message, text: str, quote: bool = False) -> None:
    """Envoie un message en texte brut, avec ou sans citation."""
    for chunk in _split_message(text):
        kwargs = {}
        if quote:
            kwargs["reply_to_message_id"] = message.message_id
        try:
            await message.reply_text(chunk, **kwargs)
        except Exception:
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
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# ── Lancement du bot ─────────────────────────────────────

def run_bot() -> None:
    """Configure et lance le bot Telegram."""
    logger.info("Démarrage du bot Telegram...")

    defaults = Defaults(parse_mode=None)
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).defaults(defaults).build()

    # Commandes
    for cmd in ["start", "clear", "facts", "search", "forget"]:
        app.add_handler(CommandHandler(cmd, handle_command))

    # Messages texte
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Messages vocaux
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Photos
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Erreurs
    app.add_error_handler(error_handler)

    logger.info("Bot pret ! En attente de messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
