"""
telegram_bot.py — Interface Telegram pour Jarvis.

Comportement naturel :
  - Pas de message "je réfléchis" — juste typing comme une vraie personne
  - Batching : accumule les messages rapides et répond à tout d'un coup
  - Reply sélectif : cite un message spécifique quand c'est pertinent
  - Vocaux et photos supportés
"""

import asyncio
import logging
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from telegram import Update, Message
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

# Délai d'attente pour regrouper les messages rapides (secondes)
BATCH_DELAY = 3.0

# Buffer de messages en attente par user_id
_message_buffers: dict[int, list[dict]] = defaultdict(list)
# Locks pour éviter les réponses simultanées au même utilisateur
_user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
# Timers actifs par utilisateur
_batch_timers: dict[int, asyncio.Task] = {}


# ── Vérification d'accès ────────────────────────────────

def _is_authorized(user_id: int) -> bool:
    if not config.ALLOWED_USER_IDS:
        return True
    return user_id in config.ALLOWED_USER_IDS


# ── Handlers ─────────────────────────────────────────────

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les commandes /start, /clear, /facts, /search, /forget."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    text = update.message.text
    response = brain.handle_command(user_id, text)
    if response:
        await _safe_reply(update.message, response)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Accumule les messages rapides puis répond à tout d'un coup.
    Si l'utilisateur envoie plusieurs messages en < 3s, ils sont regroupés.
    """
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    user_message = update.message.text
    if not user_message:
        return

    # Récupérer le contexte du message cité (reply-to)
    quoted_text = _get_quoted_context(update)
    if quoted_text:
        user_message = f"[En reponse a : \"{quoted_text}\"]\n{user_message}"

    # Ajouter au buffer
    _message_buffers[user_id].append({
        "text": user_message,
        "message": update.message,
        "message_id": update.message.message_id,
    })

    logger.info("Message buffered (%s) : %s", update.effective_user.first_name, user_message[:60])

    # Annuler le timer précédent s'il existe
    if user_id in _batch_timers and not _batch_timers[user_id].done():
        _batch_timers[user_id].cancel()

    # Lancer un nouveau timer — quand il expire, on traite tout le batch
    _batch_timers[user_id] = asyncio.create_task(
        _process_batch_after_delay(user_id, update.message.chat, context)
    )


async def _process_batch_after_delay(user_id: int, chat, context) -> None:
    """Attend BATCH_DELAY secondes puis traite tous les messages accumulés."""
    await asyncio.sleep(BATCH_DELAY)

    async with _user_locks[user_id]:
        # Récupérer et vider le buffer
        buffered = _message_buffers.pop(user_id, [])
        if not buffered:
            return

        # Le dernier message reçu — on répondra à celui-ci ou on citera un autre
        last_msg: Message = buffered[-1]["message"]

        # Construire le message combiné
        if len(buffered) == 1:
            # Un seul message : comportement normal
            combined_text = buffered[0]["text"]
            reply_to_msg = last_msg
        else:
            # Plusieurs messages : les numéroter pour que le LLM puisse citer
            parts = []
            for i, buf in enumerate(buffered, 1):
                parts.append(f"[Message {i}] {buf['text']}")
            combined_text = (
                f"L'utilisateur a envoye {len(buffered)} messages d'affilee :\n\n"
                + "\n".join(parts)
                + "\n\nReponds a tous ces messages de maniere fluide. "
                "Si certains messages sont lies, regroupe ta reponse. "
                "Si un message merite une reponse specifique, mentionne-le."
            )
            reply_to_msg = last_msg
            logger.info("Batch de %d messages pour user %d", len(buffered), user_id)

        # Typing naturel — comme une vraie personne
        typing_task = asyncio.create_task(_keep_typing(chat))

        try:
            response = brain.think_and_respond(user_id, combined_text)
        except Exception as e:
            logger.exception("Erreur : %s", e)
            response = "Desole, j'ai eu un souci. Reessaie."
        finally:
            typing_task.cancel()

        # Choisir à quel message répondre (quote)
        # Si plusieurs messages, on reply au dernier (le plus récent)
        # Si un seul, on reply directement dessus
        quote_msg = _pick_quote_message(buffered, response)

        await _safe_reply(quote_msg, response, quote=True)


def _pick_quote_message(buffered: list[dict], response: str) -> Message:
    """
    Choisit le message le plus pertinent à citer.
    Si un seul message → celui-ci.
    Si plusieurs → le dernier par défaut (plus naturel sur Telegram).
    """
    if len(buffered) == 1:
        return buffered[0]["message"]

    # On cite le dernier message (le plus naturel dans un flux Telegram)
    return buffered[-1]["message"]


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les messages vocaux — transcription + réponse."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    try:
        file = await voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        transcript = brain.transcribe_audio(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        if not transcript:
            await update.message.reply_text("J'ai pas compris l'audio, tu peux repeter ?")
            return

        logger.info("Vocal transcrit : %s", transcript[:80])

        typing_task = asyncio.create_task(_keep_typing(update.message.chat))
        try:
            response = brain.think_and_respond(user_id, transcript)
        except Exception as e:
            logger.exception("Erreur : %s", e)
            response = "Desole, j'ai eu un souci avec le vocal."
        finally:
            typing_task.cancel()

        await _safe_reply(update.message, response, quote=True)

    except Exception as e:
        logger.exception("Erreur traitement vocal : %s", e)
        await update.message.reply_text("J'ai pas reussi a traiter le vocal.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les photos — analyse + réponse."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    caption = update.message.caption or "Que vois-tu sur cette image ?"

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        typing_task = asyncio.create_task(_keep_typing(update.message.chat))
        try:
            response = brain.analyze_image(user_id, tmp_path, caption)
        except Exception as e:
            logger.exception("Erreur : %s", e)
            response = "J'ai pas reussi a voir l'image."
        finally:
            typing_task.cancel()

        Path(tmp_path).unlink(missing_ok=True)
        await _safe_reply(update.message, response, quote=True)

    except Exception as e:
        logger.exception("Erreur photo : %s", e)
        await update.message.reply_text("Erreur avec l'image.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Erreur Telegram : %s", context.error, exc_info=context.error)


# ── Utilitaires ──────────────────────────────────────────

def _get_quoted_context(update: Update) -> str | None:
    reply = update.message.reply_to_message
    if reply and reply.text:
        return reply.text[:500]
    return None


async def _keep_typing(chat, interval: float = 4.0) -> None:
    try:
        while True:
            await chat.send_action("typing")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


async def _safe_reply(message: Message, text: str, quote: bool = False) -> None:
    for chunk in _split_message(text):
        kwargs = {}
        if quote:
            kwargs["reply_to_message_id"] = message.message_id
        try:
            await message.reply_text(chunk, **kwargs)
        except Exception:
            try:
                await message.reply_text(chunk)
            except Exception:
                pass


def _split_message(text: str, max_len: int = 4000) -> list[str]:
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
    logger.info("Demarrage du bot Telegram...")

    defaults = Defaults(parse_mode=None)
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).defaults(defaults).build()

    for cmd in ["start", "clear", "facts", "search", "forget"]:
        app.add_handler(CommandHandler(cmd, handle_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(error_handler)

    logger.info("Bot pret !")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
