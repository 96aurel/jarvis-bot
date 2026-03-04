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
import re
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

# Extensions de fichiers texte supportés
_TEXT_EXTENSIONS = {
    '.txt', '.csv', '.json', '.py', '.js', '.ts', '.html', '.css',
    '.md', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.log',
    '.sh', '.bat', '.sql', '.r', '.java', '.c', '.cpp', '.h', '.rb',
}


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
        buffered = _message_buffers.pop(user_id, [])
        if not buffered:
            return

        typing_task = asyncio.create_task(_keep_typing(chat))

        if len(buffered) == 1:
            # Un seul message — réponse directe
            try:
                response = brain.think_and_respond(user_id, buffered[0]["text"])
            except Exception as e:
                logger.exception("Erreur : %s", e)
                response = "Desole, j'ai eu un souci. Reessaie."
            finally:
                typing_task.cancel()
            await _safe_reply(buffered[0]["message"], response, quote=True)
        else:
            # Plusieurs messages — réponse individuelle à chacun (quote séparé)
            logger.info("Batch de %d messages pour user %d", len(buffered), user_id)
            parts = []
            for i, buf in enumerate(buffered, 1):
                parts.append(f"[MSG {i}] {buf['text']}")
            combined = (
                f"L'utilisateur a envoye {len(buffered)} messages d'affilee :\n\n"
                + "\n".join(parts)
                + "\n\nReponds a chaque message separement. "
                "Prefixe chaque reponse avec [R1], [R2], etc. correspondant au numero du message. "
                "Si des messages sont lies tu peux mentionner le lien mais reponds quand meme a chacun. "
                "Si un message est trivial (genre 'ok' ou un emoji) tu peux repondre tres court."
            )
            try:
                response = brain.think_and_respond(user_id, combined)
            except Exception as e:
                logger.exception("Erreur : %s", e)
                response = "Desole, j'ai eu un souci. Reessaie."
            finally:
                typing_task.cancel()

            # Parser les réponses individuelles [R1], [R2], etc.
            replies = _parse_batch_response(response, len(buffered))
            if replies:
                for msg_idx, text in replies:
                    target = buffered[msg_idx]["message"] if 0 <= msg_idx < len(buffered) else buffered[-1]["message"]
                    await _safe_reply(target, text, quote=True)
            else:
                # Parsing échoué — envoyer tout au dernier message
                await _safe_reply(buffered[-1]["message"], response, quote=True)


def _parse_batch_response(response: str, num_messages: int) -> list[tuple[int, str]] | None:
    """Parse les marqueurs [R1], [R2], ... dans la réponse du LLM."""
    markers = list(re.finditer(r'\[R(\d+)\]', response))
    if not markers:
        return None

    results = []
    for j, match in enumerate(markers):
        msg_idx = int(match.group(1)) - 1  # 0-based
        start = match.end()
        end = markers[j + 1].start() if j + 1 < len(markers) else len(response)
        text = response[start:end].strip()
        if text:
            results.append((msg_idx, text))

    return results if results else None


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


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gère les fichiers (PDF, txt, code, etc.) — extraction + réponse."""
    user_id = update.effective_user.id
    if not _is_authorized(user_id):
        return

    doc = update.message.document
    if not doc:
        return

    filename = doc.file_name or "fichier"
    caption = update.message.caption or f"Analyse ce fichier : {filename}"
    ext = Path(filename).suffix.lower()

    try:
        file = await doc.get_file()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        # Extraire le texte selon le type de fichier
        if ext == '.pdf':
            text_content = _extract_pdf_text(tmp_path)
        elif ext in _TEXT_EXTENSIONS:
            text_content = Path(tmp_path).read_text(encoding='utf-8', errors='replace')
        else:
            Path(tmp_path).unlink(missing_ok=True)
            await update.message.reply_text(
                f"Je sais pas lire les fichiers {ext} pour l'instant. "
                "J'accepte les PDF et les fichiers texte/code."
            )
            return

        Path(tmp_path).unlink(missing_ok=True)

        if not text_content or len(text_content.strip()) < 10:
            await update.message.reply_text("J'ai pas reussi a extraire du texte de ce fichier.")
            return

        # Tronquer si trop long
        if len(text_content) > 8000:
            text_content = text_content[:8000] + "\n\n[... fichier tronque]"

        combined = f"[Fichier : {filename}]\n\n{text_content}\n\nQuestion/contexte : {caption}"

        typing_task = asyncio.create_task(_keep_typing(update.message.chat))
        try:
            response = brain.think_and_respond(user_id, combined)
        except Exception as e:
            logger.exception("Erreur document : %s", e)
            response = "J'ai pas reussi a analyser le fichier."
        finally:
            typing_task.cancel()

        await _safe_reply(update.message, response, quote=True)

    except Exception as e:
        logger.exception("Erreur document : %s", e)
        await update.message.reply_text("Erreur avec ce fichier.")


def _extract_pdf_text(path: str) -> str:
    """Extrait le texte d'un fichier PDF."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages[:30]):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i+1} ---\n{text}")
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pypdf non installe — lecture PDF impossible")
        return ""
    except Exception as e:
        logger.error("Erreur extraction PDF : %s", e)
        return ""


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
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(error_handler)

    logger.info("Bot pret !")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
