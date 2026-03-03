"""
memory.py — Mémoire persistante de Jarvis (SQLite).

Stocke :
  • l'historique complet des conversations
  • les faits importants (études, deadlines, préférences…)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

import config


def _connect() -> sqlite3.Connection:
    """Retourne une connexion à la base SQLite."""
    conn = sqlite3.connect(config.DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Crée les tables si elles n'existent pas encore."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            role        TEXT    NOT NULL,          -- 'user' | 'assistant' | 'system'
            content     TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS facts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            category    TEXT    NOT NULL,           -- ex: 'exam', 'deadline', 'preference'
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conv_user   ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_facts_user   ON facts(user_id, category);
    """)
    conn.commit()
    conn.close()


# ── Conversations ────────────────────────────────────────

def save_message(user_id: int, role: str, content: str) -> None:
    """Enregistre un message dans l'historique."""
    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_history(user_id: int, limit: int = 20) -> list[dict]:
    """
    Récupère les N derniers messages d'un utilisateur.
    Retourne une liste de dicts {"role": …, "content": …}.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM conversations "
        "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    # Remet dans l'ordre chronologique
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_history(user_id: int) -> int:
    """Supprime tout l'historique d'un utilisateur. Renvoie le nombre de lignes supprimées."""
    conn = _connect()
    cur = conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


# ── Faits / mémoire longue ──────────────────────────────

def save_fact(user_id: int, category: str, key: str, value: str) -> None:
    """
    Enregistre ou met à jour un fait.
    Si (user_id, category, key) existe déjà, on met à jour la valeur.
    """
    now = datetime.now().isoformat()
    conn = _connect()
    existing = conn.execute(
        "SELECT id FROM facts WHERE user_id = ? AND category = ? AND key = ?",
        (user_id, category, key),
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE facts SET value = ?, updated_at = ? WHERE id = ?",
            (value, now, existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO facts (user_id, category, key, value, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, category, key, value, now, now),
        )
    conn.commit()
    conn.close()


def get_facts(user_id: int, category: str | None = None) -> list[dict]:
    """Récupère les faits d'un utilisateur, optionnellement filtrés par catégorie."""
    conn = _connect()
    if category:
        rows = conn.execute(
            "SELECT category, key, value, updated_at FROM facts "
            "WHERE user_id = ? AND category = ? ORDER BY updated_at DESC",
            (user_id, category),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT category, key, value, updated_at FROM facts "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_fact(user_id: int, key: str) -> bool:
    """Supprime un fait par sa clé. Renvoie True si quelque chose a été supprimé."""
    conn = _connect()
    cur = conn.execute(
        "DELETE FROM facts WHERE user_id = ? AND key = ?", (user_id, key)
    )
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_facts_summary(user_id: int) -> str:
    """Renvoie un résumé texte de tous les faits mémorisés, pour injection dans le prompt."""
    facts = get_facts(user_id)
    if not facts:
        return "Aucun fait mémorisé pour le moment."
    lines = []
    for f in facts:
        lines.append(f"- [{f['category']}] {f['key']} : {f['value']}")
    return "\n".join(lines)
