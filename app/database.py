import json
import sqlite3
from pathlib import Path
from typing import Any

from app.chunking_service import TextChunk


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "document_assistant.db"


def get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.
    """
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


def initialise_database() -> None:
    """
    Create all required database tables.
    """
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                character_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_number INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                start_index INTEGER NOT NULL,
                end_index INTEGER NOT NULL,
                character_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE,

                UNIQUE(document_id, chunk_number)
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                response_time_ms INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE
            );
            """
        )


def save_document(
    *,
    original_filename: str,
    stored_filename: str,
    file_type: str,
    file_path: str,
    word_count: int,
    character_count: int,
    chunks: list[TextChunk],
) -> int:
    """
    Save document metadata and all document chunks.
    """
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                original_filename,
                stored_filename,
                file_type,
                file_path,
                word_count,
                character_count,
                chunk_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_filename,
                stored_filename,
                file_type,
                file_path,
                word_count,
                character_count,
                len(chunks),
            ),
        )

        document_id = cursor.lastrowid

        if document_id is None:
            raise RuntimeError(
                "The document could not be saved."
            )

        chunk_rows = [
            (
                document_id,
                chunk.chunk_id,
                chunk.text,
                chunk.start_index,
                chunk.end_index,
                len(chunk.text),
            )
            for chunk in chunks
        ]

        connection.executemany(
            """
            INSERT INTO document_chunks (
                document_id,
                chunk_number,
                chunk_text,
                start_index,
                end_index,
                character_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )

        return document_id


def get_document(
    document_id: int,
) -> dict[str, Any] | None:
    """
    Retrieve one document by its database ID.
    """
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                original_filename,
                stored_filename,
                file_type,
                file_path,
                word_count,
                character_count,
                chunk_count,
                created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def get_document_chunks(
    document_id: int,
) -> list[dict[str, Any]]:
    """
    Retrieve all SQLite chunks belonging to a document.
    """
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                document_id,
                chunk_number,
                chunk_text,
                start_index,
                end_index,
                character_count,
                created_at
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_number
            """,
            (document_id,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]

def get_all_documents(
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Return the most recently uploaded documents.
    """
    safe_limit = max(
        1,
        min(limit, 500),
    )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                original_filename,
                stored_filename,
                file_type,
                file_path,
                word_count,
                character_count,
                chunk_count,
                created_at
            FROM documents
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]

def delete_document(
    document_id: int,
) -> None:
    """
    Delete a document.

    Related chunks and chat history are deleted automatically
    because their foreign keys use ON DELETE CASCADE.
    """
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (document_id,),
        )


def save_chat_message(
    *,
    document_id: int,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    response_time_ms: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    model_name: str,
) -> int:
    """
    Save one completed question-and-answer interaction.
    """
    sources_json = json.dumps(
        sources,
        ensure_ascii=False,
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO chat_history (
                document_id,
                question,
                answer,
                sources_json,
                response_time_ms,
                input_tokens,
                output_tokens,
                total_tokens,
                model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                question,
                answer,
                sources_json,
                response_time_ms,
                input_tokens,
                output_tokens,
                total_tokens,
                model_name,
            ),
        )

        chat_id = cursor.lastrowid

        if chat_id is None:
            raise RuntimeError(
                "The chat message could not be saved."
            )

        return chat_id


def get_chat_history(
    document_id: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Return the most recent chat messages for one document.

    The messages are returned in chronological order.
    """
    safe_limit = max(
        1,
        min(limit, 100),
    )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                document_id,
                question,
                answer,
                sources_json,
                response_time_ms,
                input_tokens,
                output_tokens,
                total_tokens,
                model_name,
                created_at
            FROM chat_history
            WHERE document_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                document_id,
                safe_limit,
            ),
        ).fetchall()

    history = []

    for row in reversed(rows):
        item = dict(row)

        sources_json = item.pop(
            "sources_json",
            "[]",
        )

        try:
            item["sources"] = json.loads(
                sources_json
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            item["sources"] = []

        history.append(item)

    return history