from pathlib import Path
from typing import Any

import chromadb

from app.chunking_service import TextChunk


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "document_chunks"


class VectorStoreError(Exception):
    """Raised when a vector database operation fails."""


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )


def get_collection():
    try:
        client = get_chroma_client()

        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": (
                    "Embeddings for uploaded document chunks"
                ),
                "hnsw:space": "cosine",
            },
        )

    except Exception as exc:
        raise VectorStoreError(
            "The vector collection could not be opened."
        ) from exc


def store_document_embeddings(
    *,
    document_id: int,
    original_filename: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> int:
    if not chunks:
        raise VectorStoreError(
            "There are no chunks to store."
        )

    if len(chunks) != len(embeddings):
        raise VectorStoreError(
            "The number of chunks and embeddings does not match."
        )

    ids = [
        f"document-{document_id}-chunk-{chunk.chunk_id}"
        for chunk in chunks
    ]

    documents = [
        chunk.text
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": document_id,
            "filename": original_filename,
            "chunk_number": chunk.chunk_id,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "character_count": len(chunk.text),
        }
        for chunk in chunks
    ]

    try:
        collection = get_collection()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    except Exception as exc:
        raise VectorStoreError(
            "The embeddings could not be stored."
        ) from exc

    return len(ids)


def delete_document_embeddings(
    document_id: int,
) -> None:
    try:
        collection = get_collection()

        collection.delete(
            where={
                "document_id": document_id
            }
        )

    except Exception as exc:
        raise VectorStoreError(
            "The document embeddings could not be deleted."
        ) from exc


def get_vector_count() -> int:
    try:
        collection = get_collection()
        return collection.count()

    except Exception as exc:
        raise VectorStoreError(
            "The vector count could not be retrieved."
        ) from exc

def delete_document_embeddings(
    document_id: int,
) -> int:
    """
    Delete every vector belonging to one document.

    Returns the number of vectors found before deletion.
    """
    try:
        collection = get_collection()

        existing = collection.get(
            where={
                "document_id": document_id
            }
        )

        vector_ids = existing.get(
            "ids",
            []
        )

        if not vector_ids:
            return 0

        collection.delete(
            ids=vector_ids
        )

        return len(vector_ids)

    except Exception as exc:
        raise VectorStoreError(
            "The document vectors could not be deleted."
        ) from exc

def search_document_chunks(
    *,
    document_id: int,
    query_embedding: list[float],
    number_of_results: int = 3,
) -> list[dict[str, Any]]:
    if number_of_results <= 0:
        raise VectorStoreError(
            "The number of results must be greater than zero."
        )

    try:
        collection = get_collection()

        document_vector_count = collection.get(
            where={
                "document_id": document_id
            }
        )

        available_count = len(
            document_vector_count.get("ids", [])
        )

        if available_count == 0:
            return []

        result_count = min(
            number_of_results,
            available_count,
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=result_count,
            where={
                "document_id": document_id
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except Exception as exc:
        raise VectorStoreError(
            "The semantic search could not be completed."
        ) from exc

    ids = results.get("ids", [[]])[0]
    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    search_results = []

    for vector_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        similarity_score = max(
            0.0,
            min(
                1.0,
                1.0 - float(distance),
            ),
        )

        search_results.append(
            {
                "vector_id": vector_id,
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
                "similarity_score": similarity_score,
                "similarity_percentage": round(
                    similarity_score * 100,
                    2,
                ),
            }
        )

    return search_results