from functools import lru_cache

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated."""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.
    """
    try:
        return SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )
    except Exception as exc:
        raise EmbeddingError(
            "The embedding model could not be loaded."
        ) from exc


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate one normalized embedding per text.
    """
    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text.strip()
    ]

    if not cleaned_texts:
        return []

    try:
        model = get_embedding_model()

        embeddings = model.encode(
            cleaned_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    except Exception as exc:
        raise EmbeddingError(
            "Embeddings could not be generated."
        ) from exc

    return embeddings.tolist()


def generate_query_embedding(
    query: str,
) -> list[float]:
    """
    Generate an embedding for a user question.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise EmbeddingError(
            "The query cannot be empty."
        )

    embeddings = generate_embeddings(
        [cleaned_query]
    )

    if not embeddings:
        raise EmbeddingError(
            "The query embedding could not be generated."
        )

    return embeddings[0]