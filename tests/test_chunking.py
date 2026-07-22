import pytest

from app.chunking_service import (
    clean_text,
    split_text_into_chunks,
)


def test_clean_text_removes_extra_blank_lines():
    text = """
    First paragraph.


    Second paragraph.
    """

    cleaned = clean_text(text)

    assert cleaned == (
        "First paragraph.\n\n"
        "Second paragraph."
    )


def test_short_text_creates_one_chunk():
    text = "This is a short document."

    chunks = split_text_into_chunks(
        text,
        chunk_size=700,
        chunk_overlap=100,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == 1
    assert chunks[0].text == text
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == len(text)


def test_long_text_creates_multiple_chunks():
    text = "This is a sentence. " * 150

    chunks = split_text_into_chunks(
        text,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.text
        assert chunk.end_index > chunk.start_index
        assert len(chunk.text) <= 300


def test_chunks_have_sequential_ids():
    text = "Document content. " * 100

    chunks = split_text_into_chunks(
        text,
        chunk_size=200,
        chunk_overlap=40,
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert chunk_ids == list(
        range(1, len(chunks) + 1)
    )


def test_invalid_chunk_size_raises_error():
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        split_text_into_chunks(
            "Example text",
            chunk_size=0,
            chunk_overlap=0,
        )


def test_negative_overlap_raises_error():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        split_text_into_chunks(
            "Example text",
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(
        ValueError,
        match="smaller than chunk size",
    ):
        split_text_into_chunks(
            "Example text",
            chunk_size=100,
            chunk_overlap=100,
        )


def test_empty_text_returns_no_chunks():
    chunks = split_text_into_chunks(
        "   \n\n   ",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks == []