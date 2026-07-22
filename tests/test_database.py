from app.chunking_service import TextChunk
import app.database as database


def configure_test_database(
    monkeypatch,
    tmp_path,
):
    test_database_path = (
        tmp_path / "test_document_assistant.db"
    )

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        test_database_path,
    )

    database.initialise_database()

    return test_database_path


def create_test_chunks():
    return [
        TextChunk(
            chunk_id=1,
            text="First document chunk.",
            start_index=0,
            end_index=21,
        ),
        TextChunk(
            chunk_id=2,
            text="Second document chunk.",
            start_index=15,
            end_index=37,
        ),
    ]


def test_save_and_get_document(
    monkeypatch,
    tmp_path,
):
    configure_test_database(
        monkeypatch,
        tmp_path,
    )

    document_id = database.save_document(
        original_filename="sample.txt",
        stored_filename="unique_sample.txt",
        file_type=".txt",
        file_path="/temporary/sample.txt",
        word_count=6,
        character_count=44,
        chunks=create_test_chunks(),
    )

    document = database.get_document(
        document_id
    )

    assert document is not None
    assert document["id"] == document_id
    assert (
        document["original_filename"]
        == "sample.txt"
    )
    assert document["chunk_count"] == 2


def test_document_chunks_are_saved(
    monkeypatch,
    tmp_path,
):
    configure_test_database(
        monkeypatch,
        tmp_path,
    )

    document_id = database.save_document(
        original_filename="sample.txt",
        stored_filename="saved_sample.txt",
        file_type=".txt",
        file_path="/temporary/sample.txt",
        word_count=6,
        character_count=44,
        chunks=create_test_chunks(),
    )

    chunks = database.get_document_chunks(
        document_id
    )

    assert len(chunks) == 2
    assert chunks[0]["chunk_number"] == 1
    assert (
        chunks[0]["chunk_text"]
        == "First document chunk."
    )


def test_save_and_get_chat_history(
    monkeypatch,
    tmp_path,
):
    configure_test_database(
        monkeypatch,
        tmp_path,
    )

    document_id = database.save_document(
        original_filename="sample.txt",
        stored_filename="chat_sample.txt",
        file_type=".txt",
        file_path="/temporary/sample.txt",
        word_count=6,
        character_count=44,
        chunks=create_test_chunks(),
    )

    chat_id = database.save_chat_message(
        document_id=document_id,
        question="What is this document?",
        answer="It is a sample document.",
        sources=[
            {
                "filename": "sample.txt",
                "chunk_number": 1,
                "similarity_percentage": 80.0,
                "text": "First document chunk.",
            }
        ],
        response_time_ms=1200,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        model_name="test-model",
    )

    history = database.get_chat_history(
        document_id
    )

    assert chat_id == 1
    assert len(history) == 1

    item = history[0]

    assert (
        item["question"]
        == "What is this document?"
    )
    assert (
        item["answer"]
        == "It is a sample document."
    )
    assert item["total_tokens"] == 120
    assert item["sources"][0][
        "filename"
    ] == "sample.txt"


def test_deleting_document_cascades(
    monkeypatch,
    tmp_path,
):
    configure_test_database(
        monkeypatch,
        tmp_path,
    )

    document_id = database.save_document(
        original_filename="sample.txt",
        stored_filename="delete_sample.txt",
        file_type=".txt",
        file_path="/temporary/sample.txt",
        word_count=6,
        character_count=44,
        chunks=create_test_chunks(),
    )

    database.save_chat_message(
        document_id=document_id,
        question="Test?",
        answer="Test answer.",
        sources=[],
        response_time_ms=100,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        model_name="test-model",
    )

    database.delete_document(
        document_id
    )

    assert (
        database.get_document(document_id)
        is None
    )

    assert (
        database.get_document_chunks(
            document_id
        )
        == []
    )

    assert (
        database.get_chat_history(
            document_id
        )
        == []
    )