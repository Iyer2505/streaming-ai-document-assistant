import io
from pathlib import Path


def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"AI Document Assistant" in response.data


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"
    assert data["vector_count"] == 1
    assert "request_id" in data


def test_upload_requires_file(client):
    response = client.post(
        "/upload",
        data={},
    )

    assert response.status_code == 400
    assert b"No document was included" in response.data


def test_upload_rejects_unsupported_file(client):
    response = client.post(
        "/upload",
        data={
            "document": (
                io.BytesIO(b"image content"),
                "image.jpg",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"Unsupported file type" in response.data


def test_successful_txt_upload(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "save_document",
        lambda **kwargs: 99,
    )

    monkeypatch.setattr(
        app_module,
        "get_chat_history",
        lambda **kwargs: [],
    )

    response = client.post(
        "/upload",
        data={
            "document": (
                io.BytesIO(
                    b"The assistant allows users "
                    b"to upload documents."
                ),
                "sample.txt",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Document ID:" in response.data
    assert b"99" in response.data
    assert b"Stored embeddings:" in response.data


def test_search_rejects_empty_question(client):
    response = client.post(
        "/search",
        json={
            "document_id": 1,
            "question": "",
        },
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Please enter a question."
    assert data["status_code"] == 400
    assert "request_id" in data


def test_search_rejects_invalid_document_id(client):
    response = client.post(
        "/search",
        json={
            "document_id": "invalid",
            "question": "What is this?",
        },
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "The document ID is invalid."
    assert "request_id" in data


def test_successful_semantic_search(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: {
            "id": document_id,
            "original_filename": "sample.txt",
        },
    )

    monkeypatch.setattr(
        app_module,
        "retrieve_search_results",
        lambda **kwargs: [
            {
                "text": (
                    "The assistant supports "
                    "document questions."
                ),
                "metadata": {
                    "filename": "sample.txt",
                    "chunk_number": 1,
                    "start_index": 0,
                    "end_index": 50,
                },
                "similarity_percentage": 75.5,
            }
        ],
    )

    response = client.post(
        "/search",
        json={
            "document_id": 1,
            "question": "What does the assistant do?",
        },
    )

    assert response.status_code == 200

    data = response.get_json()

    assert len(data["search_results"]) == 1

    assert (
        data["search_results"][0][
            "similarity_percentage"
        ]
        == 75.5
    )

    assert "request_id" in data


def test_chat_history_endpoint(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: {
            "id": document_id,
            "original_filename": "sample.txt",
        },
    )

    monkeypatch.setattr(
        app_module,
        "get_chat_history",
        lambda **kwargs: [
            {
                "id": 1,
                "question": "Test question",
                "answer": "Test answer",
                "sources": [],
            }
        ],
    )

    response = client.get(
        "/chat-history/1"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["document_id"] == 1
    assert len(data["history"]) == 1
    assert data["history"][0]["answer"] == "Test answer"
    assert "request_id" in data


def test_document_dashboard_loads(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_all_documents",
        lambda limit=100: [],
    )

    response = client.get(
        "/documents"
    )

    assert response.status_code == 200
    assert b"Document Dashboard" in response.data
    assert b"Uploaded documents" in response.data


def test_document_dashboard_lists_documents(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_all_documents",
        lambda limit=100: [
            {
                "id": 12,
                "original_filename": "report.pdf",
                "stored_filename": "unique_report.pdf",
                "file_type": ".pdf",
                "file_path": "/uploads/unique_report.pdf",
                "word_count": 450,
                "character_count": 2800,
                "chunk_count": 5,
                "created_at": "2026-07-22 15:00:00",
            }
        ],
    )

    response = client.get(
        "/documents"
    )

    assert response.status_code == 200
    assert b"report.pdf" in response.data
    assert b"Document ID:" in response.data
    assert b"450" in response.data
    assert b"5" in response.data
    assert b"Open document" in response.data
    assert b"Delete" in response.data


def test_open_existing_document(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: {
            "id": document_id,
            "original_filename": "existing.txt",
            "stored_filename": "stored_existing.txt",
            "file_type": ".txt",
            "file_path": "/uploads/stored_existing.txt",
            "word_count": 20,
            "character_count": 140,
            "chunk_count": 2,
            "created_at": "2026-07-22 15:00:00",
        },
    )

    monkeypatch.setattr(
        app_module,
        "get_document_chunks",
        lambda document_id: [
            {
                "id": 1,
                "document_id": document_id,
                "chunk_number": 1,
                "chunk_text": (
                    "The first existing document chunk."
                ),
                "start_index": 0,
                "end_index": 35,
                "character_count": 35,
                "created_at": "2026-07-22 15:00:00",
            },
            {
                "id": 2,
                "document_id": document_id,
                "chunk_number": 2,
                "chunk_text": (
                    "The second existing document chunk."
                ),
                "start_index": 30,
                "end_index": 66,
                "character_count": 36,
                "created_at": "2026-07-22 15:00:00",
            },
        ],
    )

    monkeypatch.setattr(
        app_module,
        "get_chat_history",
        lambda **kwargs: [
            {
                "id": 1,
                "document_id": 7,
                "question": "What is this?",
                "answer": "An existing document.",
                "sources": [],
                "response_time_ms": 300,
                "total_tokens": 45,
                "model_name": "test-model",
                "created_at": "2026-07-22 15:01:00",
            }
        ],
    )

    response = client.get(
        "/documents/7"
    )

    assert response.status_code == 200
    assert b"Existing document opened successfully" in response.data
    assert b"existing.txt" in response.data
    assert b"Document ID:" in response.data
    assert b"The first existing document chunk." in response.data
    assert b"What is this?" in response.data
    assert b"An existing document." in response.data


def test_open_missing_document(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: None,
    )

    response = client.get(
        "/documents/999"
    )

    assert response.status_code == 404
    assert b"could not be found" in response.data


def test_delete_document_successfully(
    client,
    monkeypatch,
    tmp_path,
    app_module,
):
    uploaded_file = (
        tmp_path / "delete_me.txt"
    )

    uploaded_file.write_text(
        "Temporary test document.",
        encoding="utf-8",
    )

    deletion_calls = {
        "vectors": None,
        "database": None,
    }

    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: {
            "id": document_id,
            "original_filename": "delete_me.txt",
            "stored_filename": "delete_me.txt",
            "file_type": ".txt",
            "file_path": str(uploaded_file),
            "word_count": 3,
            "character_count": 24,
            "chunk_count": 1,
            "created_at": "2026-07-22 15:00:00",
        },
    )

    def fake_delete_vectors(document_id):
        deletion_calls["vectors"] = (
            document_id
        )

        return 3

    def fake_delete_database_document(
        document_id,
    ):
        deletion_calls["database"] = (
            document_id
        )

    monkeypatch.setattr(
        app_module,
        "delete_document_embeddings",
        fake_delete_vectors,
    )

    monkeypatch.setattr(
        app_module,
        "delete_document",
        fake_delete_database_document,
    )

    response = client.post(
        "/documents/15/delete",
        json={},
    )

    assert response.status_code == 200

    data = response.get_json()

    assert (
        data["message"]
        == "The document was deleted successfully."
    )

    assert data["document_id"] == 15
    assert data["deleted_vectors"] == 3
    assert "request_id" in data

    assert deletion_calls["vectors"] == 15
    assert deletion_calls["database"] == 15

    assert not uploaded_file.exists()


def test_delete_missing_document(
    client,
    monkeypatch,
    app_module,
):
    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: None,
    )

    response = client.post(
        "/documents/999/delete",
        json={},
    )

    assert response.status_code == 404

    data = response.get_json()

    assert (
        data["error"]
        == "The document could not be found."
    )

    assert data["status_code"] == 404
    assert "request_id" in data


def test_delete_document_handles_vector_error(
    client,
    monkeypatch,
    tmp_path,
    app_module,
):
    uploaded_file = (
        tmp_path / "vector_error.txt"
    )

    uploaded_file.write_text(
        "Temporary content.",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app_module,
        "get_document",
        lambda document_id: {
            "id": document_id,
            "original_filename": "vector_error.txt",
            "stored_filename": "vector_error.txt",
            "file_type": ".txt",
            "file_path": str(uploaded_file),
            "word_count": 2,
            "character_count": 18,
            "chunk_count": 1,
            "created_at": "2026-07-22 15:00:00",
        },
    )

    def raise_vector_error(
        document_id,
    ):
        raise app_module.VectorStoreError(
            "The document vectors could not be deleted."
        )

    monkeypatch.setattr(
        app_module,
        "delete_document_embeddings",
        raise_vector_error,
    )

    response = client.post(
        "/documents/20/delete",
        json={},
    )

    assert response.status_code == 500

    data = response.get_json()

    assert (
        data["error"]
        == "The document vectors could not be deleted."
    )

    assert uploaded_file.exists()