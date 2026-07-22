import json
import time
from pathlib import Path
from uuid import uuid4

from flask import (
    Flask,
    Response,
    g,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from werkzeug.exceptions import (
    HTTPException,
    RequestEntityTooLarge,
)
from werkzeug.utils import secure_filename

from app.chunking_service import (
    split_text_into_chunks,
)
from app.database import (
    delete_document,
    get_all_documents,
    get_chat_history,
    get_document,
    get_document_chunks,
    initialise_database,
    save_chat_message,
    save_document,
)
from app.document_service import (
    DocumentExtractionError,
    extract_text,
)
from app.embedding_service import (
    EmbeddingError,
    generate_embeddings,
    generate_query_embedding,
)
from app.llm_service import (
    LLMServiceError,
    stream_grounded_answer,
)
from app.logging_service import (
    configure_logging,
)
from app.vector_store import (
    VectorStoreError,
    delete_document_embeddings,
    get_vector_count,
    search_document_chunks,
    store_document_embeddings,
)


app = Flask(__name__)

configure_logging(
    app.logger
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {
    "pdf",
    "docx",
    "txt",
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

initialise_database()


@app.before_request
def start_request_tracking() -> None:
    """
    Assign a unique request ID and begin timing the request.
    """
    incoming_request_id = request.headers.get(
        "X-Request-ID",
        "",
    ).strip()

    g.request_id = (
        incoming_request_id
        or uuid4().hex
    )

    g.request_started_at = (
        time.perf_counter()
    )

    app.logger.info(
        "Request started",
        extra={
            "request_id": g.request_id,
            "method": request.method,
            "path": request.path,
        },
    )


@app.after_request
def finish_request_tracking(
    response,
):
    """
    Log the completed request and attach its request ID.
    """
    request_id = getattr(
        g,
        "request_id",
        "unknown",
    )

    started_at = getattr(
        g,
        "request_started_at",
        None,
    )

    duration_ms = None

    if started_at is not None:
        duration_ms = round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            2,
        )

    response.headers[
        "X-Request-ID"
    ] = request_id

    app.logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.path,
            "status_code": (
                response.status_code
            ),
            "duration_ms": duration_ms,
        },
    )

    return response


def current_request_id() -> str:
    """
    Return the current request ID.
    """
    return getattr(
        g,
        "request_id",
        "unknown",
    )


def is_api_request() -> bool:
    """
    Determine whether the current route expects JSON.
    """
    return (
        request.path == "/search"
        or request.path == "/stream-answer"
        or request.path.startswith(
            "/chat-history/"
        )
        or (
            request.path.startswith(
                "/documents/"
            )
            and request.path.endswith(
                "/delete"
            )
        )
        or request.is_json
    )


def api_error(
    message: str,
    status_code: int,
):
    """
    Return a consistent JSON error response.
    """
    return jsonify(
        {
            "error": message,
            "status_code": status_code,
            "request_id": (
                current_request_id()
            ),
        }
    ), status_code


def allowed_file(
    filename: str,
) -> bool:
    """
    Return True if the uploaded filename has a supported extension.
    """
    return (
        "." in filename
        and filename.rsplit(
            ".",
            1,
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


def retrieve_search_results(
    *,
    document_id: int,
    question: str,
) -> list[dict]:
    """
    Generate the question embedding and retrieve relevant chunks.
    """
    query_embedding = (
        generate_query_embedding(
            question
        )
    )

    return search_document_chunks(
        document_id=document_id,
        query_embedding=query_embedding,
        number_of_results=3,
    )


@app.get("/")
def home():
    """
    Display the main upload page.
    """
    return render_template(
        "index.html"
    )


@app.get("/documents")
def document_dashboard():
    """
    Display all previously uploaded documents.
    """
    documents = get_all_documents(
        limit=100
    )

    return render_template(
        "dashboard.html",
        documents=documents,
    )


@app.get(
    "/documents/<int:document_id>"
)
def open_document(
    document_id: int,
):
    """
    Reopen an existing document and continue its conversation.
    """
    document = get_document(
        document_id
    )

    if document is None:
        return render_template(
            "index.html",
            error=(
                "The requested document "
                "could not be found."
            ),
            request_id=(
                current_request_id()
            ),
        ), 404

    document_chunks = (
        get_document_chunks(
            document_id
        )
    )

    chunk_previews = [
        {
            "chunk_id": (
                chunk["chunk_number"]
            ),
            "text": (
                chunk["chunk_text"]
            ),
            "character_count": (
                chunk["character_count"]
            ),
            "start_index": (
                chunk["start_index"]
            ),
            "end_index": (
                chunk["end_index"]
            ),
        }
        for chunk in document_chunks[:5]
    ]

    text_preview = "\n\n".join(
        chunk["chunk_text"]
        for chunk in document_chunks[:3]
    )

    if len(text_preview) > 1500:
        text_preview = (
            text_preview[:1500]
            + "\n\n..."
        )

    chat_history_items = (
        get_chat_history(
            document_id=document_id,
            limit=20,
        )
    )

    try:
        total_vector_count = (
            get_vector_count()
        )

    except VectorStoreError:
        total_vector_count = 0

        app.logger.exception(
            "Vector count failed while opening document",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
            },
        )

    app.logger.info(
        "Existing document opened",
        extra={
            "request_id": (
                current_request_id()
            ),
            "document_id": (
                document_id
            ),
        },
    )

    return render_template(
        "index.html",
        success=(
            "Existing document opened successfully."
        ),
        document_id=(
            document["id"]
        ),
        uploaded_filename=(
            document[
                "original_filename"
            ]
        ),
        character_count=(
            document[
                "character_count"
            ]
        ),
        word_count=(
            document[
                "word_count"
            ]
        ),
        text_preview=(
            text_preview
        ),
        chunk_count=(
            document[
                "chunk_count"
            ]
        ),
        chunk_previews=(
            chunk_previews
        ),
        stored_embedding_count=(
            document[
                "chunk_count"
            ]
        ),
        embedding_dimension=384,
        total_vector_count=(
            total_vector_count
        ),
        chat_history=(
            chat_history_items
        ),
    )


@app.post(
    "/documents/<int:document_id>/delete"
)
def remove_document(
    document_id: int,
):
    """
    Delete a document and its related data.

    This deletes:
    - ChromaDB embeddings
    - SQLite document record
    - SQLite chunks through cascade deletion
    - SQLite chat history through cascade deletion
    - Uploaded file from disk
    """
    document = get_document(
        document_id
    )

    if document is None:
        return api_error(
            "The document could not be found.",
            404,
        )

    file_path = Path(
        document["file_path"]
    )

    try:
        deleted_vector_count = (
            delete_document_embeddings(
                document_id
            )
        )

        delete_document(
            document_id
        )

        file_path.unlink(
            missing_ok=True
        )

        app.logger.info(
            "Document deleted successfully",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
            },
        )

        return jsonify(
            {
                "message": (
                    "The document was deleted successfully."
                ),
                "document_id": (
                    document_id
                ),
                "deleted_vectors": (
                    deleted_vector_count
                ),
                "request_id": (
                    current_request_id()
                ),
            }
        )

    except VectorStoreError as exc:
        app.logger.exception(
            "Document vector deletion failed",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
            },
        )

        return api_error(
            str(exc),
            500,
        )

    except Exception:
        app.logger.exception(
            "Document deletion failed",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
            },
        )

        return api_error(
            "The document could not be deleted.",
            500,
        )


@app.post("/upload")
def upload_document():
    """
    Upload, extract, chunk, save and embed a document.
    """
    if "document" not in request.files:
        return render_template(
            "index.html",
            error=(
                "No document was included "
                "in the request."
            ),
            request_id=(
                current_request_id()
            ),
        ), 400

    uploaded_file = request.files[
        "document"
    ]

    if uploaded_file.filename == "":
        return render_template(
            "index.html",
            error=(
                "Please select a document "
                "before uploading."
            ),
            request_id=(
                current_request_id()
            ),
        ), 400

    if not allowed_file(
        uploaded_file.filename
    ):
        return render_template(
            "index.html",
            error=(
                "Unsupported file type. "
                "Please upload PDF, DOCX or TXT."
            ),
            request_id=(
                current_request_id()
            ),
        ), 400

    safe_original_name = secure_filename(
        uploaded_file.filename
    )

    if not safe_original_name:
        return render_template(
            "index.html",
            error=(
                "The filename is invalid."
            ),
            request_id=(
                current_request_id()
            ),
        ), 400

    unique_filename = (
        f"{uuid4().hex}_"
        f"{safe_original_name}"
    )

    save_path = (
        app.config["UPLOAD_FOLDER"]
        / unique_filename
    )

    uploaded_file.save(
        save_path
    )

    try:
        extracted_text = extract_text(
            save_path
        )

    except DocumentExtractionError as exc:
        save_path.unlink(
            missing_ok=True
        )

        app.logger.warning(
            "Document extraction failed",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "path": (
                    request.path
                ),
            },
        )

        return render_template(
            "index.html",
            error=str(exc),
            request_id=(
                current_request_id()
            ),
        ), 400

    character_count = len(
        extracted_text
    )

    word_count = len(
        extracted_text.split()
    )

    chunks = split_text_into_chunks(
        extracted_text,
        chunk_size=700,
        chunk_overlap=100,
    )

    chunk_count = len(
        chunks
    )

    if chunk_count == 0:
        save_path.unlink(
            missing_ok=True
        )

        return render_template(
            "index.html",
            error=(
                "No chunks could be created "
                "from this document."
            ),
            request_id=(
                current_request_id()
            ),
        ), 400

    try:
        document_id = save_document(
            original_filename=(
                safe_original_name
            ),
            stored_filename=(
                unique_filename
            ),
            file_type=(
                save_path.suffix.lower()
            ),
            file_path=str(
                save_path
            ),
            word_count=(
                word_count
            ),
            character_count=(
                character_count
            ),
            chunks=(
                chunks
            ),
        )

    except Exception:
        save_path.unlink(
            missing_ok=True
        )

        app.logger.exception(
            "Failed to save document to SQLite",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "path": (
                    request.path
                ),
            },
        )

        return render_template(
            "index.html",
            error=(
                "The document was processed, "
                "but it could not be saved."
            ),
            request_id=(
                current_request_id()
            ),
        ), 500

    try:
        chunk_texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = generate_embeddings(
            chunk_texts
        )

        stored_embedding_count = (
            store_document_embeddings(
                document_id=(
                    document_id
                ),
                original_filename=(
                    safe_original_name
                ),
                chunks=(
                    chunks
                ),
                embeddings=(
                    embeddings
                ),
            )
        )

        total_vector_count = (
            get_vector_count()
        )

    except (
        EmbeddingError,
        VectorStoreError,
    ) as exc:
        delete_document(
            document_id
        )

        save_path.unlink(
            missing_ok=True
        )

        app.logger.exception(
            "Embedding or vector storage failed",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
                "path": (
                    request.path
                ),
            },
        )

        return render_template(
            "index.html",
            error=str(exc),
            request_id=(
                current_request_id()
            ),
        ), 500

    chunk_previews = [
        {
            "chunk_id": (
                chunk.chunk_id
            ),
            "text": (
                chunk.text
            ),
            "character_count": len(
                chunk.text
            ),
            "start_index": (
                chunk.start_index
            ),
            "end_index": (
                chunk.end_index
            ),
        }
        for chunk in chunks[:5]
    ]

    preview_limit = 1500

    text_preview = extracted_text[
        :preview_limit
    ]

    if len(extracted_text) > preview_limit:
        text_preview += "\n\n..."

    embedding_dimension = (
        len(embeddings[0])
        if embeddings
        else 0
    )

    chat_history_items = (
        get_chat_history(
            document_id=(
                document_id
            ),
            limit=20,
        )
    )

    app.logger.info(
        "Document processed successfully",
        extra={
            "request_id": (
                current_request_id()
            ),
            "document_id": (
                document_id
            ),
        },
    )

    return render_template(
        "index.html",
        success=(
            "Document uploaded, processed, "
            "chunked and embedded successfully."
        ),
        document_id=(
            document_id
        ),
        uploaded_filename=(
            safe_original_name
        ),
        character_count=(
            character_count
        ),
        word_count=(
            word_count
        ),
        text_preview=(
            text_preview
        ),
        chunk_count=(
            chunk_count
        ),
        chunk_previews=(
            chunk_previews
        ),
        stored_embedding_count=(
            stored_embedding_count
        ),
        embedding_dimension=(
            embedding_dimension
        ),
        total_vector_count=(
            total_vector_count
        ),
        chat_history=(
            chat_history_items
        ),
    )


@app.post("/search")
def semantic_search():
    """
    Retrieve relevant chunks for a question.
    """
    data = request.get_json(
        silent=True
    ) or {}

    document_id_value = data.get(
        "document_id"
    )

    question = str(
        data.get(
            "question",
            "",
        )
    ).strip()

    try:
        document_id = int(
            document_id_value
        )

    except (
        TypeError,
        ValueError,
    ):
        return api_error(
            "The document ID is invalid.",
            400,
        )

    if not question:
        return api_error(
            "Please enter a question.",
            400,
        )

    if len(question) > 1000:
        return api_error(
            (
                "The question is too long. "
                "Use fewer than 1,000 characters."
            ),
            400,
        )

    document = get_document(
        document_id
    )

    if document is None:
        return api_error(
            "The document could not be found.",
            404,
        )

    try:
        search_results = (
            retrieve_search_results(
                document_id=(
                    document_id
                ),
                question=(
                    question
                ),
            )
        )

    except (
        EmbeddingError,
        VectorStoreError,
    ) as exc:
        app.logger.exception(
            "Semantic search failed",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
                "path": (
                    request.path
                ),
            },
        )

        return api_error(
            str(exc),
            500,
        )

    if not search_results:
        return api_error(
            (
                "No relevant document "
                "chunks were found."
            ),
            404,
        )

    return jsonify(
        {
            "document_id": (
                document_id
            ),
            "question": (
                question
            ),
            "search_results": (
                search_results
            ),
            "request_id": (
                current_request_id()
            ),
        }
    )


@app.post("/stream-answer")
def stream_answer():
    """
    Stream a grounded answer as newline-delimited JSON.
    """
    data = request.get_json(
        silent=True
    ) or {}

    document_id_value = data.get(
        "document_id"
    )

    question = str(
        data.get(
            "question",
            "",
        )
    ).strip()

    try:
        document_id = int(
            document_id_value
        )

    except (
        TypeError,
        ValueError,
    ):
        return api_error(
            "The document ID is invalid.",
            400,
        )

    if not question:
        return api_error(
            "Please enter a question.",
            400,
        )

    if len(question) > 1000:
        return api_error(
            (
                "The question is too long. "
                "Use fewer than 1,000 characters."
            ),
            400,
        )

    document = get_document(
        document_id
    )

    if document is None:
        return api_error(
            "The document could not be found.",
            404,
        )

    try:
        search_results = (
            retrieve_search_results(
                document_id=(
                    document_id
                ),
                question=(
                    question
                ),
            )
        )

    except (
        EmbeddingError,
        VectorStoreError,
    ) as exc:
        app.logger.exception(
            "Retrieval failed before streaming",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "document_id": (
                    document_id
                ),
                "path": (
                    request.path
                ),
            },
        )

        return api_error(
            str(exc),
            500,
        )

    if not search_results:
        return api_error(
            "No document chunks were found.",
            404,
        )

    request_id = current_request_id()

    def generate():
        start_time = time.perf_counter()

        answer_parts: list[str] = []

        input_tokens = None
        output_tokens = None
        total_tokens = None
        model_name = ""

        try:
            for stream_event in (
                stream_grounded_answer(
                    question=(
                        question
                    ),
                    search_results=(
                        search_results
                    ),
                )
            ):
                event_type = (
                    stream_event.get(
                        "type"
                    )
                )

                if event_type == "delta":
                    text_delta = str(
                        stream_event.get(
                            "text",
                            "",
                        )
                    )

                    if not text_delta:
                        continue

                    answer_parts.append(
                        text_delta
                    )

                    yield (
                        json.dumps(
                            {
                                "type": (
                                    "delta"
                                ),
                                "text": (
                                    text_delta
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                elif event_type == "completed":
                    input_tokens = (
                        stream_event.get(
                            "input_tokens"
                        )
                    )

                    output_tokens = (
                        stream_event.get(
                            "output_tokens"
                        )
                    )

                    total_tokens = (
                        stream_event.get(
                            "total_tokens"
                        )
                    )

                    model_name = str(
                        stream_event.get(
                            "model_name",
                            "",
                        )
                    )

            complete_answer = "".join(
                answer_parts
            ).strip()

            if not complete_answer:
                raise LLMServiceError(
                    "The model returned an empty answer."
                )

            response_time_ms = round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            source_records = []

            for result in search_results:
                metadata = (
                    result.get(
                        "metadata"
                    )
                    or {}
                )

                source_records.append(
                    {
                        "filename": (
                            metadata.get(
                                "filename"
                            )
                        ),
                        "chunk_number": (
                            metadata.get(
                                "chunk_number"
                            )
                        ),
                        "start_index": (
                            metadata.get(
                                "start_index"
                            )
                        ),
                        "end_index": (
                            metadata.get(
                                "end_index"
                            )
                        ),
                        "similarity_percentage": (
                            result.get(
                                "similarity_percentage"
                            )
                        ),
                        "text": (
                            result.get(
                                "text",
                                "",
                            )
                        ),
                    }
                )

            chat_id = save_chat_message(
                document_id=(
                    document_id
                ),
                question=(
                    question
                ),
                answer=(
                    complete_answer
                ),
                sources=(
                    source_records
                ),
                response_time_ms=(
                    response_time_ms
                ),
                input_tokens=(
                    input_tokens
                ),
                output_tokens=(
                    output_tokens
                ),
                total_tokens=(
                    total_tokens
                ),
                model_name=(
                    model_name
                ),
            )

            app.logger.info(
                "Chat interaction saved",
                extra={
                    "request_id": (
                        request_id
                    ),
                    "document_id": (
                        document_id
                    ),
                    "chat_id": (
                        chat_id
                    ),
                    "duration_ms": (
                        response_time_ms
                    ),
                },
            )

            yield (
                json.dumps(
                    {
                        "type": (
                            "completed"
                        ),
                        "chat_id": (
                            chat_id
                        ),
                        "response_time_ms": (
                            response_time_ms
                        ),
                        "input_tokens": (
                            input_tokens
                        ),
                        "output_tokens": (
                            output_tokens
                        ),
                        "total_tokens": (
                            total_tokens
                        ),
                        "model_name": (
                            model_name
                        ),
                        "request_id": (
                            request_id
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        except LLMServiceError as exc:
            app.logger.exception(
                "Streaming answer failed",
                extra={
                    "request_id": (
                        request_id
                    ),
                    "document_id": (
                        document_id
                    ),
                    "path": (
                        request.path
                    ),
                },
            )

            yield (
                json.dumps(
                    {
                        "type": (
                            "error"
                        ),
                        "message": (
                            str(exc)
                        ),
                        "request_id": (
                            request_id
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        except Exception:
            app.logger.exception(
                "Unexpected streaming error",
                extra={
                    "request_id": (
                        request_id
                    ),
                    "document_id": (
                        document_id
                    ),
                    "path": (
                        request.path
                    ),
                },
            )

            yield (
                json.dumps(
                    {
                        "type": (
                            "error"
                        ),
                        "message": (
                            "The answer could not "
                            "be completed or saved."
                        ),
                        "request_id": (
                            request_id
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    response = Response(
        stream_with_context(
            generate()
        ),
        content_type=(
            "application/x-ndjson; "
            "charset=utf-8"
        ),
    )

    response.headers[
        "Cache-Control"
    ] = "no-cache"

    response.headers[
        "X-Accel-Buffering"
    ] = "no"

    return response


@app.get(
    "/chat-history/<int:document_id>"
)
def chat_history(
    document_id: int,
):
    """
    Return saved chat history for one document.
    """
    document = get_document(
        document_id
    )

    if document is None:
        return api_error(
            "The document could not be found.",
            404,
        )

    history = get_chat_history(
        document_id=(
            document_id
        ),
        limit=20,
    )

    return jsonify(
        {
            "document_id": (
                document_id
            ),
            "filename": (
                document[
                    "original_filename"
                ]
            ),
            "history": (
                history
            ),
            "request_id": (
                current_request_id()
            ),
        }
    )


@app.errorhandler(
    RequestEntityTooLarge
)
def handle_large_file(
    error,
):
    """
    Handle uploads larger than 10 MB.
    """
    return render_template(
        "index.html",
        error=(
            "The document is too large. "
            "Maximum size is 10 MB."
        ),
        request_id=(
            current_request_id()
        ),
    ), 413


@app.errorhandler(HTTPException)
def handle_http_exception(
    error: HTTPException,
):
    """
    Handle standard HTTP exceptions.
    """
    status_code = (
        error.code
        if error.code is not None
        else 500
    )

    message = (
        error.description
        or "The request could not be completed."
    )

    if status_code >= 500:
        app.logger.error(
            "HTTP server error",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "method": (
                    request.method
                ),
                "path": (
                    request.path
                ),
                "status_code": (
                    status_code
                ),
            },
        )

    if is_api_request():
        return api_error(
            message,
            status_code,
        )

    return render_template(
        "index.html",
        error=(
            message
        ),
        request_id=(
            current_request_id()
        ),
    ), status_code


@app.errorhandler(Exception)
def handle_unexpected_exception(
    error: Exception,
):
    """
    Handle unexpected application exceptions.
    """
    app.logger.exception(
        "Unhandled application exception",
        extra={
            "request_id": (
                current_request_id()
            ),
            "method": (
                request.method
            ),
            "path": (
                request.path
            ),
            "status_code": 500,
        },
    )

    message = (
        "An unexpected application error occurred."
    )

    if is_api_request():
        return api_error(
            message,
            500,
        )

    return render_template(
        "index.html",
        error=(
            message
        ),
        request_id=(
            current_request_id()
        ),
    ), 500


@app.get("/health")
def health_check():
    """
    Return application and vector-store health.
    """
    try:
        vector_count = (
            get_vector_count()
        )

        return {
            "status": (
                "healthy"
            ),
            "application": (
                "AI Document Assistant"
            ),
            "vector_count": (
                vector_count
            ),
            "request_id": (
                current_request_id()
            ),
        }

    except VectorStoreError:
        app.logger.exception(
            "Health check failed",
            extra={
                "request_id": (
                    current_request_id()
                ),
                "path": (
                    request.path
                ),
            },
        )

        return {
            "status": (
                "unhealthy"
            ),
            "application": (
                "AI Document Assistant"
            ),
            "error": (
                "Vector database unavailable."
            ),
            "request_id": (
                current_request_id()
            ),
        }, 503


if __name__ == "__main__":
    app.run(
        debug=True,
        threaded=True,
    )