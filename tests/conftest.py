import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLASK_APP_PATH = PROJECT_ROOT / "app.py"


def load_flask_app_module():
    """
    Load the root app.py file without confusing it
    with the app/ Python package.
    """
    module_name = "flask_application"

    existing_module = sys.modules.get(module_name)

    if existing_module is not None:
        return existing_module

    spec = importlib.util.spec_from_file_location(
        module_name,
        FLASK_APP_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Flask application from {FLASK_APP_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="session")
def app_module():
    """
    Return the loaded root app.py module.
    """
    return load_flask_app_module()


@pytest.fixture()
def flask_app(
    monkeypatch,
    tmp_path,
    app_module,
):
    """
    Configure the Flask application for isolated tests.
    """
    test_upload_folder = tmp_path / "uploads"
    test_upload_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    app_module.app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=test_upload_folder,
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )

    monkeypatch.setattr(
        app_module,
        "generate_embeddings",
        lambda texts: [
            [0.1, 0.2, 0.3]
            for _ in texts
        ],
    )

    monkeypatch.setattr(
        app_module,
        "generate_query_embedding",
        lambda question: [0.1, 0.2, 0.3],
    )

    monkeypatch.setattr(
        app_module,
        "store_document_embeddings",
        lambda **kwargs: len(
            kwargs["chunks"]
        ),
    )

    monkeypatch.setattr(
        app_module,
        "get_vector_count",
        lambda: 1,
    )

    yield app_module.app


@pytest.fixture()
def client(flask_app):
    """
    Return Flask's test client.
    """
    return flask_app.test_client()