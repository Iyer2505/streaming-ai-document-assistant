from pathlib import Path

import pymupdf
from docx import Document


class DocumentExtractionError(Exception):
    """Raised when text cannot be extracted."""


def extract_txt(file_path: Path) -> str:
    try:
        return file_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        return file_path.read_text(
            encoding="latin-1"
        )


def extract_pdf(file_path: Path) -> str:
    pages = []

    try:
        with pymupdf.open(file_path) as document:
            for page_number, page in enumerate(
                document,
                start=1,
            ):
                page_text = page.get_text(
                    "text"
                ).strip()

                if page_text:
                    pages.append(
                        f"[Page {page_number}]\n"
                        f"{page_text}"
                    )

    except Exception as exc:
        raise DocumentExtractionError(
            "The PDF could not be read."
        ) from exc

    return "\n\n".join(pages)


def extract_docx(file_path: Path) -> str:
    try:
        document = Document(file_path)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

    except Exception as exc:
        raise DocumentExtractionError(
            "The Word document could not be read."
        ) from exc

    return "\n\n".join(paragraphs)


def extract_text(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    extractors = {
        ".txt": extract_txt,
        ".pdf": extract_pdf,
        ".docx": extract_docx,
    }

    extractor = extractors.get(extension)

    if extractor is None:
        raise DocumentExtractionError(
            f"Unsupported document type: {extension}"
        )

    extracted_text = extractor(file_path).strip()

    if not extracted_text:
        raise DocumentExtractionError(
            "No readable text was found in the document."
        )

    return extracted_text