from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start_index: int
    end_index: int


def clean_text(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    cleaned_lines = []
    previous_line_empty = False

    for line in lines:
        if line:
            cleaned_lines.append(line)
            previous_line_empty = False

        elif not previous_line_empty:
            cleaned_lines.append("")
            previous_line_empty = True

    return "\n".join(cleaned_lines).strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    chunks = []
    start_index = 0
    chunk_id = 1
    text_length = len(cleaned_text)

    while start_index < text_length:
        proposed_end = min(
            start_index + chunk_size,
            text_length,
        )

        end_index = proposed_end

        if proposed_end < text_length:
            paragraph_break = cleaned_text.rfind(
                "\n\n",
                start_index,
                proposed_end,
            )

            sentence_break = cleaned_text.rfind(
                ". ",
                start_index,
                proposed_end,
            )

            minimum_break_position = (
                start_index + int(chunk_size * 0.6)
            )

            if paragraph_break >= minimum_break_position:
                end_index = paragraph_break

            elif sentence_break >= minimum_break_position:
                end_index = sentence_break + 1

        chunk_text = cleaned_text[
            start_index:end_index
        ].strip()

        if chunk_text:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                )
            )

            chunk_id += 1

        if end_index >= text_length:
            break

        next_start = end_index - chunk_overlap

        if next_start <= start_index:
            next_start = end_index

        start_index = next_start

    return chunks