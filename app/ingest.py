"""Turn uploaded files (pdf/docx/txt/md/json) into text, then into overlapping chunks."""
import io
import json
from pathlib import PurePath

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".json", ".csv"}


class UnsupportedFileError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext == ".pdf":
        return _pdf_text(data)
    if ext == ".docx":
        return _docx_text(data)
    if ext == ".json":
        try:
            return json.dumps(json.loads(data.decode("utf-8")), indent=2)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UnsupportedFileError(f"invalid JSON in {filename}: {exc}") from exc
    if ext in SUPPORTED_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    raise UnsupportedFileError(f"unsupported file type '{ext}'; supported: {sorted(SUPPORTED_EXTENSIONS)}")


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _docx_text(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to chunk_size chars and carry an
    `overlap`-char tail into the next chunk. Oversized paragraphs are sliding-window split."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            step = chunk_size - overlap
            for start in range(0, len(para), step):
                chunks.append(para[start:start + chunk_size])
                if start + chunk_size >= len(para):
                    break
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{para}" if tail else para
        if len(current) > chunk_size:
            current = para
    if current:
        chunks.append(current)
    return chunks
