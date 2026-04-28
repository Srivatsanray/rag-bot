import os
import re
from typing import List

# remove table and figure caption and added lightweight chunking
import aiofiles
import pymupdf
import pymupdf4llm
from config.settings import TEMPFILE_UPLOAD_DIRECTORY
from fastapi import UploadFile
from utils.logger import logger

PLACEHOLDER_PREFIX = "SECTION"


CHUNK_TOKEN_SIZE = 400  # target tokens per chunk
CHUNK_OVERLAP_TOKENS = 50  # overlap between consecutive chunks


def _strip_heading_markdown(text: str) -> str:
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"_{1,3}", "", text)
    return text.strip()


def _clean_text(text: str, block_type: str) -> str:
    # Code blocks: only normalise <br> tags — never collapse spaces (preserves indentation).
    if block_type == "code":
        return re.sub(r"<br\s*/?>", "\n", text).strip()

    # Table blocks: normalise <br> tags and collapse extra spaces.
    if block_type == "table":
        text = re.sub(r"<br\s*/?>", "\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"^-\s+", "", line)
        # Drop bare page-number lines (e.g. "13", "14")
        if re.match(r"^\d+$", line.strip()):
            continue
        if re.match(r"^-{3,}$", line.strip()):
            continue
        # Drop bold-wrapped picture artifact lines that leaked through
        # e.g. "**==> picture [455 x 200] intentionally omitted <==**"
        if re.match(
            r"^\*{0,2}==>.*picture.*omitted.*<==\*{0,2}$", line.strip(), re.IGNORECASE
        ):
            continue
        # Drop bold-wrapped picture text fence lines
        # e.g. "**----- Start of picture text -----**"
        if re.match(
            r"^\*{0,2}-{3,}\s*(Start|End) of picture text", line.strip(), re.IGNORECASE
        ):
            continue
        cleaned.append(line)

    text = " ".join(cleaned)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _find_sentence_end(words: list[str], target: int, search_window: int = 40) -> int:
    """
    Scan backwards from `target` within `search_window` words to find the
    index AFTER the last word that ends a sentence (i.e. ends with . ? ! or
    their quoted/parenthesised variants).

    Returns `target` unchanged if no sentence boundary is found in the window,
    so the caller always gets a valid cut point.
    """
    sentence_end = re.compile(r'[.?!]["\')]?$')
    lo = max(0, target - search_window)
    for i in range(target - 1, lo - 1, -1):
        if sentence_end.search(words[i]):
            return i + 1  # cut AFTER this word
    return target


def _find_sentence_start(words: list[str], target: int, search_window: int = 40) -> int:
    """
    Scan forwards from `target` within `search_window` words to find the index
    of the first word that begins a new sentence (i.e. the previous word ended
    a sentence).

    Returns `target` unchanged if no sentence boundary is found in the window.
    """
    sentence_end = re.compile(r'[.?!]["\')]?$')
    hi = min(len(words) - 1, target + search_window)
    for i in range(target, hi):
        if sentence_end.search(words[i]):
            # words[i+1] is the start of the next sentence
            if i + 1 <= hi:
                return i + 1
    return target


def _split_into_chunks(text: str, page: int, topic: str, block_type: str) -> list[dict]:
    """
    Split a long paragraph block into fixed-size overlapping chunks.

    Tables and code blocks are never split — returned as a single chunk.

    Chunk boundaries are snapped to sentence edges so every chunk begins
    and ends at a complete sentence. This matters for cross-encoder scoring:
    a chunk starting mid-sentence is incoherent input that depresses the
    relevance score even when the content is correct.

    Overlap is also sentence-aligned: the next chunk starts at the sentence
    boundary nearest to (end - overlap_words), so the overlapping region is
    always a complete thought rather than a word fragment.
    """
    if block_type in ("table", "code"):
        return [{"text": text, "page": page, "topic": topic, "block_type": block_type}]

    words = text.split()
    if not words:
        return []

    # approx words per chunk: 4 chars/token * CHUNK_TOKEN_SIZE / avg word length ~5
    words_per_chunk = CHUNK_TOKEN_SIZE * 4 // 5  # ~240 words
    overlap_words = CHUNK_OVERLAP_TOKENS * 4 // 5  # ~40 words

    chunks = []
    start = 0
    while start < len(words):
        raw_end = min(start + words_per_chunk, len(words))

        # Snap end to nearest sentence boundary (scan back up to 40 words).
        # If we are already at the last word, skip snapping.
        if raw_end < len(words):
            end = _find_sentence_end(words, raw_end)
        else:
            end = raw_end

        chunk_text = " ".join(words[start:end])
        chunks.append(
            {
                "text": chunk_text,
                "page": page,
                "topic": topic,
                "block_type": block_type,
            }
        )

        if end >= len(words):
            break

        # Overlap start: back up by overlap_words from the snapped end,
        # then snap forward to the next sentence start so the chunk never
        # begins mid-sentence.
        # Safety guarantee: next start must be strictly greater than the
        # current start. If snapping does not advance us, fall back to the
        # raw word-count position which always moves forward.
        raw_next = end - overlap_words
        snapped_next = _find_sentence_start(words, max(raw_next, 0))
        start = snapped_next if snapped_next > start else max(raw_next, start + 1)

    return chunks


def validate_pdf(file: UploadFile, max_size_mb: int = 200):
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type: {file.filename}")
        raise ValueError(f"{file.filename} is not a valid PDF file.")

    file_size_mb = len(file.file.read()) / (1024 * 1024)
    file.file.seek(0)

    if file_size_mb > max_size_mb:
        logger.warning(f"File too large: {file.filename} ({file_size_mb:.2f} MB)")
        raise ValueError(
            f"PDF file size exceeds the maximum allowed size of {max_size_mb} MB."
        )

    logger.debug(f"Validated PDF: {file.filename} ({file_size_mb:.2f} MB)")


async def save_uploaded_file(files: List[UploadFile]) -> List[str]:
    os.makedirs(TEMPFILE_UPLOAD_DIRECTORY, exist_ok=True)
    file_paths = []

    for file in files:
        validate_pdf(file)
        file_path = os.path.join(TEMPFILE_UPLOAD_DIRECTORY, file.filename)
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        file_paths.append(file_path)
        logger.debug(f"Saved uploaded file: {file.filename} to {file_path}")

    return file_paths


def _parse_page_markdown(md_text: str, page_number: int) -> list[dict]:
    """
    Parse pymupdf4llm markdown for a single page into typed blocks.
    Page number is injected directly since we process one page at a time.
    """
    blocks = []
    lines = md_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Skip inline picture reference lines
        if re.match(r"^==>\s*picture", stripped, re.IGNORECASE):
            i += 1
            continue

        # Skip picture text blocks entirely
        if re.match(r"^-{3,}\s*Start of picture text", stripped, re.IGNORECASE):
            while i < len(lines):
                i += 1
                if i < len(lines) and re.match(
                    r"^-{3,}\s*End of picture text", lines[i].strip(), re.IGNORECASE
                ):
                    i += 1
                    break
            continue

        # Headings — check h3 before h2 before h1
        h3 = re.match(r"^###\s+(.*)", stripped)
        h2 = re.match(r"^##\s+(.*)", stripped)
        h1 = re.match(r"^#\s+(.*)", stripped)

        if h3:
            blocks.append(
                {
                    "type": "heading",
                    "text": _strip_heading_markdown(h3.group(1)),
                    "page": page_number,
                }
            )
            i += 1
        elif h2:
            blocks.append(
                {
                    "type": "heading",
                    "text": _strip_heading_markdown(h2.group(1)),
                    "page": page_number,
                }
            )
            i += 1
        elif h1:
            blocks.append(
                {
                    "type": "heading",
                    "text": _strip_heading_markdown(h1.group(1)),
                    "page": page_number,
                }
            )
            i += 1

        # Table — accumulate contiguous pipe lines
        elif stripped.startswith("|"):
            table_lines = []
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("|") or re.match(r"^\|[-:]+\|", s):
                    table_lines.append(s)
                    i += 1
                else:
                    break
            text = "\n".join(table_lines).strip()
            if text:
                blocks.append({"type": "table", "text": text, "page": page_number})

        # Fenced code block — accumulate until closing ```
        # pymupdf4llm emits ```[language] ... ``` fences for code listings.
        elif re.match(r"^```", stripped):
            code_lines = []
            i += 1  # skip opening fence line
            while i < len(lines):
                s = lines[i]
                if s.strip() == "```":
                    i += 1  # skip closing fence line
                    break
                code_lines.append(s)
                i += 1
            text = "\n".join(code_lines).strip()
            if text:
                blocks.append({"type": "code", "text": text, "page": page_number})

        # Paragraph — accumulate contiguous non-special lines
        else:
            para_lines = []
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    break
                if re.match(r"^#{1,3}\s+", s):
                    break
                if s.startswith("|"):
                    break
                if re.match(r"^-{3,}\s*(Start|End) of picture text", s, re.IGNORECASE):
                    break
                if re.match(r"^==>\s*picture", s, re.IGNORECASE):
                    break
                if re.match(r"^```", s):
                    break
                para_lines.append(s)
                i += 1
            text = " ".join(para_lines).strip()
            if text:
                blocks.append({"type": "paragraph", "text": text, "page": page_number})

    return blocks


def _parse_markdown_blocks(pdf_path: str) -> list[dict]:
    """Process one page at a time so page numbers are always accurate."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    blocks = []
    for page_num in range(total_pages):
        md_text = pymupdf4llm.to_markdown(
            pdf_path, pages=[page_num], show_progress=False, force_text=True
        )
        page_blocks = _parse_page_markdown(md_text, page_number=page_num + 1)
        blocks.extend(page_blocks)

    return blocks


def _get_structure(pdf_path: str) -> list[dict]:
    """
    Convert PDF to structured, chunked blocks.

    Heading blocks set the current topic label for subsequent blocks.
    Paragraph blocks are split into fixed-size overlapping chunks.
    Table and code blocks are kept whole regardless of size.
    Bare page-number paragraphs (e.g. a block whose entire text is "13") are
    dropped before chunking — they are PDF footer artefacts, not content.
    """
    raw_blocks = _parse_markdown_blocks(pdf_path)

    structured = []
    current_topic = None
    placeholder_count = 0

    for block in raw_blocks:
        if block["type"] == "heading":
            current_topic = block["text"]
            continue

        # Drop bare page-number blocks that slipped through parsing
        # (a paragraph whose entire content is a single integer)
        if block["type"] == "paragraph" and re.match(r"^\d+$", block["text"].strip()):
            continue

        if current_topic is None:
            placeholder_count += 1
            current_topic = f"{PLACEHOLDER_PREFIX}_{placeholder_count}"

        cleaned = _clean_text(block["text"], block["type"])
        if not cleaned:
            continue

        chunks = _split_into_chunks(
            text=cleaned,
            page=block["page"],
            topic=current_topic,
            block_type=block["type"],
        )
        structured.extend(chunks)

    return structured


def load_documents_from_paths(file_paths: list[str]) -> list[dict]:
    docs = []
    for file_path in file_paths:
        source = os.path.basename(file_path)
        blocks = _get_structure(file_path)

        for block in blocks:
            docs.append(
                {
                    "text": block["text"],
                    "metadata": {
                        "source": source,
                        "page": block["page"],
                        "topic": block["topic"],
                        "block_type": block["block_type"],
                    },
                }
            )

        logger.debug(f"Extracted {len(blocks)} chunks from {source}")

    return docs
