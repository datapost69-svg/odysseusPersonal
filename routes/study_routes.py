"""Local study-library API.

All study material is stored under ODYSSEUS_DATA_DIR. This router deliberately
has no web-search, cloud transcription, or remote model integration.
"""
import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.auth_helpers import require_user
from src.constants import DATA_DIR

logger = logging.getLogger(__name__)
_study_embedder = None
router = APIRouter(prefix="/api/study", tags=["study"])
MAX_AUDIO_BYTES = 512 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".flac", ".aac"}
JURISDICTIONS = {"england_wales", "uk_wide", "scotland", "northern_ireland", "eu", "other", "unknown"}
JURISDICTION_LABELS = {"england_wales": "England and Wales", "uk_wide": "UK-wide", "scotland": "Scotland", "northern_ireland": "Northern Ireland", "eu": "European Union", "other": "Other jurisdiction", "unknown": "Unconfirmed jurisdiction"}


def _owner_root(owner: str) -> Path:
    # Hash the account name so it is never used as a filesystem path.
    key = hashlib.sha256((owner or "local").encode("utf-8")).hexdigest()[:24]
    root = (Path(DATA_DIR) / "study_library" / key).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_modules(root: Path) -> list[dict]:
    path = root / "modules.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read study module index")
        raise HTTPException(500, "Study library index could not be read")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".study-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _module(root: Path, module_id: str) -> dict:
    found = next((m for m in _read_modules(root) if m.get("id") == module_id), None)
    if not found:
        raise HTTPException(404, "Module not found")
    return found


@router.get("/status")
async def study_status(owner: str = Depends(require_user)):
    return {"available": True, "storage": "local", "transcription": "faster-whisper (local only)"}


@router.get("/modules")
async def list_modules(owner: str = Depends(require_user)):
    root = _owner_root(owner)
    return {"modules": _read_modules(root)}


@router.post("/modules")
async def create_module(payload: dict, owner: str = Depends(require_user)):
    name = str(payload.get("name") or "").strip()
    if not name or len(name) > 120:
        raise HTTPException(400, "Enter a module name (up to 120 characters)")
    root = _owner_root(owner)
    modules = _read_modules(root)
    if any(m.get("name", "").casefold() == name.casefold() for m in modules):
        raise HTTPException(409, "A module with that name already exists")
    item = {
        "id": uuid.uuid4().hex,
        "name": name,
        "code": str(payload.get("code") or "").strip()[:40],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    modules.append(item)
    _write_json(root / "modules.json", modules)
    (root / "modules" / item["id"] / "audio").mkdir(parents=True, exist_ok=True)
    (root / "modules" / item["id"] / "transcripts").mkdir(parents=True, exist_ok=True)
    return {"module": item}


@router.post("/audio")
async def import_audio(
    module_id: str = Form(...),
    file: UploadFile = File(...),
    owner: str = Depends(require_user),
):
    root = _owner_root(owner)
    _module(root, module_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(400, "Unsupported audio format. Use WAV, MP3, M4A, MP4, WebM, OGG, FLAC or AAC.")
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if not audio:
        raise HTTPException(400, "The selected audio file is empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file exceeds the 512 MB limit")
    item_id = uuid.uuid4().hex
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(file.filename or "lecture").stem).strip(".-_")[:80] or "lecture"
    module_dir = root / "modules" / module_id
    audio_dir = module_dir / "audio"
    transcript_dir = module_dir / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{item_id}-{safe_name}{suffix}"
    audio_path.write_bytes(audio)
    try:
        transcript = await run_in_threadpool(_transcribe_local, str(audio_path))
    except Exception as exc:
        logger.exception("Local study transcription failed")
        raise HTTPException(500, f"Local transcription failed: {exc}")
    record = {
        "id": item_id,
        "module_id": module_id,
        "title": safe_name.replace("-", " "),
        "audio_file": audio_path.relative_to(root).as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": transcript["language"],
        "jurisdiction": "england_wales",
        "duration": transcript["duration"],
        "segments": transcript["segments"],
    }
    _write_json(transcript_dir / f"{item_id}.json", record)
    transcript_chunks = []
    for offset in range(0, len(record["segments"]), 6):
        group = record["segments"][offset:offset + 6]
        if not group:
            continue
        start = max(0, int(float(group[0]["start"])))
        end = max(start, int(float(group[-1]["end"])))
        transcript_chunks.append({
            "source_id": item_id, "source_type": "lecture", "jurisdiction": "england_wales",
            "title": record["title"],
            "location": f"lecture {start // 60:02d}:{start % 60:02d}–{end // 60:02d}:{end % 60:02d}",
            "text": " ".join(segment["text"] for segment in group),
        })
    try:
        await run_in_threadpool(_index_records, root, module_id, transcript_chunks)
    except Exception as exc:
        logger.exception("Local semantic indexing failed for lecture transcript")
        raise HTTPException(503, f"Local semantic indexing is unavailable: {exc}")
    return {"lecture": {k: v for k, v in record.items() if k != "segments"}, "segments": record["segments"]}


def _transcribe_local(audio_path: str) -> dict:
    # This is intentionally independent of the configurable general STT
    # provider: study material must never be sent to a remote endpoint.
    from faster_whisper import WhisperModel
    model = getattr(_transcribe_local, "_model", None)
    if model is None:
        model = WhisperModel(
            os.getenv("ODYSSEUS_STUDY_WHISPER_MODEL", "small"),
            device="cpu",
            compute_type="int8",
        )
        _transcribe_local._model = model
    segments, info = model.transcribe(audio_path, vad_filter=True, beam_size=5)
    rows = [
        {"start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": s.text.strip()}
        for s in segments
        if s.text and s.text.strip()
    ]
    return {
        "language": getattr(info, "language", None),
        "duration": round(float(getattr(info, "duration", 0) or 0), 2),
        "segments": rows,
    }



def _ocr_image(image) -> str:
    """Run optional local Tesseract OCR on a PIL image; never use a remote service."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR support requires the local pytesseract package. Install the Study requirements."
        ) from exc
    command = os.getenv("ODYSSEUS_TESSERACT_CMD")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    language = os.getenv("ODYSSEUS_STUDY_OCR_LANG", "eng")
    try:
        return (pytesseract.image_to_string(image, lang=language) or "").strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on PATH. Install the local Tesseract executable "
            "and set ODYSSEUS_TESSERACT_CMD if it is outside PATH."
        ) from exc


def _extract_source(path: Path, suffix: str) -> list[dict]:
    """Extract text locally while preserving page/slide/paragraph locations.

    Image OCR is performed only with the local Tesseract executable. No file or
    extracted content is sent to a cloud OCR service.
    """
    from io import BytesIO
    pages: list[dict] = []
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pdfium_doc = None
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            warnings = []
            # OCR pages that have no meaningful text layer (typical scanned PDFs).
            if len(text) < 80:
                try:
                    if pdfium_doc is None:
                        import pypdfium2 as pdfium
                        pdfium_doc = pdfium.PdfDocument(str(path))
                    ocr_dpi = max(150, min(float(os.getenv("ODYSSEUS_STUDY_OCR_DPI", "220")), 400))
                    bitmap = pdfium_doc[number - 1].render(scale=ocr_dpi / 72.0)
                    image = bitmap.to_pil()
                    ocr_text = _ocr_image(image)
                    if ocr_text:
                        text = (text + "\n" + ocr_text).strip()
                except RuntimeError as exc:
                    if not text:
                        raise
                    warnings.append(str(exc))
            if text:
                row = {"location": f"p. {number}", "text": text}
                if warnings:
                    row["warnings"] = warnings
                pages.append(row)
        if pdfium_doc is not None:
            pdfium_doc.close()
    elif suffix == ".pptx":
        from pptx import Presentation
        for number, slide in enumerate(Presentation(str(path)).slides, 1):
            bits = []
            warnings = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    bits.append(shape.text)
                if getattr(shape, "has_table", False):
                    bits.extend(cell.text for row in shape.table.rows for cell in row.cells)
                if getattr(shape, "has_chart", False):
                    try:
                        chart = shape.chart
                        if chart.has_title:
                            bits.append("Chart title: " + chart.chart_title.text_frame.text)
                        categories = []
                        try:
                            categories = [str(category.label) for category in chart.plots[0].categories]
                        except (AttributeError, IndexError, TypeError):
                            pass
                        for series in chart.series:
                            values = list(series.values)
                            if categories and len(categories) == len(values):
                                pairs = [f"{label}: {value}" for label, value in zip(categories, values)]
                                bits.append(f"Chart series {series.name}: " + "; ".join(pairs))
                            else:
                                bits.append(f"Chart series {series.name}: " + ", ".join(str(value) for value in values))
                    except (AttributeError, IndexError, TypeError, ValueError):
                        logger.warning("Could not extract a PowerPoint chart on slide %s", number)
                # OCR text embedded as screenshots, scanned diagrams, or images.
                if getattr(shape, "shape_type", None) == 13:  # MSO_SHAPE_TYPE.PICTURE
                    try:
                        from PIL import Image
                        image = Image.open(BytesIO(shape.image.blob)).convert("RGB")
                        image_text = _ocr_image(image)
                        if image_text:
                            bits.append(image_text)
                    except Exception as exc:
                        # Keep native slide text, but expose any image-OCR failure
                        # (including unsupported SVG/EMF image formats) so the
                        # user knows extraction may be partial.
                        logger.warning("Slide-image OCR failed on slide %s: %s", number, exc)
                        warnings.append(f"Slide-image OCR failed: {type(exc).__name__}: {exc}")
            # Include speaker notes where PowerPoint provides them.
            try:
                notes = slide.notes_slide.notes_text_frame.text
                if notes and notes.strip():
                    bits.append("Speaker notes: " + notes.strip())
            except (AttributeError, KeyError, ValueError):
                pass
            text = "\n".join(x.strip() for x in bits if x and x.strip())
            if text:
                row = {"location": f"slide {number}", "text": text}
                if warnings:
                    row["warnings"] = warnings
                pages.append(row)
    elif suffix == ".docx":
        from docx import Document as WordDocument
        doc = WordDocument(str(path))
        for number, paragraph in enumerate(doc.paragraphs, 1):
            text = (paragraph.text or "").strip()
            if text:
                pages.append({"location": f"paragraph {number}", "text": text})
        for table_no, table in enumerate(doc.tables, 1):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            text = "\n".join(row for row in rows if row.strip())
            if text:
                pages.append({"location": f"table {table_no}", "text": text})
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if text.strip():
            pages.append({"location": "text", "text": text.strip()})
    return pages

def _source_chunks(pages: list[dict], size: int = 1400, overlap: int = 180) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"location": page["location"], "text": chunk})
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def _get_study_embedder():
    """Return a strictly local FastEmbed instance; never use configured HTTP embeddings."""
    global _study_embedder
    if _study_embedder is None:
        from src.embeddings import FastEmbedClient
        model = os.getenv("ODYSSEUS_STUDY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _study_embedder = FastEmbedClient(model=model)
    return _study_embedder


def _semantic_index_path(root: Path, module_id: str) -> Path:
    return root / "modules" / module_id / "semantic_index.json"


def _index_records(root: Path, module_id: str, records: list[dict]) -> int:
    """Embed and persist source passages locally, retaining their citation metadata."""
    if not records:
        return 0
    embedder = _get_study_embedder()
    texts = [str(row.get("text", ""))[:1800] for row in records]
    vectors = embedder.encode(texts, normalize_embeddings=True)
    path = _semantic_index_path(root, module_id)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (OSError, json.JSONDecodeError):
            logger.warning("Rebuilding unreadable study semantic index for module %s", module_id)
    model_name = getattr(embedder, "model", "local-fastembed")
    by_id = {
        row.get("id"): row for row in existing
        if isinstance(row, dict) and row.get("id") and row.get("embedding_model") == model_name
    }
    for record, vector in zip(records, vectors):
        identity = hashlib.sha256(
            (str(record.get("source_id")) + "|" + str(record.get("location")) + "|" + str(record.get("text"))).encode("utf-8")
        ).hexdigest()
        by_id[identity] = {
            "id": identity,
            "source_id": record.get("source_id"),
            "source_type": record.get("source_type", "document"),
            "material_type": record.get("material_type", "other"),
            "jurisdiction": record.get("jurisdiction", "england_wales"),
            "embedding_model": getattr(embedder, "model", "local-fastembed"),
            "title": record.get("title", "Untitled source"),
            "location": record.get("location", "unknown location"),
            "text": str(record.get("text", ""))[:1800],
            "embedding": [float(value) for value in vector],
        }
    _write_json(path, list(by_id.values()))
    return len(records)


def _ensure_module_semantic_index(root: Path, module: dict) -> None:
    """Backfill the semantic index for documents imported before semantic search existed."""
    module_id = module["id"]
    path = _semantic_index_path(root, module_id)
    indexed = set()
    current_model = os.getenv("ODYSSEUS_STUDY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if path.exists():
        try:
            indexed = {
                row.get("source_id") for row in json.loads(path.read_text(encoding="utf-8"))
                if isinstance(row, dict) and row.get("embedding_model") == current_model
            }
        except (OSError, json.JSONDecodeError):
            indexed = set()
    records = []
    source_dir = root / "modules" / module_id / "sources"
    if source_dir.exists():
        for source_path in source_dir.glob("*.json"):
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if source.get("id") in indexed:
                continue
            records.extend({
                "source_id": source.get("id"), "source_type": "document",
                "material_type": source.get("material_type", "other"),
                "jurisdiction": source.get("jurisdiction", "england_wales"),
                "title": source.get("title"), "location": chunk.get("location"),
                "text": chunk.get("text", ""),
            } for chunk in source.get("chunks", []))
    transcript_dir = root / "modules" / module_id / "transcripts"
    if transcript_dir.exists():
        for transcript_path in transcript_dir.glob("*.json"):
            try:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if transcript.get("id") in indexed:
                continue
            segments = transcript.get("segments", [])
            for offset in range(0, len(segments), 6):
                group = segments[offset:offset + 6]
                if not group:
                    continue
                start = max(0, int(float(group[0].get("start", 0))))
                end = max(start, int(float(group[-1].get("end", start))))
                records.append({
                    "source_id": transcript.get("id"), "source_type": "lecture", "jurisdiction": transcript.get("jurisdiction", "england_wales"),
                    "title": transcript.get("title"),
                    "location": f"lecture {start // 60:02d}:{start % 60:02d}–{end // 60:02d}:{end % 60:02d}",
                    "text": " ".join(segment.get("text", "") for segment in group),
                })
    if records:
        _index_records(root, module_id, records)


def _semantic_search(root: Path, modules: list[dict], query: str, module_id: str | None = None, limit: int = 8) -> list[dict]:
    eligible_modules = [module for module in modules if not module_id or module.get("id") == module_id]
    if not eligible_modules:
        return []
    embedder = _get_study_embedder()
    query_vector = embedder.encode([query], normalize_embeddings=True)[0]
    hits = []
    for module in eligible_modules:
        _ensure_module_semantic_index(root, module)
        path = _semantic_index_path(root, module["id"])
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable semantic index for module %s", module.get("id"))
            continue
        for row in rows if isinstance(rows, list) else []:
            vector = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(vector, list) or len(vector) != len(query_vector):
                continue
            score = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
            if score <= 0:
                continue
            hits.append({
                "module": module.get("name"),
                "module_id": module.get("id"),
                "source": row.get("title"),
                "source_id": row.get("source_id"),
                "source_type": row.get("source_type"),
                "material_type": row.get("material_type", "other"),
                "jurisdiction": row.get("jurisdiction", "england_wales"),
                "location": row.get("location"),
                "snippet": row.get("text", ""),
                "score": round(score, 4),
            })
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:max(1, min(limit, 20))]




@router.post("/sources")
async def import_source(
    module_id: str = Form(...),
    file: UploadFile = File(...),
    material_type: str = Form("other"),
    jurisdiction: str = Form("england_wales"),
    owner: str = Depends(require_user),
):
    root = _owner_root(owner)
    _module(root, module_id)
    material_type = material_type if material_type in {"reading", "slides", "seminar_sheet", "other"} else "other"
    jurisdiction = jurisdiction if jurisdiction in JURISDICTIONS else "unknown"
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {".pdf", ".pptx", ".docx", ".txt", ".md"}
    if suffix not in allowed:
        raise HTTPException(400, "Supported files: PDF, PowerPoint, Word, TXT and Markdown.")
    raw = await file.read(100 * 1024 * 1024 + 1)
    if not raw:
        raise HTTPException(400, "The selected file is empty")
    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(413, "Document exceeds the 100 MB limit")
    source_id = uuid.uuid4().hex
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(file.filename or "source").name).strip(".-_")[:100] or "source"
    folder = root / "modules" / module_id / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / f"{source_id}-{safe_name}"
    original.write_bytes(raw)
    try:
        pages = await run_in_threadpool(_extract_source, original, suffix)
    except ImportError as exc:
        raise HTTPException(503, f"Local parser dependency is missing: {exc.name}")
    except Exception as exc:
        logger.exception("Study document extraction failed")
        raise HTTPException(422, f"Could not extract text from this document: {exc}")
    if not pages:
        raise HTTPException(
            422,
            "No extractable text was found. If this is a scanned PDF or image-only PowerPoint, install local Tesseract OCR and try again.",
        )
    chunks = _source_chunks(pages)
    record = {
        "id": source_id,
        "module_id": module_id,
        "title": Path(safe_name).stem.replace("-", " "),
        "filename": safe_name,
        "format": suffix.lstrip("."),
        "material_type": material_type,
        "jurisdiction": jurisdiction,
        "file": original.relative_to(root).as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": len(pages),
        "extraction_warnings": sorted({warning for page in pages for warning in page.get("warnings", [])}),
        "chunks": chunks,
    }
    _write_json(folder / f"{source_id}.json", record)
    try:
        await run_in_threadpool(
            _index_records, root, module_id, [
                {"source_id": source_id, "source_type": "document", "material_type": material_type, "jurisdiction": jurisdiction,
                 "title": record["title"], "location": chunk["location"], "text": chunk["text"]}
                for chunk in chunks
            ]
        )
    except Exception as exc:
        logger.exception("Local semantic indexing failed for imported study source")
        raise HTTPException(503, f"Local semantic indexing is unavailable: {exc}")
    return {"source": {k: v for k, v in record.items() if k != "chunks"}}


@router.get("/modules/{module_id}/sources")
async def list_sources(module_id: str, owner: str = Depends(require_user)):
    root = _owner_root(owner)
    _module(root, module_id)
    folder = root / "modules" / module_id / "sources"
    rows = []
    if folder.exists():
        for path in folder.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                rows.append({k: v for k, v in item.items() if k != "chunks"})
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable study source %s", path.name)
    return {"sources": sorted(rows, key=lambda x: x.get("created_at", ""), reverse=True)}


@router.get("/search")
async def search_study_library(q: str, module_id: str | None = None, owner: str = Depends(require_user)):
    query_terms = set(re.findall(r"[\w'-]{2,}", q.casefold()))
    if not query_terms:
        raise HTTPException(400, "Enter a search phrase")
    root = _owner_root(owner)
    modules = _read_modules(root)
    selected = [m for m in modules if not module_id or m.get("id") == module_id]
    results = []
    for module in selected:
        folder = root / "modules" / module["id"] / "sources"
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.glob("*.json"):
            try:
                source = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for chunk in source.get("chunks", []):
                words = re.findall(r"[\w'-]{2,}", chunk.get("text", "").casefold())
                counts = {term: words.count(term) for term in query_terms}
                score = sum(min(count, 4) for count in counts.values()) / max(1, len(query_terms))
                if score:
                    results.append({
                        "module": module.get("name"),
                        "module_id": module.get("id"),
                        "source": source.get("title"),
                        "source_id": source.get("id"),
                        "location": chunk.get("location"),
                        "snippet": chunk.get("text", "")[:700],
                        "score": round(score, 3),
                    })
        # Lecture transcripts participate in the same cross-module search.
        transcript_folder = root / "modules" / module["id"] / "transcripts"
        if transcript_folder.exists():
            for transcript_path in transcript_folder.glob("*.json"):
                try:
                    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for segment in transcript.get("segments", []):
                    text = segment.get("text", "")
                    words = re.findall(r"[\w'-]{2,}", text.casefold())
                    counts = {term: words.count(term) for term in query_terms}
                    score = sum(min(count, 4) for count in counts.values()) / max(1, len(query_terms))
                    if score:
                        start = max(0, int(float(segment.get("start", 0))))
                        timestamp = f"{start // 60:02d}:{start % 60:02d}"
                        results.append({
                            "module": module.get("name"),
                            "module_id": module.get("id"),
                            "source": transcript.get("title"),
                            "source_id": transcript.get("id"),
                            "location": f"lecture {timestamp}",
                            "snippet": text[:700],
                            "score": round(score, 3),
                        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:20], "count": len(results), "scope": "local"}



@router.get("/semantic-search")
async def semantic_search_study_library(
    q: str,
    module_id: str | None = None,
    limit: int = 8,
    owner: str = Depends(require_user),
):
    query = q.strip()
    if not query or len(query) > 1000:
        raise HTTPException(400, "Enter a search query of up to 1,000 characters.")
    root = _owner_root(owner)
    try:
        hits = await run_in_threadpool(
            _semantic_search, root, _read_modules(root), query, module_id, limit
        )
    except Exception as exc:
        logger.exception("Local study semantic search failed")
        raise HTTPException(503, f"Local semantic search is unavailable: {exc}")
    return {"results": hits, "count": len(hits), "scope": "local-semantic"}


@router.get("/modules/{module_id}/lectures")
async def list_lectures(module_id: str, owner: str = Depends(require_user)):
    root = _owner_root(owner)
    _module(root, module_id)
    folder = root / "modules" / module_id / "transcripts"
    lectures = []
    if folder.exists():
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                lectures.append({k: v for k, v in row.items() if k != "segments"})
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable study transcript %s", path.name)
    return {"lectures": lectures}


@router.get("/modules/{module_id}/lectures/{lecture_id}")
async def get_lecture(module_id: str, lecture_id: str, owner: str = Depends(require_user)):
    root = _owner_root(owner)
    _module(root, module_id)
    path = root / "modules" / module_id / "transcripts" / f"{lecture_id}.json"
    if not path.is_file():
        raise HTTPException(404, "Lecture not found")
    return json.loads(path.read_text(encoding="utf-8"))


def setup_study_routes():
    return router


def _safe_note_filename(value: str) -> str:
    stem = re.sub(r"[^\w. -]+", "", value, flags=re.UNICODE).strip(" .-_")
    return (stem or "Study notes")[:100]


def save_notes_as_pdf(payload: dict, owner: str) -> dict:
    """Render agent-authored Markdown to a local, module-organised PDF.

    Odysseus owns note composition. This function only renders and stores the
    finished notes; it never calls a model or sends course material remotely.
    """
    module_id = str(payload.get("module_id") or "").strip()
    title = str(payload.get("title") or "").strip()
    note_type = str(payload.get("note_type") or "").strip()
    markdown_text = payload.get("markdown")
    if not module_id:
        raise ValueError("module_id is required.")
    if not title or len(title) > 160:
        raise ValueError("title must contain 1–160 characters.")
    if note_type not in {"lecture", "seminar"}:
        raise ValueError("note_type must be 'lecture' or 'seminar'.")
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise ValueError("markdown must contain the completed study notes.")
    if len(markdown_text) > 200000:
        raise ValueError("markdown exceeds the 200,000-character limit.")

    # ReportLab is a local dependency; no cloud conversion service is used.
    from io import BytesIO
    import markdown as markdown_lib
    import nh3
    from bs4 import BeautifulSoup
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    import reportlab
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle,
    )

    # Prefer an installed Unicode font (Arial on Windows, DejaVu Sans on
    # Linux). ReportLab's bundled Vera fallback is deliberately last: it is
    # reliable for basic Latin but does not cover every academic character.
    windows_fonts = Path(os.getenv("WINDIR", r"C:\Windows")) / "Fonts"
    font_sets = [
        (windows_fonts / "arial.ttf", windows_fonts / "arialbd.ttf",
         windows_fonts / "ariali.ttf", windows_fonts / "arialbi.ttf"),
        (windows_fonts / "segoeui.ttf", windows_fonts / "segoeuib.ttf",
         windows_fonts / "segoeuii.ttf", windows_fonts / "segoeuiz.ttf"),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf")),
        (Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
         Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
         Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf"),
         Path("/usr/share/fonts/truetype/liberation2/LiberationSans-BoldItalic.ttf")),
    ]
    font_dir = Path(reportlab.__file__).parent / "fonts"
    font_sets.append((
        font_dir / "Vera.ttf", font_dir / "VeraBd.ttf",
        font_dir / "VeraIt.ttf", font_dir / "VeraBI.ttf",
    ))
    selected_fonts = next(
        (font_set for font_set in font_sets if all(path.is_file() for path in font_set)),
        font_sets[-1],
    )
    for name, font_path in zip(
        ("StudySans", "StudySans-Bold", "StudySans-Italic", "StudySans-BoldItalic"),
        selected_fonts,
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_path)))
    registerFontFamily(
        "StudySans", normal="StudySans", bold="StudySans-Bold",
        italic="StudySans-Italic", boldItalic="StudySans-BoldItalic",
    )

    root = _owner_root(owner)
    module = _module(root, module_id)
    module_dir = (root / "modules" / module_id).resolve()
    output_folder = module_dir / ("Lecture Notes" if note_type == "lecture" else "Seminar Notes")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_folder = output_folder.resolve()
    if not output_folder.is_relative_to(module_dir):
        raise ValueError("The notes folder resolves outside the selected module.")

    safe_title = _safe_note_filename(title)
    filename = f"{safe_title} - notes.pdf"
    destination = output_folder / filename
    if destination.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = output_folder / f"{safe_title} - notes ({stamp}-{uuid.uuid4().hex[:6]}).pdf"

    html = markdown_lib.markdown(
        markdown_text, extensions=["tables", "fenced_code", "sane_lists"]
    )
    # Only a small, safe subset of inline HTML is retained for ReportLab.
    html = nh3.clean(
        html,
        tags={"p", "strong", "b", "em", "i", "u", "a", "code", "br", "span",
              "h1", "h2", "h3", "h4", "ul", "ol", "li", "pre",
              "blockquote", "table", "thead", "tbody", "tr", "th", "td"},
        attributes={"a": {"href"}, "span": {"class"}},
        url_schemes={"http", "https", "mailto"},
    )
    soup = BeautifulSoup(html, "html.parser")
    # A semantic marker lets the agent distinguish external/model additions
    # from course-source notes. It is not arbitrary user-controlled styling.
    for span in soup.find_all("span"):
        if "supplementary" in span.get("class", []):
            span.name = "font"
            span.attrs = {"color": "#B3261E"}
        else:
            span.unwrap()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="StudyTitle", parent=styles["Title"], fontName="StudySans-Bold",
        fontSize=20, leading=25, alignment=TA_CENTER, spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyH1", parent=styles["Heading1"], fontName="StudySans-Bold", fontSize=16,
        leading=20, spaceBefore=5 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyH2", parent=styles["Heading2"], fontName="StudySans-Bold", fontSize=13,
        leading=17, spaceBefore=4 * mm, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyH3", parent=styles["Heading3"], fontName="StudySans-Bold", fontSize=11,
        leading=14, spaceBefore=3 * mm, spaceAfter=1.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyH4", parent=styles["Heading4"], fontName="StudySans-Bold", fontSize=10,
        leading=13, spaceBefore=2 * mm, spaceAfter=1 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyBody", parent=styles["BodyText"], fontName="StudySans", fontSize=9.5,
        leading=14, spaceAfter=2.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyCode", parent=styles["Code"], fontName="StudySans", fontSize=8,
        leading=10, leftIndent=5 * mm, rightIndent=5 * mm, spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudyQuote", parent=styles["BodyText"], fontName="StudySans-Italic",
        leftIndent=8 * mm, rightIndent=5 * mm, textColor=colors.HexColor("#444444"),
        spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="StudySmall", parent=styles["BodyText"], fontName="StudySans", fontSize=8,
        leading=10,
    ))
    story = [Paragraph(escape(title), styles["StudyTitle"])]
    for element in soup.contents:
        if not getattr(element, "name", None):
            continue
        tag = element.name.lower()
        if tag in {"h1", "h2", "h3"}:
            story.append(Paragraph(element.decode_contents(), styles[f"Study{tag.upper()}"]))
        elif tag in {"h4"}:
            story.append(Paragraph(element.decode_contents(), styles["StudyH4"]))
        elif tag == "p":
            story.append(Paragraph(element.decode_contents(), styles["StudyBody"]))
        elif tag in {"ul", "ol"}:
            for index, item in enumerate(element.find_all("li", recursive=False), 1):
                bullet = "•" if tag == "ul" else f"{index}."
                story.append(Paragraph(
                    item.decode_contents().replace("<p>", "").replace("</p>", ""), styles["StudyBody"], bulletText=bullet
                ))
        elif tag == "pre":
            story.append(Preformatted(element.get_text(), styles["StudyCode"]))
        elif tag == "blockquote":
            story.append(Paragraph(
                element.get_text(" ", strip=True), styles["StudyQuote"]
            ))
        elif tag == "table":
            rows = []
            for tr in element.find_all("tr"):
                rows.append([
                    Paragraph(cell.decode_contents(), styles["StudySmall"])
                    for cell in tr.find_all(["th", "td"], recursive=False)
                ])
            if rows:
                table = Table(rows, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8B8B8")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 3 * mm)])
        elif tag == "hr":
            story.append(Spacer(1, 3 * mm))

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"{title} - notes", author="Odysseus Study",
    )
    document.build(story)
    pdf_bytes = buffer.getvalue()
    if not pdf_bytes.startswith(b"%PDF-"):
        raise RuntimeError("PDF generation did not produce a valid PDF.")
    with destination.open("xb") as handle:
        handle.write(pdf_bytes)

    return {
        "saved": True,
        "path": str(destination),
        "module": module.get("name"),
        "title": title,
        "note_type": note_type,
        "format": "pdf",
        "bytes": len(pdf_bytes),
    }
