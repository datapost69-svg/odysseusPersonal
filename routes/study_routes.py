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
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.auth_helpers import require_user
from src.constants import DATA_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/study", tags=["study"])
MAX_AUDIO_BYTES = 512 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".webm", ".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".ogg", ".flac", ".aac"}


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
        "duration": transcript["duration"],
        "segments": transcript["segments"],
    }
    _write_json(transcript_dir / f"{item_id}.json", record)
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



def _extract_source(path: Path, suffix: str) -> list[dict]:
    """Extract text locally while preserving page/slide/paragraph locations."""
    pages: list[dict] = []
    if suffix == ".pdf":
        from pypdf import PdfReader
        for number, page in enumerate(PdfReader(str(path)).pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"location": f"p. {number}", "text": text})
    elif suffix == ".pptx":
        from pptx import Presentation
        for number, slide in enumerate(Presentation(str(path)).slides, 1):
            bits = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    bits.append(shape.text)
                if getattr(shape, "has_table", False):
                    bits.extend(cell.text for row in shape.table.rows for cell in row.cells)
            text = "\n".join(x.strip() for x in bits if x and x.strip())
            if text:
                pages.append({"location": f"slide {number}", "text": text})
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


@router.post("/sources")
async def import_source(
    module_id: str = Form(...),
    file: UploadFile = File(...),
    owner: str = Depends(require_user),
):
    root = _owner_root(owner)
    _module(root, module_id)
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
        raise HTTPException(422, "No extractable text was found. Scanned PDFs need OCR, which is not included yet.")
    chunks = _source_chunks(pages)
    record = {
        "id": source_id,
        "module_id": module_id,
        "title": Path(safe_name).stem.replace("-", " "),
        "filename": safe_name,
        "format": suffix.lstrip("."),
        "file": original.relative_to(root).as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": len(pages),
        "chunks": chunks,
    }
    _write_json(folder / f"{source_id}.json", record)
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
