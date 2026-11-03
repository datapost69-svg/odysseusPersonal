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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
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
        "audio_file": str(audio_path.relative_to(root)).replace("\\", "/"),
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
