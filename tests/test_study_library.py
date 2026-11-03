import json
from pathlib import Path

from fastapi import HTTPException

from routes.study_routes import _owner_root, _read_modules, _write_json


def test_study_library_is_partitioned_by_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("routes.study_routes.DATA_DIR", str(tmp_path))
    alice = _owner_root("alice")
    bob = _owner_root("bob")
    assert alice != bob
    _write_json(alice / "modules.json", [{"id": "a", "name": "History"}])
    assert _read_modules(alice) == [{"id": "a", "name": "History"}]
    assert _read_modules(bob) == []


def test_study_index_write_is_valid_json(tmp_path):
    path = Path(tmp_path) / "modules.json"
    _write_json(path, [{"id": "x", "name": "Politics"}])
    assert json.loads(path.read_text(encoding="utf-8"))[0]["name"] == "Politics"

def test_local_transcriber_returns_timestamped_segments(monkeypatch, tmp_path):
    import sys
    import types
    from routes.study_routes import _transcribe_local

    class Segment:
        start = 1.25
        end = 2.75
        text = "  A lecture sentence.  "

    class Info:
        language = "en"
        duration = 12.0

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert Path(path).is_file()
            assert kwargs["vad_filter"] is True
            return iter([Segment()]), Info()

    class FakeWhisperModel:
        def __init__(self, model_name, device, compute_type):
            assert device == "cpu"
            assert compute_type == "int8"
            self.model = FakeModel()

        def transcribe(self, path, **kwargs):
            return self.model.transcribe(path, **kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.delattr(_transcribe_local, "_model", raising=False)
    audio = tmp_path / "lecture.wav"
    audio.write_bytes(b"placeholder")
    result = _transcribe_local(str(audio))
    assert result["language"] == "en"
    assert result["duration"] == 12.0
    assert result["segments"] == [{"start": 1.25, "end": 2.75, "text": "A lecture sentence."}]
    monkeypatch.delattr(_transcribe_local, "_model", raising=False)


def test_semantic_index_is_local_and_returns_source_locations(tmp_path, monkeypatch):
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            vectors = []
            for text in texts:
                lowered = text.casefold()
                vector = [1.0, 0.0] if "democracy" in lowered else [0.0, 1.0]
                vectors.append(vector)
            return np.asarray(vectors, dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    modules = [{"id": "politics", "name": "Politics"}, {"id": "history", "name": "History"}]
    study._index_records(root, "politics", [{
        "source_id": "lecture-1", "source_type": "lecture", "title": "Week 1",
        "location": "lecture 02:10–03:40", "text": "Democracy and political representation",
    }])
    study._index_records(root, "history", [{
        "source_id": "reading-1", "source_type": "document", "title": "Reading A",
        "location": "p. 12", "text": "Medieval trade routes and port cities",
    }])

    hits = study._semantic_search(root, modules, "democracy")
    assert len(hits) == 1
    assert hits[0]["module"] == "Politics"
    assert hits[0]["source"] == "Week 1"
    assert hits[0]["location"] == "lecture 02:10–03:40"
    assert hits[0]["source_type"] == "lecture"
    assert (root / "modules" / "politics" / "semantic_index.json").is_file()
    assert (root / "modules" / "history" / "semantic_index.json").is_file()


def test_semantic_search_can_be_limited_to_one_module(tmp_path, monkeypatch):
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    modules = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
    for module in modules:
        study._index_records(root, module["id"], [{
            "source_id": f"source-{module['id']}", "source_type": "document",
            "title": f"Source {module['id']}", "location": "p. 1", "text": "Shared concept",
        }])
    hits = study._semantic_search(root, modules, "concept", module_id="b")
    assert len(hits) == 1
    assert hits[0]["module_id"] == "b"


def test_semantic_search_backfills_preexisting_source_chunks(tmp_path, monkeypatch):
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    module_id = "old-module"
    source_dir = root / "modules" / module_id / "sources"
    study._write_json(source_dir / "old-source.json", {
        "id": "old-source", "title": "Previously imported reading",
        "chunks": [{"location": "p. 9", "text": "The social contract legitimises political authority."}],
    })
    hits = study._semantic_search(root, [{"id": module_id, "name": "Political Theory"}], "political authority")
    assert hits[0]["source"] == "Previously imported reading"
    assert hits[0]["location"] == "p. 9"
    assert (root / "modules" / module_id / "semantic_index.json").is_file()



def test_semantic_index_preserves_material_classification(tmp_path, monkeypatch):
    import json
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        model = "fake-local"
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    study._index_records(root, "law", [{
        "source_id": "reading-1", "source_type": "document", "material_type": "reading",
        "title": "Judicial Review", "location": "p. 4", "text": "Illegality and relevant considerations",
    }])
    rows = json.loads(study._semantic_index_path(root, "law").read_text(encoding="utf-8"))
    assert rows[0]["material_type"] == "reading"



def test_semantic_index_preserves_seminar_sheet_classification(tmp_path, monkeypatch):
    import json
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        model = "fake-local"
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    study._index_records(root, "law", [{
        "source_id": "seminar-1", "source_type": "document", "material_type": "seminar_sheet",
        "title": "Seminar 1", "location": "p. 1", "text": "Discuss judicial review",
    }])
    rows = json.loads(study._semantic_index_path(root, "law").read_text(encoding="utf-8"))
    assert rows[0]["material_type"] == "seminar_sheet"



def test_safe_note_filename_removes_path_components():
    from routes.study_routes import _safe_note_filename
    assert "/" not in _safe_note_filename("../../Criminal Law: Seminar 2")
    assert chr(92) not in _safe_note_filename("Criminal" + chr(92) + "Law")


def test_pdf_extraction_uses_local_ocr_for_scanned_pages(tmp_path, monkeypatch):
    import sys
    import types
    import routes.study_routes as study

    class FakePage:
        def extract_text(self):
            return ""

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    class FakeBitmap:
        def to_pil(self):
            return object()

    class FakePdfPage:
        def render(self, scale):
            assert scale >= 1.5
            return FakeBitmap()

    class FakePdf:
        def __getitem__(self, index):
            assert index == 0
            return FakePdfPage()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=FakeReader))
    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=lambda path: FakePdf()))
    monkeypatch.setattr(study, "_ocr_image", lambda image: "Recognised scanned lecture text")
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-placeholder")

    pages = study._extract_source(pdf_path, ".pdf")

    assert pages == [{"location": "p. 1", "text": "Recognised scanned lecture text"}]


def test_powerpoint_extraction_includes_speaker_notes_and_image_ocr(tmp_path, monkeypatch):
    from io import BytesIO
    from PIL import Image
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.util import Inches
    import routes.study_routes as study

    image_buffer = BytesIO()
    Image.new("RGB", (40, 40), "white").save(image_buffer, format="PNG")
    image_buffer.seek(0)
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.shapes.add_picture(image_buffer, Inches(1), Inches(1), width=Inches(1))
    chart_data = CategoryChartData()
    chart_data.categories = ["2024", "2025"]
    chart_data.add_series("Attendance", (120, 145))
    slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(2), Inches(1), Inches(4), Inches(3), chart_data
    )
    slide.notes_slide.notes_text_frame.text = "Remember the lecturer's example."
    path = tmp_path / "lecture.pptx"
    presentation.save(path)

    monkeypatch.setattr(study, "_ocr_image", lambda image: "Text inside slide diagram")
    pages = study._extract_source(path, ".pptx")

    assert len(pages) == 1
    assert pages[0]["location"] == "slide 1"
    assert "Text inside slide diagram" in pages[0]["text"]
    assert "Attendance" in pages[0]["text"]
    assert "120" in pages[0]["text"]
    assert "145" in pages[0]["text"]
    assert "Remember the lecturer's example." in pages[0]["text"]
