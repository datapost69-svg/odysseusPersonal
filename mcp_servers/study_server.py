"""
study_server.py

Owner-scoped MCP tools for the local Study library.
All data access delegates to the same local store and retrieval functions as
routes.study_routes. The owner is injected by Odysseus at the execution
boundary; it is deliberately not an agent-supplied tool argument.
"""
import asyncio
import json
from pathlib import Path

from fastapi import HTTPException
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("study")
MAX_RESULT_CHARS = 18000


def _owner(arguments: dict) -> str:
    value = arguments.pop("_odysseus_owner", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Odysseus did not provide an authenticated owner context; refusing access.")
    return value


def _study():
    from routes import study_routes
    return study_routes


def _text(value) -> list[TextContent]:
    # Keep tool responses valid JSON. A hard string slice can corrupt the JSON
    # and silently discard the pagination metadata needed to retrieve the rest.
    if isinstance(value, dict):
        value = dict(value)
        for key in ("chunks", "segments", "results"):
            rows = value.get(key)
            if not isinstance(rows, list):
                continue
            original_count = len(rows)
            while rows and len(json.dumps(value, ensure_ascii=False, indent=2, default=str)) > MAX_RESULT_CHARS:
                rows.pop()
            if key in {"chunks", "segments"}:
                flag = "has_more_chunks" if key == "chunks" else "has_more_segments"
                offset_key = "chunk_offset" if key == "chunks" else "segment_offset"
                total_key = "chunks_total" if key == "chunks" else "segments_total"
                next_offset = int(value.get(offset_key, 0) or 0) + len(rows)
                value[flag] = bool(value.get(flag) or next_offset < value.get(total_key, next_offset))
            if len(rows) < original_count:
                value["truncated"] = True
            # Search results do not have offset pagination; callers can narrow
            # the query when a response has been shortened to fit the MCP limit.
            break
    rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(rendered) > MAX_RESULT_CHARS:
        rendered = json.dumps({
            "error": "The requested payload is too large. Retrieve fewer chunks or transcript segments per call.",
            "max_result_chars": MAX_RESULT_CHARS,
        })
    # Odysseus normalises stdio MCP errors when TextContent begins with Error:.
    # Preserve the structured payload while ensuring failures are not reported as success.
    if isinstance(value, dict) and value.get("error"):
        rendered = "Error: " + rendered
    return [TextContent(type="text", text=rendered)]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="study_list_modules",
             description="List the authenticated user's local study modules. Read-only.",
             inputSchema={"type":"object","properties":{},"additionalProperties":False}),
        Tool(name="study_search_material",
             description="Search the authenticated user's local study library semantically across all modules or one selected module. Returns excerpts with module, source, material_type (reading, slides, seminar_sheet or other), jurisdiction and page/slide/timestamp references. For targeted seminar preparation, readings are primary; seminar sheets guide topic selection; lectures and slides are supplementary. Never use this workflow to draft answers to seminar questions. Read-only.",
             inputSchema={"type":"object","properties":{
                 "query":{"type":"string","minLength":2,"maxLength":1000},
                 "module_id":{"type":"string","description":"Optional module ID; omit to search all modules."},
                 "limit":{"type":"integer","minimum":1,"maximum":20,"default":8}},
                 "required":["query"],"additionalProperties":False}),
        Tool(name="study_get_source",
             description="Retrieve an imported course document's metadata and extracted text chunks by module ID and source ID. Read-only.",
             inputSchema={"type":"object","properties":{
                 "module_id":{"type":"string"},"source_id":{"type":"string"},
                 "include_text":{"type":"boolean","default":True},
                 "chunk_offset":{"type":"integer","minimum":0,"default":0},
                 "chunk_limit":{"type":"integer","minimum":1,"maximum":50,"default":20}},
                 "required":["module_id","source_id"],"additionalProperties":False}),
        Tool(name="study_get_transcript",
             description="Retrieve a lecture transcript and timestamped segments by module ID and lecture ID. Read-only.",
             inputSchema={"type":"object","properties":{
                 "module_id":{"type":"string"},"lecture_id":{"type":"string"},
                 "start_seconds":{"type":"number","minimum":0},
                 "end_seconds":{"type":"number","minimum":0},
                 "segment_offset":{"type":"integer","minimum":0,"default":0},
                 "segment_limit":{"type":"integer","minimum":1,"maximum":500,"default":100}},
                 "required":["module_id","lecture_id"],"additionalProperties":False}),
        Tool(name="study_save_notes",
             description="Render completed Markdown notes composed by Odysseus's current agent as a PDF in the authenticated user's local Study library. Lecture notes are saved under the module's Lecture Notes folder; seminar preparation notes under Seminar Notes. The filename is '<title> - notes.pdf'. This tool only renders and saves supplied notes; it does not invoke a model or generate notes. MUST be called as the final step whenever a complete lecture or seminar note document is ready; do not leave completed notes unsaved. This is a write operation.",
             inputSchema={"type":"object","properties":{
                 "module_id":{"type":"string"},
                 "title":{"type":"string","minLength":1,"maxLength":160},
                 "note_type":{"type":"string","enum":["lecture","seminar"],"description":"lecture for general lecture notes; seminar for targeted seminar preparation notes."},
                 "markdown":{"type":"string","minLength":1,"maxLength":200000}},
                 "required":["module_id","title","note_type","markdown"],"additionalProperties":False})
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    args = dict(arguments or {})
    try:
        owner = _owner(args)
        study = _study()
        root = study._owner_root(owner)

        if name == "study_list_modules":
            return _text({"modules": study._read_modules(root), "storage":"local"})

        if name == "study_search_material":
            query = str(args.get("query") or "").strip()
            if not query or len(query) > 1000:
                raise ValueError("query must contain 2–1,000 characters.")
            module_id = str(args.get("module_id") or "").strip() or None
            limit = max(1, min(int(args.get("limit", 8)), 20))
            modules = study._read_modules(root)
            if module_id:
                study._module(root, module_id)
            hits = await asyncio.to_thread(study._semantic_search, root, modules, query, module_id, limit)
            return _text({"results":hits,"count":len(hits),"scope":"local-semantic"})

        if name == "study_get_source":
            module_id = str(args.get("module_id") or "").strip()
            source_id = str(args.get("source_id") or "").strip()
            if not module_id or not source_id or "/" in source_id or "\\" in source_id:
                raise ValueError("Valid module_id and source_id are required.")
            study._module(root, module_id)
            path = root / "modules" / module_id / "sources" / (source_id + ".json")
            if not path.is_file():
                raise ValueError("Source not found in this module.")
            row = json.loads(path.read_text(encoding="utf-8"))
            chunks = row.pop("chunks", [])
            if args.get("include_text", True):
                offset = max(0, int(args.get("chunk_offset", 0)))
                limit = max(1, min(int(args.get("chunk_limit", 20)), 50))
                row["chunks_total"] = len(chunks)
                row["chunk_offset"] = offset
                row["chunks"] = chunks[offset:offset + limit]
                row["has_more_chunks"] = offset + limit < len(chunks)
            return _text(row)

        if name == "study_get_transcript":
            module_id = str(args.get("module_id") or "").strip()
            lecture_id = str(args.get("lecture_id") or "").strip()
            if not module_id or not lecture_id or "/" in lecture_id or "\\" in lecture_id:
                raise ValueError("Valid module_id and lecture_id are required.")
            study._module(root, module_id)
            path = root / "modules" / module_id / "transcripts" / (lecture_id + ".json")
            if not path.is_file():
                raise ValueError("Transcript not found in this module.")
            row = json.loads(path.read_text(encoding="utf-8"))
            start = args.get("start_seconds")
            end = args.get("end_seconds")
            segments = row.get("segments", [])
            if start is not None or end is not None:
                lo = float(start or 0)
                hi = float(end) if end is not None else float("inf")
                if hi < lo:
                    raise ValueError("end_seconds must be greater than or equal to start_seconds.")
                segments = [s for s in segments
                            if float(s.get("end", 0)) >= lo and float(s.get("start", 0)) <= hi]
            offset = args.get("segment_offset", 0)
            limit = args.get("segment_limit", 100)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("segment_offset must be a non-negative integer.")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
                raise ValueError("segment_limit must be between 1 and 500.")
            row["segments_total"] = len(segments)
            row["segment_offset"] = offset
            row["segments"] = segments[offset:offset + limit]
            row["has_more_segments"] = offset + limit < len(segments)
            return _text(row)

        if name == "study_save_notes":
            payload = {
                "module_id": str(args.get("module_id") or "").strip(),
                "title": str(args.get("title") or "").strip(),
                "note_type": args.get("note_type"),
                "markdown": args.get("markdown"),
            }
            result = await asyncio.to_thread(study.save_notes_as_pdf, payload, owner)
            return _text(result)

        return _text({"error":f"Unknown study tool: {name}"})
    except HTTPException as exc:
        return _text({"error":exc.detail,"status":exc.status_code})
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _text({"error":str(exc)})
    except Exception as exc:
        return _text({"error":f"Study tool failed: {type(exc).__name__}: {exc}"})


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
