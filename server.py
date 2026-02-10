from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from asyncio.subprocess import PIPE
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

WORKSPACE_ROOT = Path(__file__).resolve().parent
MODULES_ROOT = WORKSPACE_ROOT / "modules"

PROGRESS_PREFIX = "PROGRESS:"
MAX_STDOUT_LINES = 200

app = FastAPI(title="Learn Deep Learning API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunMode(str):
    CALLABLE = "callable"
    MODULE = "module"
    SCRIPT = "script"


class CreateRunRequest(BaseModel):
    mode: str = Field(description="One of: callable, module, script")
    target: str = Field(description="Target selector based on mode")
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RunRecord:
    chapter_path: str
    mode: str
    target: str
    args: list[Any]
    kwargs: dict[str, Any]
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    stdout_tail: list[str] = field(default_factory=list)
    last_progress: dict[str, Any] | None = None
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)


CHAPTER_RUNS: dict[str, RunRecord] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_chapter(chapter_path: str) -> Path:
    chapter = (MODULES_ROOT / chapter_path).resolve()
    try:
        chapter.relative_to(MODULES_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid chapter path.") from exc
    if not chapter.exists() or not chapter.is_dir():
        raise HTTPException(status_code=404, detail="Chapter does not exist.")
    return chapter


def ensure_target_in_chapter(chapter_dir: Path, file_path: Path) -> None:
    try:
        file_path.resolve().relative_to(chapter_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Target is outside requested chapter path.",
        ) from exc


def module_to_file(module_name: str) -> Path:
    candidate = WORKSPACE_ROOT / Path(*module_name.split("."))
    return candidate.with_suffix(".py")


def file_to_module(file_path: Path) -> str:
    relative = file_path.resolve().relative_to(WORKSPACE_ROOT.resolve())
    return ".".join(relative.with_suffix("").parts)


def discover_runnables(chapter_dir: Path) -> list[dict[str, Any]]:
    runnables: list[dict[str, Any]] = []
    for py_file in sorted(chapter_dir.rglob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = file_to_module(py_file)
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # Keep discovery resilient even if one module fails import.
            continue
        exported_functions = []
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__ != module.__name__:
                continue
            if name.startswith("_"):
                continue
            exported_functions.append(name)
        if exported_functions:
            runnables.append(
                {
                    "kind": "callable",
                    "module": module_name,
                    "functions": exported_functions,
                }
            )
        runnables.append({"kind": "module", "module": module_name})
        runnables.append(
            {
                "kind": "script",
                "path": str(py_file.resolve().relative_to(MODULES_ROOT.resolve())),
            }
        )
    return runnables


def serialize_run(record: RunRecord) -> dict[str, Any]:
    return {
        "chapter_path": record.chapter_path,
        "mode": record.mode,
        "target": record.target,
        "args": record.args,
        "kwargs": record.kwargs,
        "status": record.status,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "result": record.result,
        "error": record.error,
        "stdout_tail": record.stdout_tail,
        "last_progress": record.last_progress,
    }


def append_stdout(record: RunRecord, line: str) -> None:
    stripped = line.strip()
    if not stripped:
        return
    record.stdout_tail.append(stripped)
    if len(record.stdout_tail) > MAX_STDOUT_LINES:
        record.stdout_tail = record.stdout_tail[-MAX_STDOUT_LINES:]


async def broadcast(record: RunRecord, event: dict[str, Any]) -> None:
    dead: list[asyncio.Queue[dict[str, Any]]] = []
    for queue in record.subscribers:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(queue)
    for queue in dead:
        record.subscribers.discard(queue)


async def emit_progress(record: RunRecord, event: dict[str, Any]) -> None:
    payload = {"type": "progress", **event}
    record.last_progress = payload
    await broadcast(record, payload)


async def set_run_status(record: RunRecord, status: str) -> None:
    record.status = status
    await broadcast(record, {"type": "status", "status": status})


async def run_callable(record: RunRecord, chapter_dir: Path) -> None:
    if ":" not in record.target:
        raise HTTPException(
            status_code=400,
            detail="Callable target must be '<module_path>:<function_name>'.",
        )
    module_name, function_name = record.target.split(":", 1)
    module_file = module_to_file(module_name)
    ensure_target_in_chapter(chapter_dir, module_file)
    if not module_file.exists():
        raise HTTPException(status_code=404, detail="Callable module does not exist.")

    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise HTTPException(status_code=404, detail="Callable target was not found.")

    loop = asyncio.get_running_loop()

    def progress_callback(event: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(emit_progress(record, event), loop)

    kwargs = dict(record.kwargs)
    kwargs.setdefault("progress_callback", progress_callback)
    result = await asyncio.to_thread(fn, *record.args, **kwargs)
    record.result = result if isinstance(result, dict) else {"result": result}


async def run_subprocess(record: RunRecord, command: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE)
    if process.stdout is not None:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            append_stdout(record, decoded)
            if decoded.startswith(PROGRESS_PREFIX):
                raw_event = decoded[len(PROGRESS_PREFIX) :].strip()
                try:
                    event = json.loads(raw_event)
                    await emit_progress(record, event)
                except json.JSONDecodeError:
                    append_stdout(record, f"Invalid progress JSON: {raw_event}")
            else:
                await broadcast(record, {"type": "stdout", "line": decoded})

    stderr_data = b""
    if process.stderr is not None:
        stderr_data = await process.stderr.read()
    returncode = await process.wait()

    if returncode != 0:
        message = stderr_data.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"Subprocess failed with exit code {returncode}")


async def execute_run(record: RunRecord) -> None:
    chapter_dir = resolve_chapter(record.chapter_path)
    record.started_at = utc_now()
    await set_run_status(record, "running")
    try:
        if record.mode == RunMode.CALLABLE:
            await run_callable(record, chapter_dir)
        elif record.mode == RunMode.MODULE:
            module_file = module_to_file(record.target)
            ensure_target_in_chapter(chapter_dir, module_file)
            if not module_file.exists():
                raise HTTPException(status_code=404, detail="Module target does not exist.")
            command = ["python", "-m", record.target, *[str(a) for a in record.args]]
            await run_subprocess(record, command)
            if record.result is None:
                record.result = {"ok": True}
        elif record.mode == RunMode.SCRIPT:
            script_file = (MODULES_ROOT / record.target).resolve()
            ensure_target_in_chapter(chapter_dir, script_file)
            if not script_file.exists():
                raise HTTPException(status_code=404, detail="Script target does not exist.")
            command = ["python", str(script_file), *[str(a) for a in record.args]]
            await run_subprocess(record, command)
            if record.result is None:
                record.result = {"ok": True}
        else:
            raise HTTPException(status_code=400, detail="Unsupported run mode.")
        await set_run_status(record, "completed")
    except Exception as exc:
        record.error = str(exc)
        await set_run_status(record, "failed")
    finally:
        record.finished_at = utc_now()
        await broadcast(record, {"type": "terminal", "status": record.status})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/chapters")
async def list_chapters() -> dict[str, list[str]]:
    chapters = sorted(
        [
            str(path.relative_to(MODULES_ROOT.resolve()))
            for path in MODULES_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        ]
    )
    return {"chapters": chapters}


@app.get("/chapters/{chapter_path:path}/runnables")
async def list_runnables(chapter_path: str) -> dict[str, Any]:
    chapter_dir = resolve_chapter(chapter_path)
    return {
        "chapter_path": chapter_path,
        "runnables": discover_runnables(chapter_dir),
    }


@app.get("/chapters/{chapter_path:path}/samples")
async def list_chapter_samples(chapter_path: str) -> dict[str, Any]:
    resolve_chapter(chapter_path)
    if chapter_path != "torch":
        raise HTTPException(status_code=404, detail="Sample images not configured.")
    try:
        from modules.torch.train import get_balanced_test_samples
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "chapter_path": chapter_path,
        "samples": get_balanced_test_samples(data_dir="data"),
    }


@app.get("/chapters/{chapter_path:path}/samples/{sample_id}.png")
async def get_chapter_sample_image(chapter_path: str, sample_id: int) -> Response:
    resolve_chapter(chapter_path)
    if chapter_path != "torch":
        raise HTTPException(status_code=404, detail="Sample images not configured.")
    try:
        from modules.torch.train import get_fmnist_sample_png
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        png_bytes = get_fmnist_sample_png(sample_id=sample_id, data_dir="data")
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png_bytes, media_type="image/png")


@app.post("/chapters/{chapter_path:path}/runs")
async def create_run(chapter_path: str, request: CreateRunRequest) -> dict[str, Any]:
    resolve_chapter(chapter_path)
    existing = CHAPTER_RUNS.get(chapter_path)
    if existing is not None and existing.status in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="A run is already in progress.")
    record = RunRecord(
        chapter_path=chapter_path,
        mode=request.mode,
        target=request.target,
        args=request.args,
        kwargs=request.kwargs,
    )
    CHAPTER_RUNS[chapter_path] = record
    asyncio.create_task(execute_run(record))
    return {"run": serialize_run(record)}


@app.get("/chapters/{chapter_path:path}/run")
async def get_run(chapter_path: str) -> dict[str, Any]:
    record = CHAPTER_RUNS.get(chapter_path)
    if record is None:
        raise HTTPException(status_code=404, detail="No run found for chapter.")
    return {"run": serialize_run(record)}


@app.websocket("/chapters/{chapter_path:path}/ws")
async def run_events(websocket: WebSocket, chapter_path: str) -> None:
    record = CHAPTER_RUNS.get(chapter_path)
    if record is None:
        await websocket.close(code=4404, reason="Run not found")
        return
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    record.subscribers.add(queue)
    await websocket.send_json({"type": "snapshot", "run": serialize_run(record)})
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        record.subscribers.discard(queue)
