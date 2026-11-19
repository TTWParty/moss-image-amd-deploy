import base64
import io
import json
import os
import random
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from opt_moss import SPECTRUM_PRESETS, begin_generation, optimize_pipeline, spectrum_stats

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", str(BASE_DIR / "MOSS-Image-SFT"))
OUT_DIR = Path(os.environ.get("MOSS_OUT_DIR", str(BASE_DIR / "outputs")))
THUMB_DIR = OUT_DIR / ".thumbs"
WEB_INDEX = BASE_DIR / "web" / "index.html"
API_KEY = os.environ.get("MOSS_API_KEY", "moss-secret-change-me")
PRECISION = os.environ.get("MOSS_PRECISION", "int8")  # int8 | bf16 | fp8
SAVE_OUTPUTS = os.environ.get("MOSS_SAVE_OUTPUTS", "1") != "0"

OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MOSS-Image Web UI & OpenAI-Compatible API")
GEN_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
neg_cache = {}
pipe = None


class _JobCancelled(Exception):
    pass


def _check_auth(authorization: Optional[str], x_api_key: Optional[str]):
    if not API_KEY:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="API Key 错误或未填写")


def _parse_size(size_str: str):
    try:
        w, h = map(int, size_str.lower().split("x"))
    except Exception:
        raise HTTPException(status_code=400, detail="size 必须为 WxH 格式，如 1024x1024")
    w = max(256, min(2048, (w // 16) * 16))
    h = max(256, min(2048, (h // 16) * 16))
    return w, h


def _validate_skip_layers(layers: Optional[List[int]]) -> List[int]:
    if not layers:
        return []
    cleaned = []
    for idx in layers:
        i = int(idx)
        if not 0 <= i < 42:
            raise HTTPException(status_code=400, detail=f"uncond_skip_layers 必须在 0~41 之间，收到: {i}")
        if i not in cleaned:
            cleaned.append(i)
    return cleaned


def _safe_output_path(filename: str) -> Path:
    if not filename or Path(filename).name != filename or not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="非法文件名")
    p = (OUT_DIR / filename).resolve()
    if p.parent != OUT_DIR.resolve():
        raise HTTPException(status_code=400, detail="非法文件路径")
    return p


def _ensure_thumb_b64(png_path: Path) -> str:
    thumb_path = THUMB_DIR / f"{png_path.stem}.jpg"
    if not thumb_path.is_file():
        with Image.open(png_path) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320), Image.Resampling.LANCZOS)
            im.save(thumb_path, format="JPEG", quality=78)
    return base64.b64encode(thumb_path.read_bytes()).decode("ascii")


def _get_neg_embeds(neg_text: str):
    if neg_text not in neg_cache:
        neg_cache[neg_text] = pipe.encode_prompt(neg_text)
    return neg_cache[neg_text]


@app.on_event("startup")
def _startup():
    global pipe
    print(f"加载 MOSS-Image-SFT [precision={PRECISION}]...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=5.0)
    pipe.to("cuda")
    optimize_pipeline(pipe, precision=PRECISION, use_compile=True)

    neg_e, neg_m = _get_neg_embeds("")
    begin_generation(pipe, mode="exact", total_steps=2)
    _ = pipe(
        "warmup", height=1024, width=1024, num_inference_steps=2, guidance_scale=3.0,
        negative_prompt_embeds=neg_e, negative_prompt_attention_mask=neg_m,
        uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(0),
    )
    torch.cuda.synchronize()
    print("MOSS-Image Web UI & API 已就绪！")


class ImageRequest(BaseModel):
    prompt: str
    model: str = "moss-image-sft"
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    mode: str = "spectrum"  # spectrum | turbo | exact
    steps: int = Field(default=50, ge=1, le=100)
    cfg: float = Field(default=3.0, ge=1.0, le=15.0)
    shift: float = Field(default=5.0, ge=0.5, le=15.0)
    uncond_skip_layers: Optional[List[int]] = Field(default_factory=lambda: [22])
    seed: Optional[int] = None
    negative_prompt: str = ""


def _save_output(png_bytes: bytes, img: Image.Image, meta: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
    fname = f"{ts}_{PRECISION}_{meta['mode']}_{meta['size']}_s{meta['seed']}.png"
    png_path = OUT_DIR / fname
    png_path.write_bytes(png_bytes)

    # 写入元信息 JSON，供刷新页面后在图库回填参数
    meta_out = dict(meta, filename=fname, created_at=time.time())
    (OUT_DIR / f"{png_path.stem}.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 顺手生成缩略图缓存
    thumb = img.convert("RGB")
    thumb.thumbnail((320, 320), Image.Resampling.LANCZOS)
    thumb.save(THUMB_DIR / f"{png_path.stem}.jpg", format="JPEG", quality=78)
    return fname


def _run_generate(req: ImageRequest, job: Optional[dict] = None):
    if req.mode not in SPECTRUM_PRESETS:
        raise HTTPException(status_code=400, detail="mode 仅支持 spectrum / turbo / exact")
    w, h = _parse_size(req.size)
    skip_layers = _validate_skip_layers(req.uncond_skip_layers)
    cancel_evt: Optional[threading.Event] = job["cancel_event"] if job else None

    def _update_job(**kwargs):
        if job is not None:
            with JOBS_LOCK:
                job.update(kwargs)

    results = []
    with GEN_LOCK:
        if cancel_evt and cancel_evt.is_set():
            raise _JobCancelled()

        _update_job(status="running", stage="encoding", step=0, total_steps=req.steps, started_at=time.perf_counter())
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=float(req.shift))
        neg_e, neg_m = _get_neg_embeds(req.negative_prompt)
        pos_e, pos_m = pipe.encode_prompt(req.prompt)

        for i in range(req.n):
            if cancel_evt and cancel_evt.is_set():
                raise _JobCancelled()

            seed = (req.seed + i) if req.seed is not None else random.randint(0, 2**32 - 1)
            t0 = time.perf_counter()
            begin_generation(pipe, mode=req.mode, total_steps=req.steps)
            _update_job(stage="denoising", image_index=i + 1, total_images=req.n, step=0, total_steps=req.steps)

            orig_pb = pipe.progress_bar

            def _tracked_pb(items):
                total = len(items) if hasattr(items, "__len__") else req.steps
                for idx, item in enumerate(orig_pb(items)):
                    if cancel_evt and cancel_evt.is_set():
                        raise _JobCancelled()
                    _update_job(stage="denoising", step=idx + 1, total_steps=total)
                    yield item
                if cancel_evt and cancel_evt.is_set():
                    raise _JobCancelled()
                _update_job(stage="decoding", step=total, total_steps=total)

            pipe.progress_bar = _tracked_pb
            try:
                img = pipe(
                    prompt=None,
                    prompt_embeds=pos_e,
                    prompt_attention_mask=pos_m,
                    negative_prompt_embeds=neg_e,
                    negative_prompt_attention_mask=neg_m,
                    height=h,
                    width=w,
                    num_inference_steps=req.steps,
                    guidance_scale=req.cfg,
                    uncond_skip_layers=skip_layers,
                    generator=torch.Generator("cpu").manual_seed(seed),
                ).images[0]
            finally:
                pipe.progress_bar = orig_pb

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stats = spectrum_stats(pipe)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            filename = None
            if SAVE_OUTPUTS:
                filename = _save_output(
                    png_bytes,
                    img,
                    {
                        "prompt": req.prompt,
                        "negative_prompt": req.negative_prompt,
                        "mode": req.mode,
                        "size": f"{w}x{h}",
                        "steps": req.steps,
                        "cfg": req.cfg,
                        "shift": req.shift,
                        "uncond_skip_layers": skip_layers,
                        "seed": seed,
                        "elapsed_sec": round(dt, 2),
                        "stats": stats,
                        "precision": PRECISION,
                    },
                )
            results.append((png_bytes, seed, dt, stats, filename))
    return results


def _prune_old_jobs():
    with JOBS_LOCK:
        done_ids = [
            jid for jid, j in JOBS.items()
            if j["status"] in ("done", "error", "cancelled")
        ]
        if len(done_ids) > 16:
            done_ids.sort(key=lambda jid: JOBS[jid].get("created_at", 0))
            for old_id in done_ids[:-16]:
                JOBS.pop(old_id, None)


def _job_worker(job_id: str, req: ImageRequest):
    job = JOBS[job_id]
    try:
        items = _run_generate(req, job=job)
        data = [
            {
                "b64_json": base64.b64encode(png).decode("ascii"),
                "seed": seed,
                "elapsed_sec": round(dt, 2),
                "stats": stats,
                "filename": fname,
            }
            for png, seed, dt, stats, fname in items
        ]
        with JOBS_LOCK:
            job["status"] = "done"
            job["stage"] = "done"
            job["data"] = data
            job["finished_at"] = time.perf_counter()
    except _JobCancelled:
        torch.cuda.empty_cache()
        with JOBS_LOCK:
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["finished_at"] = time.perf_counter()
    except HTTPException as e:
        with JOBS_LOCK:
            job["status"] = "error"
            job["error"] = e.detail
            job["finished_at"] = time.perf_counter()
    except Exception as e:
        with JOBS_LOCK:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            job["finished_at"] = time.perf_counter()
    finally:
        _prune_old_jobs()


@app.get("/")
def web_ui():
    if not WEB_INDEX.is_file():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(WEB_INDEX)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "precision": PRECISION,
        "ready": pipe is not None,
        "auth_required": bool(API_KEY),
    }


@app.post("/api/jobs")
def create_job(
    req: ImageRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    if req.mode not in SPECTRUM_PRESETS:
        raise HTTPException(status_code=400, detail="mode 仅支持 spectrum / turbo / exact")
    _parse_size(req.size)
    _validate_skip_layers(req.uncond_skip_layers)

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "step": 0,
        "total_steps": req.steps,
        "image_index": 0,
        "total_images": req.n,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "data": None,
        "cancel_event": threading.Event(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    t = threading.Thread(target=_job_worker, args=(job_id, req), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        elapsed = 0.0
        if job["started_at"] is not None:
            end = job["finished_at"] if job["finished_at"] is not None else time.perf_counter()
            elapsed = round(end - job["started_at"], 1)
        return {
            "id": job["id"],
            "status": job["status"],
            "stage": job["stage"],
            "step": job["step"],
            "total_steps": job["total_steps"],
            "image_index": job["image_index"],
            "total_images": job["total_images"],
            "elapsed_sec": elapsed,
            "error": job["error"],
            "data": job["data"] if job["status"] == "done" else None,
        }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")
        job["cancel_event"].set()
        if job["status"] == "queued":
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
        return {"id": job_id, "cancelling": True, "status": job["status"]}


@app.get("/api/gallery")
def list_gallery(
    limit: int = 48,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    limit = max(1, min(120, limit))
    all_pngs = sorted(OUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    pngs = all_pngs[:limit]
    items = []
    for p in pngs:
        meta_file = OUT_DIR / f"{p.stem}.json"
        meta = {}
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        try:
            thumb_b64 = _ensure_thumb_b64(p)
        except Exception:
            continue
        items.append({
            "filename": p.name,
            "created_at": meta.get("created_at", p.stat().st_mtime),
            "prompt": meta.get("prompt", ""),
            "negative_prompt": meta.get("negative_prompt", ""),
            "mode": meta.get("mode", "spectrum"),
            "size": meta.get("size", ""),
            "steps": meta.get("steps", 50),
            "cfg": meta.get("cfg", 3.0),
            "shift": meta.get("shift", 5.0),
            "uncond_skip_layers": meta.get("uncond_skip_layers", [22]),
            "seed": meta.get("seed"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "stats": meta.get("stats", ""),
            "precision": meta.get("precision", ""),
            "thumb_b64": thumb_b64,
        })
    return {"items": items, "total": len(all_pngs)}


@app.get("/api/outputs/{filename}")
def get_output_image(
    filename: str,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    p = _safe_output_path(filename)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(p, media_type="image/png", filename=filename)


@app.delete("/api/outputs/{filename}")
def delete_output_image(
    filename: str,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    p = _safe_output_path(filename)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    p.unlink(missing_ok=True)
    (OUT_DIR / f"{p.stem}.json").unlink(missing_ok=True)
    (THUMB_DIR / f"{p.stem}.jpg").unlink(missing_ok=True)
    return {"deleted": filename}


@app.post("/v1/images/generations")
def openai_images(
    req: ImageRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    items = _run_generate(req)
    return {
        "created": int(time.time()),
        "data": [
            {
                "b64_json": base64.b64encode(png).decode("ascii"),
                "seed": seed,
                "elapsed_sec": round(dt, 2),
                "stats": stats,
                "filename": fname,
            }
            for png, seed, dt, stats, fname in items
        ],
    }


@app.post("/generate")
def direct_png(
    req: ImageRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    req.n = 1
    png, seed, dt, _, fname = _run_generate(req)[0]
    headers = {"X-Seed": str(seed), "X-Elapsed": f"{dt:.2f}"}
    if fname:
        headers["X-Filename"] = fname
    return Response(content=png, media_type="image/png", headers=headers)
