import base64
import io
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from opt_moss import SPECTRUM_PRESETS, begin_generation, optimize_pipeline, spectrum_stats

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", str(BASE_DIR / "MOSS-Image-SFT"))
API_KEY = os.environ.get("MOSS_API_KEY", "moss-secret-change-me")
PRECISION = os.environ.get("MOSS_PRECISION", "int8")  # int8 | bf16 | fp8

app = FastAPI(title="MOSS-Image API (OpenAI Compatible)")
GEN_LOCK = threading.Lock()
neg_cache = {}
pipe = None


def _check_auth(authorization: Optional[str], x_api_key: Optional[str]):
    if not API_KEY:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _parse_size(size_str: str):
    try:
        w, h = map(int, size_str.lower().split("x"))
    except Exception:
        raise HTTPException(status_code=400, detail="size 必须为 WxH 格式，如 1024x1024")
    w = max(256, min(2048, (w // 16) * 16))
    h = max(256, min(2048, (h // 16) * 16))
    return w, h


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
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
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
    print("MOSS-Image API 已就绪！")


class ImageRequest(BaseModel):
    prompt: str
    model: str = "moss-image-sft"
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    mode: str = "spectrum"  # spectrum | turbo | exact
    steps: int = Field(default=50, ge=1, le=100)
    cfg: float = Field(default=3.0, ge=1.0, le=15.0)
    seed: Optional[int] = None
    negative_prompt: str = ""


def _run_generate(req: ImageRequest):
    if req.mode not in SPECTRUM_PRESETS:
        raise HTTPException(status_code=400, detail="mode 仅支持 spectrum / turbo / exact")
    w, h = _parse_size(req.size)
    results = []
    with GEN_LOCK:
        neg_e, neg_m = _get_neg_embeds(req.negative_prompt)
        for i in range(req.n):
            seed = (req.seed + i) if req.seed is not None else random.randint(0, 2**32 - 1)
            t0 = time.perf_counter()
            begin_generation(pipe, mode=req.mode, total_steps=req.steps)
            img = pipe(
                req.prompt,
                height=h,
                width=w,
                num_inference_steps=req.steps,
                guidance_scale=req.cfg,
                negative_prompt_embeds=neg_e,
                negative_prompt_attention_mask=neg_m,
                uncond_skip_layers=[22],
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            results.append((buf.getvalue(), seed, dt, spectrum_stats(pipe)))
    return results


@app.get("/health")
def health():
    return {"status": "ok", "precision": PRECISION, "ready": pipe is not None}


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
            }
            for png, seed, dt, stats in items
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
    png, seed, dt, _ = _run_generate(req)[0]
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Seed": str(seed), "X-Elapsed": f"{dt:.2f}"},
    )
