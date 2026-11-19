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

from opt_moss import (
    SAMPLERS,
    SCHEDULES,
    SPECTRUM_PRESETS,
    begin_generation,
    load_lora_weights,
    optimize_pipeline,
    optimize_transformer,
    sample_flow_matching,
    spectrum_stats,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", str(BASE_DIR / "MOSS-Image-SFT"))
MODEL_ROOT = Path(MODEL_DIR).resolve().parent
EXTRA_MODELS_DIR = Path(os.environ.get("MOSS_EXTRA_MODELS_DIR", str(MODEL_ROOT / "extra_models")))
LORA_DIR = Path(os.environ.get("MOSS_LORA_DIR", str(MODEL_ROOT / "loras")))
OUT_DIR = Path(os.environ.get("MOSS_OUT_DIR", str(BASE_DIR / "outputs")))
THUMB_DIR = OUT_DIR / ".thumbs"
WEB_INDEX = BASE_DIR / "web" / "index.html"
API_KEY = os.environ.get("MOSS_API_KEY", "moss-secret-change-me")
PRECISION = os.environ.get("MOSS_PRECISION", "int8")  # int8 | bf16 | fp8
SAVE_OUTPUTS = os.environ.get("MOSS_SAVE_OUTPUTS", "1") != "0"

OUT_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MOSS-Image Web UI & OpenAI-Compatible API")
GEN_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
neg_cache = {}
pipe = None
PRIMARY_TR = None
TRANSFORMERS: dict[str, torch.nn.Module] = {}
LORAS: dict[str, list] = {}
LORA_WARMED_UP = False

BASE_MODEL_META = {
    "moss-image-sft": {
        "label": "MOSS-Image-SFT（官方 · 中文排版最强）",
        "short": "SFT",
    },
    "kromoss": {
        "label": "Kromoss（70万图全量重训 · 画质/对比度更强）",
        "short": "Kromoss",
    },
    "mossyume": {
        "label": "MossYume（2000万图重训 · 二次元+写实）",
        "short": "MossYume",
    },
    "moss-image-heal": {
        "label": "MOSS-Image-heal（数值修复版基座）",
        "short": "Heal",
    },
}

LORA_META = {
    "kromoss-8step-dmd2": {
        "label": "Kromoss 8步 DMD2-GAN（画质最强 · 推荐）",
        "short": "8步DMD2",
        "rec_steps": 8,
        "rec_cfg": 1.0,
        "rec_mode": "exact",
        "rec_base": "kromoss",
    },
    "8-steps-dmd2-lora": {
        "label": "Kromoss 8步 DMD2-GAN（画质最强 · 推荐）",
        "short": "8步DMD2",
        "rec_steps": 8,
        "rec_cfg": 1.0,
        "rec_mode": "exact",
        "rec_base": "kromoss",
    },
    "kromoss-8step-dmd": {
        "label": "Kromoss 8步 DMD",
        "short": "8步DMD",
        "rec_steps": 8,
        "rec_cfg": 1.0,
        "rec_mode": "exact",
        "rec_base": "kromoss",
    },
    "sft-4step": {
        "label": "SFT 4步 Turbo（akhaliq · 极速+双语招牌）",
        "short": "4步Turbo",
        "rec_steps": 4,
        "rec_cfg": 1.0,
        "rec_mode": "exact",
        "rec_base": "moss-image-sft",
    },
    "sft-8step": {
        "label": "SFT 8步 DMD（lodestones · 均衡快）",
        "short": "8步DMD",
        "rec_steps": 8,
        "rec_cfg": 1.0,
        "rec_mode": "exact",
        "rec_base": "moss-image-sft",
    },
    "deband": {
        "label": "DeBand 去竖向色带 LoRA",
        "short": "DeBand",
        "rec_steps": 50,
        "rec_cfg": 3.0,
        "rec_mode": "spectrum",
        "rec_base": "moss-image-sft",
    },
}


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


def _is_complete_transformer_dir(tr_dir: Path) -> bool:
    if not (tr_dir / "config.json").is_file():
        return False
    idx_file = tr_dir / "diffusion_pytorch_model.safetensors.index.json"
    if idx_file.is_file():
        try:
            wm = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
            shards = set(wm.values())
            return bool(shards) and all((tr_dir / s).is_file() and (tr_dir / s).stat().st_size > 1024 for s in shards)
        except Exception:
            return False
    single = tr_dir / "diffusion_pytorch_model.safetensors"
    return single.is_file() and single.stat().st_size > 1024


def _warmup_lora_if_needed():
    global LORA_WARMED_UP
    if LORA_WARMED_UP or not LORAS or pipe is None:
        return
    first_lora_id = next(iter(LORAS))
    print(f"预热 FP32 Turbo LoRA 编译核 ({first_lora_id})...")
    prev_tr = pipe.transformer
    try:
        pipe.transformer = PRIMARY_TR
        begin_generation(pipe, mode="exact", total_steps=1, lora_layers=LORAS[first_lora_id], lora_strength=1.0)
        _ = pipe(
            "warmup",
            height=1024,
            width=1024,
            num_inference_steps=1,
            guidance_scale=1.0,
            uncond_skip_layers=[],
            generator=torch.Generator("cpu").manual_seed(0),
        )
        torch.cuda.synchronize()
        LORA_WARMED_UP = True
    finally:
        pipe.transformer = prev_tr


def _scan_and_load_extras() -> dict:
    """Discover and load any newly added base transformers in EXTRA_MODELS_DIR and LoRAs in LORA_DIR."""
    added_models = []
    added_loras = []
    tr_cls = type(PRIMARY_TR)
    num_layers = len(PRIMARY_TR.layers)
    dim = int(PRIMARY_TR.config.dim)
    ffn_dim = int(PRIMARY_TR.config.ffn_dim)

    if EXTRA_MODELS_DIR.is_dir():
        for entry in sorted(EXTRA_MODELS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            model_id = entry.name.lower()
            if model_id in TRANSFORMERS:
                continue
            tr_dir = entry / "transformer" if (entry / "transformer" / "config.json").is_file() else entry
            if not _is_complete_transformer_dir(tr_dir):
                print(f"[跳过] 底模目录 {entry.name} 权重不完整，等待下载完成后可点「扫描新模型」")
                continue
            try:
                t0 = time.perf_counter()
                print(f"加载扩展底模 [{model_id}] ({tr_dir})...")
                tr = tr_cls.from_pretrained(str(tr_dir), torch_dtype=torch.bfloat16, local_files_only=True)
                tr.to("cuda").eval()
                optimize_transformer(
                    tr,
                    precision=PRECISION,
                    use_compile=False,
                    shared_block_fns=PRIMARY_TR._shared_block_fns,
                    shared_lora_block_fns=PRIMARY_TR._shared_lora_block_fns,
                    h64=PRIMARY_TR._h64,
                )
                TRANSFORMERS[model_id] = tr
                added_models.append(model_id)
                print(f"扩展底模 [{model_id}] 已就绪（{time.perf_counter() - t0:.1f}s）")
            except Exception as e:
                print(f"[警告] 加载扩展底模 {model_id} 失败: {e}")

    if LORA_DIR.is_dir():
        for fp in sorted(LORA_DIR.glob("*.safetensors")):
            lora_id = fp.stem.lower()
            if lora_id in LORAS or lora_id == "none":
                continue
            if fp.stat().st_size < 1024 * 1024:
                print(f"[跳过] LoRA 文件 {fp.name} 体积过小（可能未下载完）")
                continue
            try:
                t0 = time.perf_counter()
                print(f"加载 FP32 LoRA [{lora_id}] ({fp.name})...")
                LORAS[lora_id] = load_lora_weights(
                    fp, num_layers=num_layers, dim=dim, ffn_dim=ffn_dim, device="cuda"
                )
                added_loras.append(lora_id)
                print(f"LoRA [{lora_id}] 已驻留显存（{time.perf_counter() - t0:.1f}s）")
            except Exception as e:
                print(f"[警告] 加载 LoRA {fp.name} 失败: {e}")

    if added_loras:
        _warmup_lora_if_needed()

    return {"added_models": added_models, "added_loras": added_loras}


def _build_catalog() -> dict:
    models = []
    for mid in TRANSFORMERS:
        info = BASE_MODEL_META.get(mid, {"label": mid, "short": mid})
        models.append({"id": mid, "label": info["label"], "short": info["short"], "loaded": True})

    loras = [{
        "id": "none",
        "label": "不使用 LoRA（50 步精修）",
        "short": "无LoRA",
        "loaded": True,
        "rec_steps": 50,
        "rec_cfg": 3.0,
        "rec_mode": "spectrum",
        "rec_base": None,
    }]
    for lid in LORAS:
        info = LORA_META.get(lid, {
            "label": f"{lid}（自定义 LoRA）",
            "short": lid,
            "rec_steps": 8,
            "rec_cfg": 1.0,
            "rec_mode": "exact",
            "rec_base": None,
        })
        loras.append({"id": lid, "loaded": True, **info})

    # Find matching Kromoss 8-step DMD2 LoRA id (supports both canonical and upstream filename)
    kromoss_dmd2_id = "kromoss-8step-dmd2" if "kromoss-8step-dmd2" in LORAS else (
        "8-steps-dmd2-lora" if "8-steps-dmd2-lora" in LORAS else "kromoss-8step-dmd2"
    )

    presets = [
        {
            "id": "kromoss-8step",
            "label": "⚡ Kromoss 8步（画质最强 · 推荐）",
            "model": "kromoss",
            "lora": kromoss_dmd2_id,
            "lora_strength": 1.0,
            "mode": "exact",
            "sampler": "euler",
            "schedule": "default",
            "steps": 8,
            "cfg": 1.0,
            "shift": 5.0,
            "uncond_skip_layers": [],
            "available": ("kromoss" in TRANSFORMERS) and (kromoss_dmd2_id in LORAS),
        },
        {
            "id": "sft-4step",
            "label": "⚡ SFT 4步（极速 · 双语文字强）",
            "model": "moss-image-sft",
            "lora": "sft-4step",
            "lora_strength": 1.0,
            "mode": "exact",
            "sampler": "euler",
            "schedule": "default",
            "steps": 4,
            "cfg": 1.0,
            "shift": 5.0,
            "uncond_skip_layers": [],
            "available": ("moss-image-sft" in TRANSFORMERS) and ("sft-4step" in LORAS),
        },
        {
            "id": "sft-8step",
            "label": "⚡ SFT 8步（均衡快 · 中文稳）",
            "model": "moss-image-sft",
            "lora": "sft-8step",
            "lora_strength": 1.0,
            "mode": "exact",
            "sampler": "euler",
            "schedule": "default",
            "steps": 8,
            "cfg": 1.0,
            "shift": 5.0,
            "uncond_skip_layers": [],
            "available": ("moss-image-sft" in TRANSFORMERS) and ("sft-8step" in LORAS),
        },
        {
            "id": "kromoss-50step",
            "label": "🎨 Kromoss 50步（写实精修）",
            "model": "kromoss",
            "lora": "none",
            "lora_strength": 1.0,
            "mode": "spectrum",
            "sampler": "euler",
            "schedule": "default",
            "steps": 50,
            "cfg": 3.0,
            "shift": 5.0,
            "uncond_skip_layers": [22],
            "available": "kromoss" in TRANSFORMERS,
        },
        {
            "id": "sft-50step",
            "label": "🎨 SFT 50步（密集中文海报）",
            "model": "moss-image-sft",
            "lora": "none",
            "lora_strength": 1.0,
            "mode": "spectrum",
            "sampler": "euler",
            "schedule": "default",
            "steps": 50,
            "cfg": 3.0,
            "shift": 5.0,
            "uncond_skip_layers": [22],
            "available": "moss-image-sft" in TRANSFORMERS,
        },
        {
            "id": "heunpp2-sgm",
            "label": "💎 Heun++2 + SGM（50步极致画质 · ~90s）",
            "model": "kromoss" if "kromoss" in TRANSFORMERS else "moss-image-sft",
            "lora": "none",
            "lora_strength": 1.0,
            "mode": "exact",
            "sampler": "heunpp2",
            "schedule": "sgm_uniform",
            "steps": 50,
            "cfg": 3.0,
            "shift": 5.0,
            "uncond_skip_layers": [22],
            "available": True,
        },
    ]
    return {
        "models": models,
        "loras": loras,
        "presets": presets,
        "extra_models_dir": str(EXTRA_MODELS_DIR),
        "lora_dir": str(LORA_DIR),
    }


@app.on_event("startup")
def _startup():
    global pipe, PRIMARY_TR
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
    PRIMARY_TR = pipe.transformer
    TRANSFORMERS["moss-image-sft"] = PRIMARY_TR

    # 预热无 LoRA 主编译核
    neg_e, neg_m = _get_neg_embeds("")
    begin_generation(pipe, mode="exact", total_steps=2)
    _ = pipe(
        "warmup", height=1024, width=1024, num_inference_steps=2, guidance_scale=3.0,
        negative_prompt_embeds=neg_e, negative_prompt_attention_mask=neg_m,
        uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(0),
    )
    torch.cuda.synchronize()

    # 自动扫描并常驻加载 extra_models/ (如 Kromoss) 和 loras/ (4步/8步 Turbo LoRA)
    _scan_and_load_extras()
    print(f"MOSS-Image Web UI & API 已就绪！已加载底模: {list(TRANSFORMERS.keys())}，LoRA: {list(LORAS.keys())}")


class ImageRequest(BaseModel):
    prompt: str
    model: str = "moss-image-sft"
    lora: Optional[str] = "none"
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    mode: str = "spectrum"  # spectrum | turbo | exact
    sampler: str = "euler"  # euler | heun | heunpp2
    schedule: str = "default"  # default | sgm_uniform
    steps: int = Field(default=50, ge=1, le=100)
    cfg: float = Field(default=3.0, ge=1.0, le=15.0)
    shift: float = Field(default=5.0, ge=0.5, le=15.0)
    uncond_skip_layers: Optional[List[int]] = Field(default_factory=lambda: [22])
    seed: Optional[int] = None
    negative_prompt: str = ""


def _resolve_model_and_lora(req: ImageRequest):
    model_id = (req.model or "moss-image-sft").strip().lower()
    if model_id not in TRANSFORMERS:
        raise HTTPException(
            status_code=400,
            detail=f"未知或未加载的底模 '{model_id}'，当前可用: {list(TRANSFORMERS.keys())}",
        )
    lora_id = (req.lora or "none").strip().lower()
    if lora_id in ("", "none", "null"):
        lora_id = "none"
        lora_layers = None
    else:
        if lora_id not in LORAS:
            raise HTTPException(
                status_code=400,
                detail=f"未知或未加载的 LoRA '{lora_id}'，当前可用: {list(LORAS.keys())}",
            )
        lora_layers = LORAS[lora_id]
    return model_id, lora_id, lora_layers


def _save_output(png_bytes: bytes, img: Image.Image, meta: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
    tag = meta.get("model", "sft")
    if meta.get("lora") and meta["lora"] != "none":
        tag = f"{tag}_{meta['lora']}"
    elif meta.get("sampler") and meta["sampler"] != "euler":
        tag = f"{tag}_{meta['sampler']}"
        if meta.get("schedule") == "sgm_uniform":
            tag += "_sgm"
    else:
        tag = f"{tag}_{meta['mode']}"
        if meta.get("schedule") == "sgm_uniform":
            tag += "_sgm"
    safe_tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tag)[:40]
    fname = f"{ts}_{PRECISION}_{safe_tag}_{meta['size']}_s{meta['seed']}.png"
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
    sampler = (req.sampler or "euler").strip().lower()
    if sampler not in SAMPLERS:
        raise HTTPException(status_code=400, detail=f"sampler 仅支持 {SAMPLERS}")
    schedule = (req.schedule or "default").strip().lower()
    if schedule not in SCHEDULES:
        raise HTTPException(status_code=400, detail=f"schedule 仅支持 {SCHEDULES}")

    model_id, lora_id, lora_layers = _resolve_model_and_lora(req)
    w, h = _parse_size(req.size)
    skip_layers = _validate_skip_layers(req.uncond_skip_layers)
    cancel_evt: Optional[threading.Event] = job["cancel_event"] if job else None

    def _update_job(**kwargs):
        if job is not None:
            with JOBS_LOCK:
                job.update(kwargs)

    # 高阶采样器 (Heun / Heun++2) 需在每步内多点试探，必须关闭 Spectrum 跳步预测以保持精度
    effective_mode = "exact" if sampler != "euler" else req.mode

    results = []
    with GEN_LOCK:
        if cancel_evt and cancel_evt.is_set():
            raise _JobCancelled()

        _update_job(status="running", stage="encoding", step=0, total_steps=req.steps, started_at=time.perf_counter())
        pipe.transformer = TRANSFORMERS[model_id]
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, shift=float(req.shift))

        use_cfg = abs(float(req.cfg) - 1.0) > 1e-6
        if use_cfg:
            neg_e, neg_m = _get_neg_embeds(req.negative_prompt)
            effective_skip = skip_layers
        else:
            neg_e, neg_m = None, None
            effective_skip = []

        pos_e, pos_m = pipe.encode_prompt(req.prompt)

        for i in range(req.n):
            if cancel_evt and cancel_evt.is_set():
                raise _JobCancelled()

            seed = (req.seed + i) if req.seed is not None else random.randint(0, 2**32 - 1)
            t0 = time.perf_counter()
            begin_generation(
                pipe,
                mode=effective_mode,
                total_steps=req.steps,
                lora_layers=lora_layers,
                lora_strength=req.lora_strength,
            )
            _update_job(stage="denoising", image_index=i + 1, total_images=req.n, step=0, total_steps=req.steps)

            use_custom = (sampler != "euler") or (schedule != "default")
            nfe_count = None
            if use_custom:
                def _step_cb(stage, step_idx, total_s):
                    _update_job(stage=stage, step=step_idx, total_steps=total_s)

                def _cancel_chk():
                    if cancel_evt and cancel_evt.is_set():
                        raise _JobCancelled()

                img, nfe_count = sample_flow_matching(
                    pipe,
                    prompt_embeds=pos_e,
                    prompt_attention_mask=pos_m,
                    negative_prompt_embeds=neg_e,
                    negative_prompt_attention_mask=neg_m,
                    height=h,
                    width=w,
                    num_inference_steps=req.steps,
                    guidance_scale=float(req.cfg) if use_cfg else 1.0,
                    shift=float(req.shift),
                    sampler=sampler,
                    schedule=schedule,
                    uncond_skip_layers=effective_skip,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    step_callback=_step_cb,
                    cancel_check=_cancel_chk,
                )
            else:
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
                        guidance_scale=float(req.cfg) if use_cfg else 1.0,
                        uncond_skip_layers=effective_skip,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
                finally:
                    pipe.progress_bar = orig_pb

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if sampler != "euler":
                sched_txt = "SGM" if schedule == "sgm_uniform" else "Default"
                stats = f"{sampler}+{sched_txt}（{req.steps} 步 / 等效 {nfe_count} 次评估）"
            else:
                stats = spectrum_stats(pipe)
                if schedule == "sgm_uniform":
                    stats = f"SGM · {stats}"
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
                        "model": model_id,
                        "lora": lora_id,
                        "lora_strength": float(req.lora_strength),
                        "mode": effective_mode,
                        "sampler": sampler,
                        "schedule": schedule,
                        "size": f"{w}x{h}",
                        "steps": req.steps,
                        "cfg": req.cfg,
                        "shift": req.shift,
                        "uncond_skip_layers": effective_skip,
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
        "models": list(TRANSFORMERS.keys()),
        "loras": list(LORAS.keys()),
    }


@app.get("/api/models")
def get_models():
    return _build_catalog()


@app.post("/api/models/reload")
def reload_models(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    if pipe is None:
        raise HTTPException(status_code=503, detail="服务仍在初始化中")
    with GEN_LOCK:
        added = _scan_and_load_extras()
    return {**added, **_build_catalog()}


@app.post("/api/jobs")
def create_job(
    req: ImageRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_auth(authorization, x_api_key)
    if req.mode not in SPECTRUM_PRESETS:
        raise HTTPException(status_code=400, detail="mode 仅支持 spectrum / turbo / exact")
    if (req.sampler or "euler").strip().lower() not in SAMPLERS:
        raise HTTPException(status_code=400, detail=f"sampler 仅支持 {SAMPLERS}")
    if (req.schedule or "default").strip().lower() not in SCHEDULES:
        raise HTTPException(status_code=400, detail=f"schedule 仅支持 {SCHEDULES}")
    _resolve_model_and_lora(req)
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
            "model": meta.get("model", "moss-image-sft"),
            "lora": meta.get("lora", "none"),
            "lora_strength": meta.get("lora_strength", 1.0),
            "mode": meta.get("mode", "spectrum"),
            "sampler": meta.get("sampler", "euler"),
            "schedule": meta.get("schedule", "default"),
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
