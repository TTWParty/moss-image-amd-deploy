import gc
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline
from opt_moss import optimize_pipeline, begin_generation, spectrum_stats

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", str(BASE_DIR / "MOSS-Image-SFT"))
OUT_DIR = BASE_DIR / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 1456 if "--1456" in sys.argv else 1024
STEPS = 50
CFG = 3.0
SEED = 42
PROMPT = (
    'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. '
    'A small sticky note on the refrigerator door is handwritten in Chinese：“\n'
    '记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。”'
)

PLAN = [
    ("bf16", ["exact", "spectrum", "turbo"]),
    ("int8", ["exact", "spectrum", "turbo"]),
    ("fp8",  ["spectrum"]),
]


def load_pipe(precision):
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_DIR,
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.to("cuda")
    optimize_pipeline(pipe, precision=precision, use_compile=True)
    return pipe


def calc_metrics(ref_img: Image.Image, img: Image.Image):
    a = np.asarray(ref_img, dtype=np.float32)
    b = np.asarray(img, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    psnr = 99.9 if mse < 1e-10 else 10.0 * math.log10((255.0 ** 2) / mse)
    return psnr, mae


def make_grid(records, save_path):
    cols = 4
    rows = math.ceil(len(records) / cols)
    tile = 512
    header = 64
    canvas = Image.new("RGB", (cols * tile, rows * (tile + header)), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font1 = ImageFont.load_default(size=20)
        font2 = ImageFont.load_default(size=16)
    except TypeError:
        font1 = font2 = ImageFont.load_default()

    for idx, rec in enumerate(records):
        r, c = divmod(idx, cols)
        x0, y0 = c * tile, r * (tile + header)
        thumb = rec["image"].resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x0, y0 + header))
        label1 = f"{rec['precision'].upper()} + {rec['mode']}"
        if rec["is_ref"]:
            label2 = f"{rec['sec']:.1f}s (1.00x) | BASELINE"
        else:
            label2 = f"{rec['sec']:.1f}s ({rec['speedup']:.2f}x) | PSNR {rec['psnr']:.1f}dB"
        draw.text((x0 + 12, y0 + 8), label1, fill=(255, 255, 255), font=font1)
        draw.text((x0 + 12, y0 + 36), label2, fill=(130, 220, 255), font=font2)

    canvas.save(save_path)


def main():
    print(f"=== MOSS-Image 全版本对比测速 [{SIZE}x{SIZE} | {STEPS}步 | seed={SEED}] ===")
    pos_embeds = pos_mask = neg_embeds = neg_mask = None
    records = []

    for precision, modes in PLAN:
        print(f"\n>>> 加载精度版本: {precision.upper()} ...")
        pipe = load_pipe(precision)
        if pos_embeds is None:
            pe, pm = pipe.encode_prompt(PROMPT)
            ne, nm = pipe.encode_prompt("")
            pos_embeds, pos_mask = pe.detach().clone(), pm.detach().clone()
            neg_embeds, neg_mask = ne.detach().clone(), nm.detach().clone()

        tw = time.perf_counter()
        begin_generation(pipe, mode="exact", total_steps=2)
        _ = pipe(
            prompt=None,
            prompt_embeds=pos_embeds,
            prompt_attention_mask=pos_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_attention_mask=neg_mask,
            height=SIZE, width=SIZE, num_inference_steps=2, guidance_scale=CFG,
            uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(SEED),
        )
        torch.cuda.synchronize()
        print(f"    {precision.upper()} 编译预热完成：{time.perf_counter() - tw:.1f}s")

        for mode in modes:
            t0 = time.perf_counter()
            begin_generation(pipe, mode=mode, total_steps=STEPS)
            img = pipe(
                prompt=None,
                prompt_embeds=pos_embeds,
                prompt_attention_mask=pos_mask,
                negative_prompt_embeds=neg_embeds,
                negative_prompt_attention_mask=neg_mask,
                height=SIZE, width=SIZE, num_inference_steps=STEPS, guidance_scale=CFG,
                uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(SEED),
            ).images[0]
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            stats = spectrum_stats(pipe)
            fname = OUT_DIR / f"{SIZE}_{precision}_{mode}.png"
            img.save(fname)
            print(f"    [{precision.upper():4s} + {mode:8s}] {sec:5.1f}s | {stats} -> {fname}")
            records.append(dict(
                precision=precision, mode=mode, sec=sec, stats=stats, image=img, path=str(fname),
            ))

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    ref_rec = records[0]
    ref_sec = ref_rec["sec"]
    ref_img = ref_rec["image"]

    print("\n" + "=" * 86)
    print(f"{'版本':18s} | {'耗时':>7s} | {'提速':>6s} | {'PSNR(越高越像原版)':>18s} | {'MAE(越低越好)':>13s} | 说明")
    print("-" * 86)
    for idx, rec in enumerate(records):
        rec["is_ref"] = (idx == 0)
        rec["speedup"] = ref_sec / rec["sec"]
        if idx == 0:
            rec["psnr"], rec["mae"] = 99.9, 0.0
            psnr_str, mae_str = "基准", "0.00"
        else:
            rec["psnr"], rec["mae"] = calc_metrics(ref_img, rec["image"])
            psnr_str, mae_str = f"{rec['psnr']:.2f} dB", f"{rec['mae']:.2f}"
        name = f"{rec['precision'].upper()} + {rec['mode']}"
        print(f"{name:18s} | {rec['sec']:6.1f}s | {rec['speedup']:5.2f}x | {psnr_str:>18s} | {mae_str:>13s} | {rec['stats']}")
    print("=" * 86)

    grid_path = OUT_DIR / f"compare_grid_{SIZE}.png"
    make_grid(records, grid_path)
    print(f"\n7 宫格对比图已生成 -> {grid_path}")


if __name__ == "__main__":
    main()
