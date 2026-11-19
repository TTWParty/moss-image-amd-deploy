import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from diffusers import DiffusionPipeline

from opt_moss import begin_generation, optimize_pipeline, spectrum_stats

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", str(BASE_DIR / "MOSS-Image-SFT"))
OUT_DIR = Path(os.environ.get("MOSS_OUT_DIR", str(BASE_DIR / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRECISION = os.environ.get("MOSS_PRECISION", "int8")
PORT = int(os.environ.get("MOSS_GRADIO_PORT", "7860"))
SHARE = os.environ.get("MOSS_GRADIO_SHARE", "1") != "0"
UI_AUTH = os.environ.get("MOSS_UI_AUTH", "").strip()  # 格式: user:pass

GEN_LOCK = threading.Lock()
neg_cache = {}

print(f"正在加载 MOSS-Image-SFT [precision={PRECISION}]...")
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


def _get_neg_embeds(neg_text: str):
    if neg_text not in neg_cache:
        neg_cache[neg_text] = pipe.encode_prompt(neg_text)
    return neg_cache[neg_text]


# 启动预热 2 步
neg0, mask0 = _get_neg_embeds("")
begin_generation(pipe, mode="exact", total_steps=2)
_ = pipe(
    "warmup", height=1024, width=1024, num_inference_steps=2, guidance_scale=3.0,
    negative_prompt_embeds=neg0, negative_prompt_attention_mask=mask0,
    uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(0),
)
torch.cuda.synchronize()
print("预热完成，正在启动 Gradio WebUI...")


def run_generate(prompt, neg_prompt, mode, width, height, steps, cfg, seed, count):
    if not prompt or not prompt.strip():
        raise gr.Error("请先输入提示词")
    w = max(256, min(2048, (int(width) // 16) * 16))
    h = max(256, min(2048, (int(height) // 16) * 16))
    n = max(1, min(4, int(count)))
    fixed_seed = None if seed is None or int(seed) < 0 else int(seed)

    imgs = []
    logs = []
    with GEN_LOCK:
        neg_e, neg_m = _get_neg_embeds((neg_prompt or "").strip())
        for i in range(n):
            s = (fixed_seed + i) if fixed_seed is not None else random.randint(0, 2**32 - 1)
            t0 = time.perf_counter()
            begin_generation(pipe, mode=mode, total_steps=int(steps))
            img = pipe(
                prompt.strip(),
                height=h,
                width=w,
                num_inference_steps=int(steps),
                guidance_scale=float(cfg),
                negative_prompt_embeds=neg_e,
                negative_prompt_attention_mask=neg_m,
                uncond_skip_layers=[22],
                generator=torch.Generator("cpu").manual_seed(s),
            ).images[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"{ts}_{PRECISION}_{mode}_{w}x{h}_s{s}.png"
            img.save(out_path)
            imgs.append(img)
            logs.append(f"[{i+1}/{n}] 耗时 {dt:.1f}s | seed={s} | {spectrum_stats(pipe)}")
    return imgs, "\n".join(logs)


ANTI_COLLAGE = (
    "collage, grid, split screen, multi-panel, multiple views, diptych, triptych, "
    "photo strip, border, frame, worst quality, bad anatomy"
)

with gr.Blocks(title="MOSS-Image WebUI") as demo:
    gr.Markdown(f"## MOSS-Image WebUI（AMD ROCm · `{PRECISION.upper()}` 加速版）")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="正向提示词（支持中英文 / 换行排字 / JSON Layout）", lines=5)
            neg_prompt = gr.Textbox(label="负面提示词", value=ANTI_COLLAGE, lines=2)
            mode = gr.Dropdown(
                choices=["spectrum", "turbo", "exact"],
                value="spectrum",
                label="速度模式（spectrum 推荐 / turbo 极速 / exact 50步满算）",
            )
            with gr.Row():
                width = gr.Slider(256, 2048, value=896, step=16, label="宽度")
                height = gr.Slider(256, 2048, value=1152, step=16, label="高度")
            with gr.Row():
                steps = gr.Slider(10, 80, value=50, step=1, label="步数")
                cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="CFG")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="随机种子（-1 为随机）")
                count = gr.Slider(1, 4, value=1, step=1, label="生成张数")
            btn = gr.Button("开始生成", variant="primary")
        with gr.Column(scale=6):
            gallery = gr.Gallery(label="生成结果", columns=2, height="auto")
            info = gr.Textbox(label="生成信息", lines=3, interactive=False)

    btn.click(
        fn=run_generate,
        inputs=[prompt, neg_prompt, mode, width, height, steps, cfg, seed, count],
        outputs=[gallery, info],
    )

auth_tuple = tuple(UI_AUTH.split(":", 1)) if ":" in UI_AUTH else None
demo.queue().launch(server_name="0.0.0.0", server_port=PORT, share=SHARE, auth=auth_tuple)
