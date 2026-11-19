import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path
import torch
from diffusers import DiffusionPipeline
from opt_moss import optimize_pipeline, begin_generation, spectrum_stats, SPECTRUM_PRESETS

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", str(BASE_DIR / "MOSS-Image-SFT"))
OUT_DIR = Path(os.environ.get("MOSS_OUT_DIR", str(BASE_DIR / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

precision = "bf16" if "--bf16" in sys.argv else "fp8" if "--fp8" in sys.argv else "int8"
use_compile = "--no-compile" not in sys.argv

print(f"正在加载 MOSS-Image-SFT [精度={precision.upper()} | compile={use_compile}]...")
t0 = time.perf_counter()
pipe = DiffusionPipeline.from_pretrained(
    MODEL_DIR,
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.to("cuda")
optimize_pipeline(pipe, precision=precision, use_compile=use_compile)

neg_cache = {}


def get_neg_embeds(neg_text):
    if neg_text not in neg_cache:
        neg_cache[neg_text] = pipe.encode_prompt(neg_text)
    return neg_cache[neg_text]


print("正在预热编译核（有缓存约 6~15 秒）...")
tw = time.perf_counter()
neg0, mask0 = get_neg_embeds("")
begin_generation(pipe, mode="exact", total_steps=2)
_ = pipe(
    "warmup", height=1024, width=1024, num_inference_steps=2, guidance_scale=3.0,
    negative_prompt_embeds=neg0, negative_prompt_attention_mask=mask0,
    uncond_skip_layers=[22], generator=torch.Generator("cpu").manual_seed(0),
)
torch.cuda.synchronize()
print(f"全部就绪！（加载+预热共 {time.perf_counter() - t0:.1f}s，预热 {time.perf_counter() - tw:.1f}s）")

print("=" * 68)
print("直接输入中文或英文提示词回车即可出图，支持命令：")
print("  /mode spectrum    默认 4.25x 加速（INT8-ConvRot + Spectrum，推荐）")
print("  /mode turbo       5.38x 极速抽卡模式")
print("  /mode exact       50 步无跳步满算")
print("  /size 1024x1024   修改分辨率（16 的倍数，人像推荐 896x1152，高清 1456x1456）")
print("  /batch 1          单次连续抽卡张数（例如 /batch 4）")
print("  /steps 50         修改步数（官方推荐 >=50）")
print("  /cfg 3.0          修改引导强度（长提示词 3.0，短提示词建议 4.0~5.5）")
print("  /seed 42          固定种子（/seed random 恢复随机）")
print("  /neg 负面词       设置负面提示词（只输入 /neg 清空）")
print("  /bench            一键跑 1456 / 1024 冰箱便签图测速对比")
print("  /quit             退出")
print("=" * 68)

mode = "spectrum"
width, height, steps, cfg, batch_n = 1024, 1024, 50, 3.0, 1
fixed_seed = None
neg_prompt = ""


def generate_one(prompt_text, w, h, st, c, m, seed_val, neg_text):
    t1 = time.perf_counter()
    neg_e, neg_m = get_neg_embeds(neg_text)
    begin_generation(pipe, mode=m, total_steps=st)
    img = pipe(
        prompt_text,
        height=h,
        width=w,
        num_inference_steps=st,
        guidance_scale=c,
        negative_prompt_embeds=neg_e,
        negative_prompt_attention_mask=neg_m,
        uncond_skip_layers=[22],
        generator=torch.Generator("cpu").manual_seed(seed_val),
    ).images[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{ts}_{precision}_{m}_{w}x{h}_s{seed_val}.png"
    img.save(out_path)
    print(f"完成 [{precision.upper()} | {m} | {w}x{h}] 耗时 {dt:.1f}s ({spectrum_stats(pipe)}) -> {out_path}")


while True:
    try:
        tag = f"{precision.upper()}/{mode} | {width}x{height} | x{batch_n} | steps={steps} | cfg={cfg} | seed={fixed_seed or '随机'}"
        text = input(f"\n[{tag}] 提示词> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        break
    if not text:
        continue
    if text == "/quit":
        break
    if text == "/bench":
        demo = (
            'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. '
            'A small sticky note on the refrigerator door is handwritten in Chinese：“\n'
            '记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。”'
        )
        generate_one(demo, 1456, 1456, 50, 3.0, "spectrum", 42, "")
        generate_one(demo, 1024, 1024, 50, 3.0, "spectrum", 42, "")
        generate_one(demo, 1024, 1024, 50, 3.0, "turbo", 42, "")
        continue
    if text.startswith("/mode "):
        m = text.split()[1].lower()
        if m in SPECTRUM_PRESETS:
            mode = m
            print(f"已切换模式：{mode}")
        else:
            print("可选模式：spectrum / turbo / exact")
        continue
    if text.startswith("/size "):
        try:
            w, h = map(int, text.split()[1].lower().split("x"))
            assert w > 0 and h > 0 and w % 16 == 0 and h % 16 == 0
            width, height = w, h
            print(f"已设置分辨率：{width}x{height}")
        except Exception:
            print("格式错误，请用 /size 1024x1024（宽高必须为 16 的正整数倍）")
        continue
    if text.startswith("/batch "):
        batch_n = max(1, int(text.split()[1]))
        print(f"已设置单次抽卡张数：{batch_n}")
        continue
    if text.startswith("/steps "):
        steps = max(1, int(text.split()[1]))
        print(f"已设置步数：{steps}")
        continue
    if text.startswith("/cfg "):
        cfg = float(text.split()[1])
        print(f"已设置 CFG：{cfg}")
        continue
    if text.startswith("/seed "):
        val = text.split()[1].lower()
        fixed_seed = None if val == "random" else int(val)
        print(f"已设置种子：{fixed_seed or '随机'}")
        continue
    if text == "/neg" or text.startswith("/neg "):
        neg_prompt = text[4:].strip()
        print(f"已设置负面提示词：{neg_prompt or '(空)'}")
        continue

    base_seed = fixed_seed if fixed_seed is not None else random.randint(0, 2**32 - 1)
    for i in range(batch_n):
        s_val = (base_seed + i) if fixed_seed is not None else random.randint(0, 2**32 - 1)
        generate_one(text, width, height, steps, cfg, mode, s_val, neg_prompt)
