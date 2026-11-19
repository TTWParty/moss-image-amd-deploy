import math
import types
import numpy as np
import torch
from torch.nn import functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput

FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # AMD CDNA 3 fnuz = 240.0


def _rms_norm(x, weight):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return y.to(x.dtype) * weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _hadamard_64(device):
    h = torch.tensor([[1.0]], dtype=torch.float32)
    for _ in range(6):
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / 8.0).to(device=device, dtype=torch.bfloat16).contiguous()


def _quant_weight_int8_rot(w: torch.Tensor, h64: torch.Tensor):
    n, k = w.shape
    w_rot = (w.float().reshape(n, k // 64, 64) @ h64.float()).reshape(n, k)
    scale = w_rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    w_i8 = (w_rot / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return w_i8.t(), scale.reshape(1, n).contiguous()


def _int8_rot_linear(x, w_i8_t, w_scale, h64):
    orig = x.shape
    k = orig[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    pad = (16 - (m % 16)) % 16
    x_pad = F.pad(x2d, (0, 0, 0, pad))
    x_rot = (x_pad.reshape(-1, 64) @ h64).reshape(-1, k)
    amax = x_rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).float()
    x_i8 = (x_rot.float() * (127.0 / amax)).round().clamp(-127, 127).to(torch.int8).contiguous()
    out_i32 = torch._int_mm(x_i8, w_i8_t)
    out = (out_i32.float() * (amax / 127.0) * w_scale).to(x.dtype)
    return out[:m].reshape(*orig[:-1], out.shape[-1])


def _quant_weight_fp8(w: torch.Tensor):
    amax = w.abs().amax().clamp(min=1e-12).float()
    scale = amax / FP8_MAX
    w_fp8 = (w.float() * (FP8_MAX / amax)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    return w_fp8.t(), scale


def _fp8_linear(x, w_fp8_t, w_scale):
    orig = x.shape
    x2d = x.reshape(-1, orig[-1])
    m = x2d.shape[0]
    pad = (16 - (m % 16)) % 16
    x_pad = F.pad(x2d, (0, 0, 0, pad))
    amax = x_pad.abs().amax().clamp(min=1e-12).float()
    x_scale = amax / FP8_MAX
    x_fp8 = (x_pad.float() * (FP8_MAX / amax)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    out = torch._scaled_mm(x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale, out_dtype=x.dtype)
    return out[:m].reshape(*orig[:-1], out.shape[-1])


def _chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.to(dtype=torch.float32, device="cpu")
    cols = [torch.ones_like(x)]
    if degree >= 1:
        cols.append(x)
    for _ in range(2, degree + 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return torch.stack(cols[: degree + 1], dim=-1)


class _ChebyshevForecaster:
    """Spectrum-style ridge-regularized Chebyshev + linear blended feature forecaster."""

    def __init__(self, degree=4, ridge_lambda=0.1, max_history=8, blend_weight=0.5):
        self.degree = degree
        self.ridge_lambda = ridge_lambda
        self.max_history = max_history
        self.blend_weight = blend_weight
        self.anchors = []

    def ready(self):
        return len(self.anchors) >= self.degree + 1

    def update(self, coord: float, feature: torch.Tensor):
        self.anchors.append((float(coord), feature.detach().clone()))
        if len(self.anchors) > self.max_history:
            self.anchors.pop(0)

    def _weights(self, coord: float):
        coords = torch.tensor([c for c, _ in self.anchors], dtype=torch.float32)
        design = _chebyshev_basis(coords, self.degree).numpy().astype(np.float64)
        lhs = design.T @ design + self.ridge_lambda * np.eye(design.shape[1], dtype=np.float64)
        solved = np.linalg.solve(lhs, design.T)
        phi_np = _chebyshev_basis(torch.tensor([float(coord)]), self.degree).numpy().astype(np.float64)
        spectral = (phi_np @ solved).reshape(-1)

        k = len(self.anchors)
        linear = np.zeros(k, dtype=np.float64)
        prev_c, last_c = self.anchors[-2][0], self.anchors[-1][0]
        spacing = last_c - prev_c
        if abs(spacing) <= 1e-12:
            linear[-1] = 1.0
        else:
            ratio = (float(coord) - last_c) / spacing
            linear[-2] = -ratio
            linear[-1] = 1.0 + ratio
        return (self.blend_weight * spectral + (1.0 - self.blend_weight) * linear).tolist()

    def predict(self, coord: float) -> torch.Tensor:
        weights = self._weights(coord)
        ref = self.anchors[-1][1]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for w, (_, feat) in zip(weights, self.anchors):
            if w != 0.0:
                acc.add_(feat.float(), alpha=float(w))
        finfo = torch.finfo(ref.dtype)
        torch.nan_to_num_(acc, nan=0.0, posinf=float(finfo.max), neginf=float(finfo.min))
        acc.clamp_(min=float(finfo.min), max=float(finfo.max))
        return acc.to(ref.dtype)


SPECTRUM_PRESETS = {
    "exact": None,
    "spectrum": dict(warmup=8, tail=1, max_skip=2, window=2.0, flex=0.75, degree=4, ridge=0.1, history=8, blend=0.5),
    "turbo": dict(warmup=6, tail=1, max_skip=3, window=3.0, flex=0.75, degree=4, ridge=0.1, history=8, blend=0.5),
}


class _BranchState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.forecaster = _ChebyshevForecaster(cfg["degree"], cfg["ridge"], cfg["history"], cfg["blend"])
        self.step = 0
        self.consec_forecasts = 0
        self.window = float(cfg["window"])
        self.actual_count = 0
        self.forecast_count = 0

    def should_run_actual(self, total_steps: int) -> bool:
        s = self.step
        cfg = self.cfg
        if s < cfg["warmup"]:
            return True
        if cfg["tail"] > 0 and s >= max(0, total_steps - cfg["tail"]):
            return True
        if not self.forecaster.ready():
            return True
        if self.consec_forecasts >= cfg["max_skip"]:
            return True
        interval = max(1, int(math.floor(self.window)))
        if (self.consec_forecasts + 1) % interval == 0:
            return True
        return False


def _make_shared_block_fns(heads, precision="int8", use_compile=True):
    if precision == "int8":
        def _pre(x, rope, mod, n1, qn, kn, qw, qs, h64):
            sa, sca, ga, sf, scf, gf = mod.unbind(1)
            y = (_rms_norm(x, n1) * (1 + sca[:, None]) + sa[:, None]).to(torch.bfloat16)
            q, k, v = _int8_rot_linear(y, qw, qs, h64).chunk(3, -1)
            q = _rotate(_rms_norm(q.unflatten(-1, (heads, -1)), qn), rope).transpose(1, 2).to(v.dtype)
            k = _rotate(_rms_norm(k.unflatten(-1, (heads, -1)), kn), rope).transpose(1, 2).to(v.dtype)
            return q, k, v.unflatten(-1, (heads, -1)).transpose(1, 2), ga, sf, scf, gf

        def _post(x, attn, ga, sf, scf, gf, n2, ow, os_, uw, us, dw, ds, h64):
            x = x + _int8_rot_linear(attn.transpose(1, 2).flatten(2), ow, os_, h64) * ga[:, None]
            y = (_rms_norm(x, n2) * (1 + scf[:, None]) + sf[:, None]).to(torch.bfloat16)
            h, g = _int8_rot_linear(y, uw, us, h64).chunk(2, -1)
            return x + _int8_rot_linear(_situ(h, g), dw, ds, h64) * gf[:, None]

    elif precision == "fp8":
        def _pre(x, rope, mod, n1, qn, kn, qw, qs, _h64):
            sa, sca, ga, sf, scf, gf = mod.unbind(1)
            y = (_rms_norm(x, n1) * (1 + sca[:, None]) + sa[:, None]).to(torch.bfloat16)
            q, k, v = _fp8_linear(y, qw, qs).chunk(3, -1)
            q = _rotate(_rms_norm(q.unflatten(-1, (heads, -1)), qn), rope).transpose(1, 2).to(v.dtype)
            k = _rotate(_rms_norm(k.unflatten(-1, (heads, -1)), kn), rope).transpose(1, 2).to(v.dtype)
            return q, k, v.unflatten(-1, (heads, -1)).transpose(1, 2), ga, sf, scf, gf

        def _post(x, attn, ga, sf, scf, gf, n2, ow, os_, uw, us, dw, ds, _h64):
            x = x + _fp8_linear(attn.transpose(1, 2).flatten(2), ow, os_) * ga[:, None]
            y = (_rms_norm(x, n2) * (1 + scf[:, None]) + sf[:, None]).to(torch.bfloat16)
            h, g = _fp8_linear(y, uw, us).chunk(2, -1)
            return x + _fp8_linear(_situ(h, g), dw, ds) * gf[:, None]

    else:  # bf16
        def _pre(x, rope, mod, n1, qn, kn, qw, _qs, _h64):
            sa, sca, ga, sf, scf, gf = mod.unbind(1)
            y = (_rms_norm(x, n1) * (1 + sca[:, None]) + sa[:, None]).to(qw.dtype)
            q, k, v = F.linear(y, qw).chunk(3, -1)
            q = _rotate(_rms_norm(q.unflatten(-1, (heads, -1)), qn), rope).transpose(1, 2).to(v.dtype)
            k = _rotate(_rms_norm(k.unflatten(-1, (heads, -1)), kn), rope).transpose(1, 2).to(v.dtype)
            return q, k, v.unflatten(-1, (heads, -1)).transpose(1, 2), ga, sf, scf, gf

        def _post(x, attn, ga, sf, scf, gf, n2, ow, _os, uw, _us, dw, _ds, _h64):
            x = x + F.linear(attn.transpose(1, 2).flatten(2), ow) * ga[:, None]
            y = (_rms_norm(x, n2) * (1 + scf[:, None]) + sf[:, None]).to(ow.dtype)
            h, g = F.linear(y, uw).chunk(2, -1)
            return x + F.linear(_situ(h, g), dw) * gf[:, None]

    if use_compile:
        _pre = torch.compile(_pre, backend="inductor", fullgraph=True, dynamic=True, mode="default")
        _post = torch.compile(_post, backend="inductor", fullgraph=True, dynamic=True, mode="default")
    return _pre, _post


def _cached_transformer_forward(
    self, hidden_states, timestep, encoder_hidden_states,
    encoder_attention_mask=None, return_dict=True, skip_layers=None
):
    skip_layers = frozenset(() if skip_layers is None else tuple(skip_layers))
    batch, _, height, width = hidden_states.shape
    mask = encoder_attention_mask
    if mask is None:
        mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
    half = self.config.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
    mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
    head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)

    pre_fn, post_fn = self._shared_block_fns
    h64 = self._h64
    spec_cfg = self._spectrum_cfg
    total_steps = self._spectrum_total_steps
    ran_actual = False
    outputs = []

    for index in range(batch):
        bkey = (index, encoder_hidden_states.data_ptr(), skip_layers)
        use_forecast = False
        branch = None
        coord = 0.0

        if spec_cfg is not None and total_steps > spec_cfg["warmup"] + 2:
            branch = self._spectrum_branches.get(bkey)
            if branch is None:
                branch = _BranchState(spec_cfg)
                self._spectrum_branches[bkey] = branch
            coord = 2.0 * float(branch.step) / float(max(1, total_steps - 1)) - 1.0
            use_forecast = not branch.should_run_actual(total_steps)

        if use_forecast:
            feature = branch.forecaster.predict(coord)
            branch.consec_forecasts += 1
            branch.forecast_count += 1
            branch.step += 1
        else:
            ran_actual = True
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(self.video_embedder.weight.dtype).contiguous())

            cache_key = (index, height, width, encoder_hidden_states.data_ptr(), encoder_hidden_states.shape[1])
            cached = self._step_cache.get(cache_key)
            if cached is None:
                context = encoder_hidden_states[index:index + 1, mask[index].bool()]
                rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
                text = self.text_embedder(context, rope[:, height * width:])
                self._step_cache[cache_key] = (rope, text)
            else:
                rope, text = cached

            x = torch.cat((video, text), 1)
            m_i = mod[index:index + 1]
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    q, k, v, gate_a, shift_f, scale_f, gate_f = pre_fn(
                        x, rope, m_i,
                        layer.norm1.weight, layer.q_norm.weight, layer.k_norm.weight,
                        layer._w_qkv, layer._s_qkv, h64,
                    )
                    attn_out = F.scaled_dot_product_attention(q, k, v)
                    x = post_fn(
                        x, attn_out, gate_a, shift_f, scale_f, gate_f,
                        layer.norm2.weight,
                        layer._w_o, layer._s_o,
                        layer._w_up, layer._s_up,
                        layer._w_dn, layer._s_dn, h64,
                    )
            feature = x[:, :height * width]
            if branch is not None:
                branch.forecaster.update(coord, feature)
                if branch.step >= spec_cfg["warmup"]:
                    branch.window = min(float(spec_cfg["history"]), branch.window + spec_cfg["flex"])
                branch.consec_forecasts = 0
                branch.actual_count += 1
                branch.step += 1

        x_out = self.final_norm_video(feature.to(self.final_norm_video.weight.dtype))
        x_out = x_out * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
        x_out = self.final_linear_video(x_out.to(self.final_linear_video.weight.dtype))
        outputs.append(x_out.transpose(1, 2).reshape(1, -1, height, width))

    sample = torch.cat(outputs)
    # 防止 ROCm 异步任务队列在无同步长循环中溢出导致 Segmentation fault
    if ran_actual:
        torch.cuda.synchronize()
    return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


def begin_generation(pipe, mode="spectrum", total_steps=50):
    tr = pipe.transformer
    tr._step_cache.clear()
    tr._spectrum_branches.clear()
    tr._spectrum_cfg = SPECTRUM_PRESETS.get(mode, None)
    tr._spectrum_total_steps = int(total_steps)


def spectrum_stats(pipe):
    branches = list(pipe.transformer._spectrum_branches.values())
    if not branches:
        return "exact 模式（满算无跳步）"
    b = branches[0]
    return f"实算 {b.actual_count}/{b.actual_count + b.forecast_count} 步，跳过 {b.forecast_count} 步"


def optimize_pipeline(pipe, precision="int8", use_compile=True):
    """precision: 'int8' (INT8-ConvRot 推荐) | 'bf16' (原版满精度) | 'fp8' (AMD fnuz)"""
    if precision not in ("int8", "bf16", "fp8"):
        raise ValueError(f"Unsupported precision: {precision}")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tr = pipe.transformer
    tr._step_cache = {}
    tr._spectrum_branches = {}
    tr._spectrum_cfg = None
    tr._spectrum_total_steps = 50
    tr._h64 = _hadamard_64("cuda")
    dummy = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        for layer in tr.layers:
            if hasattr(layer, "_w_qkv"):
                continue
            q, k, v = layer.proj["q"], layer.proj["k"], layer.proj["v"]
            w_qkv = torch.cat([q.weight, k.weight, v.weight], dim=0).contiguous()
            w_o = layer.proj["o"].weight.contiguous()
            w_up = layer.proj["ffn_up_gate"].weight.contiguous()
            w_dn = layer.proj["ffn_down"].weight.contiguous()

            if precision == "int8":
                layer._w_qkv, layer._s_qkv = _quant_weight_int8_rot(w_qkv, tr._h64)
                layer._w_o, layer._s_o = _quant_weight_int8_rot(w_o, tr._h64)
                layer._w_up, layer._s_up = _quant_weight_int8_rot(w_up, tr._h64)
                layer._w_dn, layer._s_dn = _quant_weight_int8_rot(w_dn, tr._h64)
            elif precision == "fp8":
                layer._w_qkv, layer._s_qkv = _quant_weight_fp8(w_qkv)
                layer._w_o, layer._s_o = _quant_weight_fp8(w_o)
                layer._w_up, layer._s_up = _quant_weight_fp8(w_up)
                layer._w_dn, layer._s_dn = _quant_weight_fp8(w_dn)
            else:
                layer._w_qkv, layer._s_qkv = w_qkv, dummy
                layer._w_o, layer._s_o = w_o, dummy
                layer._w_up, layer._s_up = w_up, dummy
                layer._w_dn, layer._s_dn = w_dn, dummy

            del layer.proj["q"], layer.proj["k"], layer.proj["v"]
            del layer.proj["o"], layer.proj["ffn_up_gate"], layer.proj["ffn_down"]
    torch.cuda.empty_cache()

    tr._shared_block_fns = _make_shared_block_fns(
        tr.layers[0].heads, precision=precision, use_compile=use_compile
    )
    tr.forward = types.MethodType(_cached_transformer_forward, tr)
    return pipe
