from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


@dataclass(frozen=True)
class ModelConfig:
    model: str = "edsr_lite"
    scale: int = 4
    features: int = 64
    blocks: int = 8
    groups: int = 4
    heads: int = 4
    window_size: int = 8
    mlp_ratio: float = 2.0
    token_count: int = 0
    channel_reduction: int = 4
    residual_scale: float = 0.1
    dtype: str = "bfloat16"
    global_residual: bool = True


def _dtype(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}")


def apply_preset(config: ModelConfig, preset: str) -> ModelConfig:
    if preset == "custom":
        return config
    if preset == "edsr_lite":
        return ModelConfig(
            model="edsr_lite",
            scale=config.scale,
            dtype=config.dtype,
            global_residual=config.global_residual,
        )
    if preset == "hat_atd_tiny":
        return ModelConfig(
            model="hat_atd",
            scale=config.scale,
            features=48,
            groups=2,
            blocks=2,
            heads=3,
            window_size=8,
            mlp_ratio=2.0,
            token_count=32,
            channel_reduction=4,
            residual_scale=0.2,
            dtype=config.dtype,
            global_residual=config.global_residual,
        )
    if preset == "hat_atd_base":
        return ModelConfig(
            model="hat_atd",
            scale=config.scale,
            features=96,
            groups=4,
            blocks=4,
            heads=6,
            window_size=8,
            mlp_ratio=2.0,
            token_count=64,
            channel_reduction=4,
            residual_scale=0.15,
            dtype=config.dtype,
            global_residual=config.global_residual,
        )
    if preset == "hat_atd_large":
        return ModelConfig(
            model="hat_atd",
            scale=config.scale,
            features=180,
            groups=6,
            blocks=6,
            heads=6,
            window_size=8,
            mlp_ratio=2.0,
            token_count=128,
            channel_reduction=6,
            residual_scale=0.1,
            dtype=config.dtype,
            global_residual=config.global_residual,
        )
    if preset == "hat_atd_xlarge":
        return ModelConfig(
            model="hat_atd",
            scale=config.scale,
            features=240,
            groups=8,
            blocks=8,
            heads=8,
            window_size=8,
            mlp_ratio=2.0,
            token_count=192,
            channel_reduction=8,
            residual_scale=0.08,
            dtype=config.dtype,
            global_residual=config.global_residual,
        )
    if preset == "hat_atd_xxlarge":
        return ModelConfig(
            model="hat_atd",
            scale=config.scale,
            features=288,
            groups=10,
            blocks=8,
            heads=8,
            window_size=8,
            mlp_ratio=2.0,
            token_count=256,
            channel_reduction=8,
            residual_scale=0.06,
            dtype=config.dtype,
            global_residual=config.global_residual,
        )
    if preset == "atd_v2_tiny":
        return ModelConfig(
            model="atd_v2",
            scale=config.scale,
            features=48,
            groups=2,
            blocks=2,
            heads=3,
            window_size=8,
            mlp_ratio=2.0,
            token_count=32,
            channel_reduction=4,
            residual_scale=0.12,
            dtype=config.dtype,
            global_residual=False,
        )
    if preset == "atd_v2_base":
        return ModelConfig(
            model="atd_v2",
            scale=config.scale,
            features=144,
            groups=6,
            blocks=4,
            heads=6,
            window_size=8,
            mlp_ratio=2.0,
            token_count=128,
            channel_reduction=6,
            residual_scale=0.10,
            dtype=config.dtype,
            global_residual=False,
        )
    if preset == "atd_v2_large":
        return ModelConfig(
            model="atd_v2",
            scale=config.scale,
            features=192,
            groups=8,
            blocks=4,
            heads=8,
            window_size=8,
            mlp_ratio=2.0,
            token_count=192,
            channel_reduction=8,
            residual_scale=0.08,
            dtype=config.dtype,
            global_residual=False,
        )
    if preset == "atd_v2_xlarge":
        return ModelConfig(
            model="atd_v2",
            scale=config.scale,
            features=240,
            groups=10,
            blocks=4,
            heads=8,
            window_size=8,
            mlp_ratio=2.0,
            token_count=256,
            channel_reduction=8,
            residual_scale=0.06,
            dtype=config.dtype,
            global_residual=False,
        )
    if preset == "atd_v2_xxlarge":
        return ModelConfig(
            model="atd_v2",
            scale=config.scale,
            features=288,
            groups=12,
            blocks=4,
            heads=8,
            window_size=8,
            mlp_ratio=2.0,
            token_count=320,
            channel_reduction=8,
            residual_scale=0.05,
            dtype=config.dtype,
            global_residual=False,
        )
    raise ValueError(f"Unknown model preset: {preset}")


def pixel_shuffle(x: jnp.ndarray, scale: int) -> jnp.ndarray:
    batch, height, width, channels = x.shape
    if channels % (scale * scale) != 0:
        raise ValueError(f"Channels must be divisible by scale^2, got {channels=} {scale=}")
    out_channels = channels // (scale * scale)
    x = x.reshape(batch, height, width, scale, scale, out_channels)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(batch, height * scale, width * scale, out_channels)


def _window_partition(x: jnp.ndarray, window_size: int) -> jnp.ndarray:
    batch, height, width, channels = x.shape
    if height % window_size or width % window_size:
        raise ValueError(
            f"Input must be divisible by window_size, got {(height, width)=} {window_size=}"
        )
    x = x.reshape(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(-1, window_size * window_size, channels)


def _window_reverse(
    windows: jnp.ndarray,
    *,
    batch: int,
    height: int,
    width: int,
    window_size: int,
) -> jnp.ndarray:
    channels = windows.shape[-1]
    x = windows.reshape(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(batch, height, width, channels)


def _relative_position_index(window_size: int) -> np.ndarray:
    coords = np.stack(np.meshgrid(np.arange(window_size), np.arange(window_size), indexing="ij"))
    coords_flat = coords.reshape(2, -1)
    relative = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative = relative.transpose(1, 2, 0)
    relative[:, :, 0] += window_size - 1
    relative[:, :, 1] += window_size - 1
    relative[:, :, 0] *= 2 * window_size - 1
    return relative.sum(-1).astype(np.int32)


class ResidualBlock(nn.Module):
    features: int
    residual_scale: float
    dtype_name: str

    @nn.compact
    def __call__(self, x):
        dtype = _dtype(self.dtype_name)
        residual = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dtype)(x)
        residual = nn.relu(residual)
        residual = nn.Conv(self.features, (3, 3), padding="SAME", dtype=dtype)(residual)
        return x + residual * self.residual_scale


class EDSRLite(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        dtype = _dtype(self.config.dtype)
        x_in = x
        x = x.astype(dtype)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        skip = x
        for _ in range(self.config.blocks):
            x = ResidualBlock(
                self.config.features,
                self.config.residual_scale,
                self.config.dtype,
            )(x)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        x = x + skip
        residual = nn.Conv(3, (3, 3), padding="SAME", dtype=dtype)(x)
        return x_in + residual.astype(jnp.float32)


class WindowAttention(nn.Module):
    features: int
    num_heads: int
    window_size: int
    dtype_name: str

    def setup(self):
        self.relative_position_index = _relative_position_index(self.window_size)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.dtype_name)
        _, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        if channels % self.num_heads:
            raise ValueError(f"features must be divisible by heads, got {channels=} {self.num_heads=}")

        qkv = nn.Dense(self.features * 3, use_bias=True, dtype=dtype)(x)
        qkv = qkv.reshape(x.shape[0], tokens, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = head_dim**-0.5
        logits = jnp.einsum("bhtd,bhsd->bhts", q.astype(jnp.float32), k.astype(jnp.float32))
        logits = logits * scale

        bias_table = self.param(
            "relative_position_bias_table",
            nn.initializers.truncated_normal(stddev=0.02),
            ((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads),
            jnp.float32,
        )
        bias = bias_table[jnp.asarray(self.relative_position_index.reshape(-1))]
        bias = bias.reshape(tokens, tokens, self.num_heads)
        bias = jnp.transpose(bias, (2, 0, 1))
        logits = logits + bias[None, :, :, :]

        attention = nn.softmax(logits, axis=-1).astype(dtype)
        x = jnp.einsum("bhts,bhsd->bhtd", attention, v)
        x = jnp.transpose(x, (0, 2, 1, 3)).reshape(x.shape[0], tokens, channels)
        return nn.Dense(self.features, dtype=dtype)(x)


class ChannelAttention(nn.Module):
    features: int
    reduction: int
    dtype_name: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.dtype_name)
        hidden = max(self.features // self.reduction, 8)
        pooled = jnp.mean(x, axis=(1, 2))
        gate = nn.Dense(hidden, dtype=dtype)(pooled)
        gate = nn.gelu(gate)
        gate = nn.Dense(self.features, dtype=dtype)(gate)
        gate = nn.sigmoid(gate).reshape(x.shape[0], 1, 1, self.features)
        return x * gate.astype(dtype)


class ConvFFN(nn.Module):
    features: int
    mlp_ratio: float
    dtype_name: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.dtype_name)
        hidden = int(round(self.features * self.mlp_ratio))
        x = nn.Dense(hidden, dtype=dtype)(x)
        x = nn.gelu(x)
        x = nn.Conv(hidden, (3, 3), padding="SAME", feature_group_count=hidden, dtype=dtype)(x)
        x = nn.gelu(x)
        return nn.Dense(self.features, dtype=dtype)(x)


class HybridAttentionBlock(nn.Module):
    config: ModelConfig
    shift_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        batch, height, width, _ = x.shape

        shortcut = x
        y = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x)
        if self.shift_size:
            y = jnp.roll(y, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        windows = _window_partition(y, self.config.window_size)
        windows = WindowAttention(
            self.config.features,
            self.config.heads,
            self.config.window_size,
            self.config.dtype,
        )(windows)
        y = _window_reverse(
            windows,
            batch=batch,
            height=height,
            width=width,
            window_size=self.config.window_size,
        )
        if self.shift_size:
            y = jnp.roll(y, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        channel = ChannelAttention(
            self.config.features,
            self.config.channel_reduction,
            self.config.dtype,
        )(nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x))
        x = shortcut + (y + channel) * self.config.residual_scale

        shortcut = x
        y = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x)
        y = ConvFFN(self.config.features, self.config.mlp_ratio, self.config.dtype)(y)
        return shortcut + y * self.config.residual_scale


class AdaptiveTokenDictionary(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.config.token_count <= 0:
            return x

        dtype = _dtype(self.config.dtype)
        batch, height, width, channels = x.shape
        head_dim = channels // self.config.heads
        if channels % self.config.heads:
            raise ValueError(
                f"features must be divisible by heads, got {channels=} {self.config.heads=}"
            )

        shortcut = x
        y = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x)
        y = y.reshape(batch, height * width, channels)
        dictionary = self.param(
            "dictionary",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.config.token_count, channels),
            jnp.float32,
        ).astype(dtype)
        dictionary = jnp.broadcast_to(dictionary[None, :, :], (batch, self.config.token_count, channels))

        q = nn.Dense(channels, dtype=dtype)(y)
        kv = nn.Dense(channels * 2, dtype=dtype)(dictionary)
        q = q.reshape(batch, height * width, self.config.heads, head_dim)
        kv = kv.reshape(batch, self.config.token_count, 2, self.config.heads, head_dim)
        k = kv[:, :, 0]
        v = kv[:, :, 1]

        logits = jnp.einsum("bnhd,bkhd->bhnk", q.astype(jnp.float32), k.astype(jnp.float32))
        logits = logits * (head_dim**-0.5)
        attention = nn.softmax(logits, axis=-1).astype(dtype)
        y = jnp.einsum("bhnk,bkhd->bnhd", attention, v)
        y = y.reshape(batch, height * width, channels)
        y = nn.Dense(channels, dtype=dtype)(y)
        y = y.reshape(batch, height, width, channels)
        return shortcut + y * self.config.residual_scale


class ResidualHybridAttentionGroup(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        shortcut = x
        for index in range(self.config.blocks):
            shift = 0 if index % 2 == 0 else self.config.window_size // 2
            x = HybridAttentionBlock(self.config, shift)(x)
        x = AdaptiveTokenDictionary(self.config)(x)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        return shortcut + x


class TokenDictionaryCrossAttentionV2(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.config.token_count <= 0:
            return x, jnp.zeros_like(x)

        dtype = _dtype(self.config.dtype)
        batch, height, width, channels = x.shape
        head_dim = channels // self.config.heads
        if channels % self.config.heads:
            raise ValueError(
                f"features must be divisible by heads, got {channels=} {self.config.heads=}"
            )

        shortcut = x
        tokens = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x)
        tokens = tokens.reshape(batch, height * width, channels)
        dictionary = self.param(
            "dictionary",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.config.token_count, channels),
            jnp.float32,
        ).astype(dtype)
        dictionary = jnp.broadcast_to(
            dictionary[None, :, :],
            (batch, self.config.token_count, channels),
        )

        token_kv = nn.Dense(channels * 2, dtype=dtype, name="token_kv")(tokens)
        token_kv = token_kv.reshape(batch, height * width, 2, self.config.heads, head_dim)
        token_k = token_kv[:, :, 0]
        token_v = token_kv[:, :, 1]
        dict_q = nn.Dense(channels, dtype=dtype, name="dict_q")(dictionary)
        dict_q = dict_q.reshape(batch, self.config.token_count, self.config.heads, head_dim)
        refine_logits = jnp.einsum(
            "bkhd,bnhd->bhkn",
            dict_q.astype(jnp.float32),
            token_k.astype(jnp.float32),
        )
        refine_logits = refine_logits * (head_dim**-0.5)
        refine_attention = nn.softmax(refine_logits, axis=-1).astype(dtype)
        refined = jnp.einsum("bhkn,bnhd->bkhd", refine_attention, token_v)
        refined = refined.reshape(batch, self.config.token_count, channels)
        refined = dictionary + nn.Dense(channels, dtype=dtype, name="dict_refine")(refined) * (
            self.config.residual_scale
        )

        token_q = nn.Dense(channels, dtype=dtype, name="token_q")(tokens)
        dict_kv = nn.Dense(channels * 2, dtype=dtype, name="dict_kv")(refined)
        token_q = token_q.reshape(batch, height * width, self.config.heads, head_dim)
        dict_kv = dict_kv.reshape(batch, self.config.token_count, 2, self.config.heads, head_dim)
        dict_k = dict_kv[:, :, 0]
        dict_v = dict_kv[:, :, 1]
        logits = jnp.einsum(
            "bnhd,bkhd->bhnk",
            token_q.astype(jnp.float32),
            dict_k.astype(jnp.float32),
        )
        logits = logits * (head_dim**-0.5)
        attention = nn.softmax(logits, axis=-1).astype(dtype)
        global_tokens = jnp.einsum("bhnk,bkhd->bnhd", attention, dict_v)
        global_tokens = global_tokens.reshape(batch, height * width, channels)
        global_tokens = nn.Dense(channels, dtype=dtype, name="token_out")(global_tokens)

        category_embedding = self.param(
            "category_embedding",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.config.token_count, channels),
            jnp.float32,
        ).astype(dtype)
        category_probs = jnp.mean(attention.astype(jnp.float32), axis=1).astype(dtype)
        category_context = jnp.einsum("bnk,kc->bnc", category_probs, category_embedding)
        category_context = nn.Dense(channels, dtype=dtype, name="category_out")(category_context)

        update = global_tokens + category_context * jnp.asarray(0.5, dtype=dtype)
        update = update.reshape(batch, height, width, channels)
        category_context = category_context.reshape(batch, height, width, channels)
        return shortcut + update * self.config.residual_scale, category_context


class CategoryAwareFFN(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, category_context: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        shortcut = x
        y = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x)
        if category_context.shape == x.shape:
            category = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(category_context)
            y = y + nn.Dense(self.config.features, dtype=dtype, name="category_mix")(category)
        y = ConvFFN(self.config.features, self.config.mlp_ratio, self.config.dtype)(y)
        return shortcut + y * self.config.residual_scale


class ATDv2Block(nn.Module):
    config: ModelConfig
    shift_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        batch, height, width, _ = x.shape

        shortcut = x
        y = nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x)
        if self.shift_size:
            y = jnp.roll(y, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        windows = _window_partition(y, self.config.window_size)
        windows = WindowAttention(
            self.config.features,
            self.config.heads,
            self.config.window_size,
            self.config.dtype,
        )(windows)
        y = _window_reverse(
            windows,
            batch=batch,
            height=height,
            width=width,
            window_size=self.config.window_size,
        )
        if self.shift_size:
            y = jnp.roll(y, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        x = shortcut + y * self.config.residual_scale

        x, category_context = TokenDictionaryCrossAttentionV2(self.config)(x)
        channel = ChannelAttention(
            self.config.features,
            self.config.channel_reduction,
            self.config.dtype,
        )(nn.LayerNorm(epsilon=1e-6, dtype=dtype)(x))
        x = x + channel * self.config.residual_scale
        return CategoryAwareFFN(self.config)(x, category_context)


class ATDv2RestorationGroup(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        shortcut = x
        for index in range(self.config.blocks):
            shift = 0 if index % 2 == 0 else self.config.window_size // 2
            x = ATDv2Block(self.config, shift)(x)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        return shortcut + x * self.config.residual_scale


class Upsampler(nn.Module):
    features: int
    scale: int
    dtype_name: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.dtype_name)
        if self.scale in (2, 4, 8):
            stages = {2: 1, 4: 2, 8: 3}[self.scale]
            for _ in range(stages):
                x = nn.Conv(self.features * 4, (3, 3), padding="SAME", dtype=dtype)(x)
                x = pixel_shuffle(x, 2)
                x = nn.gelu(x)
            return x
        if self.scale == 3:
            x = nn.Conv(self.features * 9, (3, 3), padding="SAME", dtype=dtype)(x)
            x = pixel_shuffle(x, 3)
            return nn.gelu(x)
        raise ValueError(f"Unsupported scale: {self.scale}")


class HATATDSR(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        if x.shape[1] % self.config.window_size or x.shape[2] % self.config.window_size:
            raise ValueError(
                "HATATDSR expects LR height/width divisible by window_size. "
                f"got input_hw={x.shape[1:3]} window_size={self.config.window_size}"
            )

        x_in = x.astype(jnp.float32)
        x = x.astype(dtype)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        shallow = x
        for _ in range(self.config.groups):
            x = ResidualHybridAttentionGroup(self.config)(x)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        x = x + shallow
        x = Upsampler(self.config.features, self.config.scale, self.config.dtype)(x)
        residual = nn.Conv(3, (3, 3), padding="SAME", dtype=dtype)(x).astype(jnp.float32)
        if not self.config.global_residual:
            return residual
        base = jax.image.resize(
            x_in,
            (
                x_in.shape[0],
                x_in.shape[1] * self.config.scale,
                x_in.shape[2] * self.config.scale,
                x_in.shape[3],
            ),
            method="cubic",
        )
        return base + residual


class ATDv2SR(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        if x.shape[1] % self.config.window_size or x.shape[2] % self.config.window_size:
            raise ValueError(
                "ATDv2SR expects LR height/width divisible by window_size. "
                f"got input_hw={x.shape[1:3]} window_size={self.config.window_size}"
            )

        x_in = x.astype(jnp.float32)
        x = x.astype(dtype)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        shallow = x
        for _ in range(self.config.groups):
            x = ATDv2RestorationGroup(self.config)(x)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        x = x + shallow
        x = Upsampler(self.config.features, self.config.scale, self.config.dtype)(x)
        output = nn.Conv(3, (3, 3), padding="SAME", dtype=dtype)(x).astype(jnp.float32)
        if not self.config.global_residual:
            return nn.sigmoid(output)
        base = jax.image.resize(
            x_in,
            (
                x_in.shape[0],
                x_in.shape[1] * self.config.scale,
                x_in.shape[2] * self.config.scale,
                x_in.shape[3],
            ),
            method="cubic",
        )
        return base + output


def create_model(config: ModelConfig) -> nn.Module:
    if config.model == "edsr_lite":
        return EDSRLite(config)
    if config.model == "hat_atd":
        return HATATDSR(config)
    if config.model == "atd_v2":
        return ATDv2SR(config)
    raise ValueError(f"Unsupported model: {config.model}")
