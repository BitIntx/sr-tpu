from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class RefinerConfig:
    features: int = 64
    blocks: int = 10
    residual_scale: float = 0.25
    dtype: str = "bfloat16"


def _dtype(name: str):
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}")


class RefinerBlock(nn.Module):
    config: RefinerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        residual = x
        y = nn.LayerNorm(dtype=jnp.float32)(x).astype(dtype)
        y = nn.Conv(self.config.features * 2, (1, 1), dtype=dtype)(y)
        gate, value = jnp.split(y, 2, axis=-1)
        y = nn.gelu(gate) * value
        y = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(y)
        y = nn.gelu(y)
        y = nn.Conv(self.config.features, (1, 1), dtype=dtype)(y)
        return residual + y.astype(residual.dtype) * 0.1


class ResidualRefiner(nn.Module):
    config: RefinerConfig

    @nn.compact
    def __call__(self, features: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
        dtype = _dtype(self.config.dtype)
        x = features.astype(dtype)
        x = nn.Conv(self.config.features, (3, 3), padding="SAME", dtype=dtype)(x)
        for _ in range(self.config.blocks):
            x = RefinerBlock(self.config)(x)
        residual = nn.Conv(
            3,
            (3, 3),
            padding="SAME",
            dtype=dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        residual = residual.astype(jnp.float32)
        return jnp.clip(base.astype(jnp.float32) + residual * self.config.residual_scale, 0.0, 1.0)


def create_refiner(config: RefinerConfig) -> ResidualRefiner:
    return ResidualRefiner(config)
