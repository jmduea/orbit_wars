"""Helpers for optional gradient checkpointing in Flax encoders."""

from __future__ import annotations

import flax.linen as nn


def remat_if(module_cls: type[nn.Module], enabled: bool) -> type[nn.Module]:
    """Return ``module_cls`` wrapped with ``nn.remat`` when checkpointing is enabled."""

    if not enabled:
        return module_cls
    return nn.remat(module_cls, prevent_cse=False)
