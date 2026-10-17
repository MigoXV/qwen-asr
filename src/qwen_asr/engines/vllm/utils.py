from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def filter_async_engine_kwargs(
    async_engine_args_cls: type,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    valid_params = inspect.signature(async_engine_args_cls.__init__).parameters
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in valid_params
    }
    dropped_keys = sorted(set(kwargs) - set(filtered_kwargs))
    if dropped_keys:
        logger.warning(
            "Ignoring unsupported AsyncEngineArgs kwargs for this vLLM version: %s",
            ", ".join(dropped_keys),
        )
    return filtered_kwargs
