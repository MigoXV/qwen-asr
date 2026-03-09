from __future__ import annotations

import sys
import types


def install_fake_grpc() -> None:
    if "grpc" in sys.modules:
        return

    grpc = types.ModuleType("grpc")
    grpc.__version__ = "1.78.0"
    grpc.StatusCode = types.SimpleNamespace(
        INVALID_ARGUMENT="INVALID_ARGUMENT",
        INTERNAL="INTERNAL",
        UNIMPLEMENTED="UNIMPLEMENTED",
    )
    grpc.ServicerContext = object
    grpc.Channel = object
    grpc.Server = object
    grpc.experimental = types.SimpleNamespace(
        stream_stream=lambda *args, **kwargs: None
    )
    grpc.stream_stream_rpc_method_handler = lambda *args, **kwargs: None
    grpc.method_handlers_generic_handler = lambda *args, **kwargs: None
    grpc.aio = types.SimpleNamespace(
        ServicerContext=object,
        Channel=object,
        Server=object,
        server=lambda *args, **kwargs: None,
        insecure_channel=lambda *args, **kwargs: None,
    )

    utilities = types.ModuleType("grpc._utilities")
    utilities.first_version_is_lower = lambda installed, generated: False

    grpc._utilities = utilities

    sys.modules["grpc"] = grpc
    sys.modules["grpc.aio"] = grpc.aio
    sys.modules["grpc._utilities"] = utilities
