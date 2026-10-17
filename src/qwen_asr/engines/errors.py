"""引擎可用性与容量错误。"""


class EngineUnavailable(RuntimeError):
    """A worker or its IPC connection failed."""


class EngineCapacityExceeded(RuntimeError):
    """All bounded engine slots are occupied."""
