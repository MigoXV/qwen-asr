__all__ = ["ASRServicer"]


def __getattr__(name):
    if name == "ASRServicer":
        from .servicer import ASRServicer

        return ASRServicer
    raise AttributeError(name)
