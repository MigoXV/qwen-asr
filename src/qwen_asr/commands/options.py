"""命令行选项类型。"""

from enum import Enum


class Device(str, Enum):
    """服务支持的推理设备。"""

    CPU = "cpu"
    CUDA = "cuda"


class AscendGraphMode(str, Enum):
    FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
    PIECEWISE = "PIECEWISE"
