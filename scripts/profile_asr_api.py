"""按信号采集 API 主线程 CPU profile，关闭采样后可在同实例正常压测。

ASR_API_PROFILE_PATH=/home/.../api.pstats poetry run python scripts/profile_asr_api.py vllm-ascend ...
kill -USR1 <API PID> 开始，kill -USR2 <API PID> 停止并保存。
kill -WINCH <API PID> 在差分验证过的原版/优化重复清理间切换，仅供同实例消融。
"""

import cProfile
import os
from pathlib import Path
import runpy
import signal


def main():
    destination = Path(os.environ['ASR_API_PROFILE_PATH'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = None
    from qwen_asr.inferencers.text import asr_output
    optimized_cleanup = asr_output.detect_and_fix_repetitions
    reference_cleanup = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/repetition_reference.py'))['detect_and_fix_repetitions']
    reference_enabled = False

    def toggle_cleanup(_signal, _frame):
        nonlocal reference_enabled
        reference_enabled = not reference_enabled
        asr_output.detect_and_fix_repetitions = reference_cleanup if reference_enabled else optimized_cleanup
        print(f'ASR_API_REFERENCE_CLEANUP={reference_enabled}',flush=True)

    def start(_signal, _frame):
        nonlocal profile
        if profile is None:
            profile = cProfile.Profile()
            profile.enable()

    def stop(_signal, _frame):
        nonlocal profile
        if profile is not None:
            profile.disable()
            profile.dump_stats(str(destination))
            profile = None

    signal.signal(signal.SIGUSR1, start)
    signal.signal(signal.SIGUSR2, stop)
    signal.signal(signal.SIGWINCH, toggle_cleanup)
    try:
        runpy.run_module('qwen_asr.commands.app', run_name='__main__')
    finally:
        stop(None, None)


if __name__ == '__main__':
    main()
