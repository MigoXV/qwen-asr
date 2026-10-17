__all__ = ["VLLMInferencer"]


def __getattr__(name):
    # Importing the package in the API process must not initialize vLLM.
    if name == "VLLMInferencer":
        from qwen_asr.inferencers.vllm.inferencer import VLLMInferencer
        return VLLMInferencer
    raise AttributeError(name)
