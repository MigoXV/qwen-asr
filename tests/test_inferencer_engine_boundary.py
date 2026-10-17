from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from qwen_asr.inferencers.transformers import TransformersInferencer
from qwen_asr.inferencers.vllm import VLLMInferencer


class RecordingEngine:
    def __init__(self):
        self.processor = Mock(return_value={"input_ids": "prepared"})
        self.processor.apply_chat_template.return_value = "prompt:"
        self.inputs = None
        self.closed = False

    async def generate_stream(self, inputs):
        self.inputs = inputs
        try:
            yield "你"
            yield "好"
        finally:
            self.closed = True


class InferencerEngineBoundaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_preserves_audio_language_and_context(self):
        engine = RecordingEngine()
        inferencer = VLLMInferencer(engine)
        audio = np.ones(16, dtype=np.float32)
        output = [part async for part in inferencer.transcribe_stream(
            audio, 16000, context="会议", language="zh",
        )]
        self.assertEqual(output, ["你", "好"])
        self.assertEqual(engine.inputs["prompt"], "prompt:language Chinese<asr_text>")
        payload = engine.inputs["multi_modal_data"]["audio"][0]
        self.assertIs(payload[0], audio)
        self.assertEqual(payload[1], 16000)
        messages = engine.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"], "会议")
        self.assertTrue(engine.closed)

    async def test_transformers_passes_processor_output_to_engine(self):
        engine = RecordingEngine()
        inferencer = TransformersInferencer(engine)
        audio = np.ones(16, dtype=np.float32)
        output = [part async for part in inferencer.transcribe_stream(
            audio, 16000, context="会议", language="zh",
        )]
        self.assertEqual(output, ["你", "好"])
        kwargs = engine.processor.call_args.kwargs
        self.assertEqual(kwargs["text"], ["prompt:language Chinese<asr_text>"])
        self.assertIs(kwargs["audio"][0], audio)
        self.assertEqual(kwargs["sampling_rate"], 16000)
        self.assertIs(engine.inputs, engine.processor.return_value)
        self.assertTrue(engine.closed)

    async def test_closing_adapter_stream_closes_engine_stream(self):
        for adapter in (VLLMInferencer, TransformersInferencer):
            with self.subTest(adapter=adapter.__name__):
                engine = RecordingEngine()
                stream = adapter(engine).transcribe_stream(np.ones(16), 16000)
                self.assertEqual(await anext(stream), "你")
                await stream.aclose()
                self.assertTrue(engine.closed)
