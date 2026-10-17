from __future__ import annotations

import unittest

from qwen_asr.engines.vllm.engine import VLLMEngine

from tests.fakes import FakeLLM, FakeSamplingParams


class VLLMEngineLifecycleTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def make_inferencer(llm: FakeLLM) -> VLLMEngine:
        return VLLMEngine(
            model=llm,
            processor=object(),
            sampling_params=FakeSamplingParams(),
        )

    async def test_complete_request_is_not_aborted(self):
        llm = FakeLLM()
        inferencer = self.make_inferencer(llm)

        chunks = [
            chunk
            async for chunk in inferencer.generate_stream(
                {"script": [("你", False), ("你好", True)]}
            )
        ]

        self.assertEqual(chunks, ["你", "好"])
        self.assertEqual(llm.llm_engine.aborted_ids, [])
        self.assertEqual(llm.active_request_ids, set())

    async def test_closing_stream_aborts_request_immediately(self):
        llm = FakeLLM(step_delay=0.01)
        inferencer = self.make_inferencer(llm)
        stream = inferencer.generate_stream(
            {"script": [("你", False), ("你好", False), ("你好啊", True)]}
        )

        self.assertEqual(await anext(stream), "你")
        await stream.aclose()

        self.assertEqual(len(llm.llm_engine.aborted_ids), 1)
        self.assertEqual(llm.active_request_ids, set())


if __name__ == "__main__":
    unittest.main()
