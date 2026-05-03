from __future__ import annotations

import asyncio
import unittest

from tests.grpc_stub import install_fake_grpc

install_fake_grpc()

from qwen_asr.protos.asr.ux_speech_pb2 import (
    RecognitionConfig,
    StreamingRecognitionConfig,
    StreamingRecognizeRequest,
)
from qwen_asr.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr.servicer.servicer import ASRServicer

from tests.fakes import FakeAbortError, FakeContext, FakeLLM, FakeModel


class AsyncRequestIterator:
    def __init__(self, requests):
        self._iter = iter(requests)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def make_requests(
    language: str,
    interim_results: bool = True,
    audio_bytes: bytes | None = None,
    hotwords: list[str] | None = None,
    extra_requests: list[StreamingRecognizeRequest] | None = None,
):
    requests = [
        StreamingRecognizeRequest(
            streaming_config=StreamingRecognitionConfig(
                config=RecognitionConfig(
                    encoding=RecognitionConfig.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=language,
                    hotwords=hotwords or [],
                ),
                interim_results=interim_results,
            )
        )
    ]
    if audio_bytes is not None:
        requests.append(StreamingRecognizeRequest(audio_content=audio_bytes))
    if extra_requests:
        requests.extend(extra_requests)
    return AsyncRequestIterator(requests)


async def collect_responses(responses):
    collected = []
    async for response in responses:
        collected.append(response)
    return collected


def collect_transcripts(responses):
    return [response.results[0].alternative.transcript for response in responses]


def make_servicer(model):
    return make_servicer_with_context(model, "")


def make_servicer_with_context(model, system_context: str):
    original_load_inferencer = ASRServicer._load_inferencer
    ASRServicer._load_inferencer = staticmethod(
        lambda config: GrpcInferencer(inferencer=model)
    )
    try:
        config = type("FakeConfig", (), {"context": system_context})()
        return ASRServicer(config=config)
    finally:
        ASRServicer._load_inferencer = original_load_inferencer


class ASRServicerStreamingTest(unittest.IsolatedAsyncioTestCase):
    async def test_interim_and_final_results_follow_stream_events(self):
        scripts = {
            "|Chinese": [("你", False), ("你好", True)],
        }
        servicer = make_servicer(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "Chinese",
                        audio_bytes=(100).to_bytes(2, "little", signed=True),
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["你", "你好", "你好"])
            self.assertEqual(
                [response.results[0].is_final for response in responses],
                [False, False, True],
            )
        finally:
            servicer.close()

    async def test_non_interim_mode_returns_only_final_result(self):
        scripts = {
            "|English": [("hel", False), ("hello", True)],
        }
        servicer = make_servicer(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "English",
                        interim_results=False,
                        audio_bytes=(200).to_bytes(2, "little", signed=True),
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["hello"])
            self.assertTrue(responses[0].results[0].is_final)
        finally:
            servicer.close()

    async def test_forced_language_prompt_echo_is_not_exposed_in_transcript(self):
        scripts = {
            "|Chinese": [
                ("language Chinese<asr_text>你", False),
                ("language Chinese<asr_text>你好", True),
            ],
        }
        servicer = make_servicer(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "Chinese",
                        audio_bytes=(200).to_bytes(2, "little", signed=True),
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["你", "你好", "你好"])
            self.assertEqual(
                [response.results[0].alternative.words[0].word for response in responses],
                ["你", "好", ""],
            )
        finally:
            servicer.close()

    async def test_incomplete_language_prefix_is_not_exposed_in_interim_results(self):
        scripts = {
            "|auto": [
                ("lang", False),
                ("langute", False),
                ("language None", False),
                ("language None<asr_text>la", False),
                ("language None<asr_text>la la", True),
            ],
        }
        servicer = make_servicer(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "",
                        audio_bytes=(200).to_bytes(2, "little", signed=True),
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["la", "la la", "la la"])
            self.assertEqual(
                [response.results[0].is_final for response in responses],
                [False, False, True],
            )
        finally:
            servicer.close()

    async def test_client_cancellation_aborts_only_current_request(self):
        scripts = {
            "|Chinese": [("你", False), ("你好", False), ("你好啊", True)],
        }
        llm = FakeLLM(
            script_factory=lambda inp: scripts[inp["prompt"]],
            step_delay=0.02,
        )
        servicer = make_servicer(FakeModel(llm))
        context = FakeContext()
        try:
            stream = servicer.StreamingRecognize(
                make_requests(
                    "Chinese",
                    audio_bytes=(100).to_bytes(2, "little", signed=True),
                ),
                context,
            )
            first = await anext(stream)
            self.assertEqual(first.results[0].alternative.transcript, "你")

            context.set_active(False)
            remaining = await collect_responses(stream)
            self.assertEqual(remaining, [])

            deadline = asyncio.get_running_loop().time() + 1.0
            while asyncio.get_running_loop().time() < deadline:
                if llm.llm_engine.aborted_ids:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(llm.llm_engine.aborted_ids), 1)
        finally:
            servicer.close()

    async def test_concurrent_sessions_do_not_cross_stream_outputs(self):
        scripts = {
            "|Chinese": [("你", False), ("你好", True)],
            "|English": [("he", False), ("hello", True)],
        }
        servicer = make_servicer(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        )
        try:
            async def run_call(language: str):
                responses = await collect_responses(
                    servicer.StreamingRecognize(
                        make_requests(
                            language,
                            audio_bytes=(300).to_bytes(2, "little", signed=True),
                        ),
                        FakeContext(),
                    )
                )
                return collect_transcripts(responses)

            transcripts_zh, transcripts_en = await asyncio.gather(
                run_call("Chinese"),
                run_call("English"),
            )

            self.assertEqual(transcripts_zh, ["你", "你好", "你好"])
            self.assertEqual(transcripts_en, ["he", "hello", "hello"])
        finally:
            servicer.close()

    async def test_hotwords_are_joined_as_session_context(self):
        scripts = {
            "foo bar|Chinese": [("命中", True)],
        }
        servicer = make_servicer(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "Chinese",
                        audio_bytes=(100).to_bytes(2, "little", signed=True),
                        hotwords=["foo", "bar"],
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["命中", "命中"])
        finally:
            servicer.close()

    async def test_system_context_is_used_when_hotwords_are_empty(self):
        scripts = {
            "system prompt|Chinese": [("系统", True)],
        }
        servicer = make_servicer_with_context(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]])),
            "system prompt",
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "Chinese",
                        audio_bytes=(100).to_bytes(2, "little", signed=True),
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["系统", "系统"])
        finally:
            servicer.close()

    async def test_hotwords_override_system_context(self):
        scripts = {
            "session term|Chinese": [("会话", True)],
        }
        servicer = make_servicer_with_context(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]])),
            "system prompt",
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "Chinese",
                        audio_bytes=(100).to_bytes(2, "little", signed=True),
                        hotwords=["session", "term"],
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["会话", "会话"])
        finally:
            servicer.close()

    async def test_blank_hotwords_are_ignored(self):
        scripts = {
            "foo bar|Chinese": [("过滤", True)],
        }
        servicer = make_servicer_with_context(
            FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]])),
            "system prompt",
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests(
                        "Chinese",
                        audio_bytes=(100).to_bytes(2, "little", signed=True),
                        hotwords=["", " foo ", "  ", "bar"],
                    ),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["过滤", "过滤"])
        finally:
            servicer.close()

    async def test_first_message_must_be_streaming_config(self):
        servicer = make_servicer(FakeModel(FakeLLM()))
        try:
            with self.assertRaises(FakeAbortError) as exc_info:
                await collect_responses(
                    servicer.StreamingRecognize(
                        AsyncRequestIterator(
                            [
                                StreamingRecognizeRequest(
                                    audio_content=(100).to_bytes(
                                        2, "little", signed=True
                                    )
                                )
                            ]
                        ),
                        FakeContext(),
                    )
                )
            self.assertEqual(exc_info.exception.code, "INVALID_ARGUMENT")
        finally:
            servicer.close()

    async def test_second_message_must_be_audio_content(self):
        servicer = make_servicer(FakeModel(FakeLLM()))
        try:
            with self.assertRaises(FakeAbortError) as exc_info:
                await collect_responses(
                    servicer.StreamingRecognize(
                        make_requests("Chinese", audio_bytes=None),
                        FakeContext(),
                    )
                )
            self.assertEqual(exc_info.exception.code, "INVALID_ARGUMENT")
        finally:
            servicer.close()

    async def test_multiple_audio_messages_are_rejected(self):
        servicer = make_servicer(FakeModel(FakeLLM()))
        try:
            with self.assertRaises(FakeAbortError) as exc_info:
                await collect_responses(
                    servicer.StreamingRecognize(
                        make_requests(
                            "Chinese",
                            audio_bytes=(100).to_bytes(2, "little", signed=True),
                            extra_requests=[
                                StreamingRecognizeRequest(
                                    audio_content=(200).to_bytes(
                                        2, "little", signed=True
                                    )
                                )
                            ],
                        ),
                        FakeContext(),
                    )
                )
            self.assertEqual(exc_info.exception.code, "INVALID_ARGUMENT")
        finally:
            servicer.close()

    async def test_riff_header_is_stripped_before_inference(self):
        scripts = {
            "|Chinese": [("你", True)],
        }
        model = FakeModel(FakeLLM(script_factory=lambda inp: scripts[inp["prompt"]]))
        servicer = make_servicer(model)
        riff_audio = b"RIFF" + (b"\x00" * 40) + (123).to_bytes(
            2, "little", signed=True
        )
        try:
            responses = await collect_responses(
                servicer.StreamingRecognize(
                    make_requests("Chinese", audio_bytes=riff_audio),
                    FakeContext(),
                )
            )
            self.assertEqual(collect_transcripts(responses), ["你", "你"])
            seen_audio, seen_sample_rate = model.seen_audios[0]
            self.assertEqual(seen_sample_rate, 16000)
            self.assertEqual(seen_audio.shape[0], 1)
            self.assertAlmostEqual(float(seen_audio[0]), 123 / 32768.0, places=6)
        finally:
            servicer.close()


if __name__ == "__main__":
    unittest.main()
