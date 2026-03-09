from __future__ import annotations

import time
import unittest

from tests.grpc_stub import install_fake_grpc

install_fake_grpc()

from qwen_asr.servicer.stream_dispatcher import VLLMStreamDispatcher

from tests.fakes import FakeLLM, FakeSamplingParams


def collect_events(handle, timeout: float = 1.0):
    deadline = time.time() + timeout
    events = []
    while time.time() < deadline:
        remaining = max(0.01, deadline - time.time())
        event = handle.events.get(timeout=remaining)
        events.append(event)
        if event.kind in {"eos", "cancelled", "error"}:
            return events
    raise TimeoutError("timed out waiting for stream events")


class StreamDispatcherTest(unittest.TestCase):
    def test_single_request_routes_deltas_and_eos(self):
        dispatcher = VLLMStreamDispatcher(FakeLLM())
        try:
            handle = dispatcher.submit(
                {"script": [("he", False), ("hello", True)]},
                FakeSamplingParams(),
            )
            events = collect_events(handle)
            self.assertEqual(
                [(event.kind, event.text) for event in events],
                [("delta", "he"), ("delta", "llo"), ("eos", "")],
            )
        finally:
            dispatcher.shutdown()

    def test_concurrent_requests_are_routed_by_request_id(self):
        dispatcher = VLLMStreamDispatcher(FakeLLM())
        try:
            handle_a = dispatcher.submit(
                {"script": [("a1", False), ("a2", True)]},
                FakeSamplingParams(),
            )
            handle_b = dispatcher.submit(
                {"script": [("b1", False), ("b2", True)]},
                FakeSamplingParams(),
            )

            events_a = collect_events(handle_a)
            events_b = collect_events(handle_b)

            self.assertEqual(
                [(event.kind, event.text) for event in events_a],
                [("delta", "a1"), ("delta", "a2"), ("eos", "")],
            )
            self.assertEqual(
                [(event.kind, event.text) for event in events_b],
                [("delta", "b1"), ("delta", "b2"), ("eos", "")],
            )
        finally:
            dispatcher.shutdown()

    def test_cancel_aborts_only_target_request(self):
        llm = FakeLLM(step_delay=0.02)
        dispatcher = VLLMStreamDispatcher(llm)
        try:
            handle_a = dispatcher.submit(
                {"script": [("a1", False), ("a2", False), ("a3", True)]},
                FakeSamplingParams(),
            )
            handle_b = dispatcher.submit(
                {"script": [("b1", False), ("b2", True)]},
                FakeSamplingParams(),
            )

            first_event = handle_a.events.get(timeout=1.0)
            self.assertEqual((first_event.kind, first_event.text), ("delta", "a1"))

            dispatcher.cancel(handle_a.request_id)

            terminal_a = collect_events(handle_a)
            terminal_b = collect_events(handle_b)

            self.assertEqual(terminal_a[-1].kind, "cancelled")
            self.assertEqual(
                [(event.kind, event.text) for event in terminal_b],
                [("delta", "b1"), ("delta", "b2"), ("eos", "")],
            )
            self.assertEqual(llm.llm_engine.aborted_ids, [handle_a.request_id])
        finally:
            dispatcher.shutdown()

    def test_dispatcher_failure_propagates_and_blocks_new_submissions(self):
        llm = FakeLLM()
        llm.llm_engine.raise_on_step = RuntimeError("boom")
        dispatcher = VLLMStreamDispatcher(llm)
        try:
            handle = dispatcher.submit(
                {"script": [("hello", True)]},
                FakeSamplingParams(),
            )
            events = collect_events(handle)
            self.assertEqual(events[-1].kind, "error")
            self.assertIsInstance(events[-1].error, RuntimeError)

            with self.assertRaises(RuntimeError):
                dispatcher.submit({"script": [("later", True)]}, FakeSamplingParams())
        finally:
            dispatcher.shutdown()

    def test_falls_back_to_engine_request_id_when_explicit_ids_are_unsupported(self):
        llm = FakeLLM(support_explicit_request_id=False)
        dispatcher = VLLMStreamDispatcher(llm)
        try:
            handle = dispatcher.submit(
                {"script": [("x", False), ("xy", True)]},
                FakeSamplingParams(),
            )
            events = collect_events(handle)
            self.assertEqual(
                [(event.kind, event.text) for event in events],
                [("delta", "x"), ("delta", "y"), ("eos", "")],
            )
            self.assertEqual(llm.llm_engine.aborted_ids, [])
        finally:
            dispatcher.shutdown()


if __name__ == "__main__":
    unittest.main()
