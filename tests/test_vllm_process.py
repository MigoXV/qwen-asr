from __future__ import annotations

import asyncio
import struct
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from qwen_asr.engines.errors import EngineCapacityExceeded, EngineUnavailable
from qwen_asr.engines.vllm.process import VLLMEngineClient, _receive, _send


class VLLMEngineClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = VLLMEngineClient(max_inflight=2, engine_config={})
        self.path = str(Path(self.pool._directory.name) / "test.sock")
        self.process = Mock()
        self.process.is_alive.return_value = True
        self.pool._processes = [self.process]
        self.pool._paths = [self.path]
        self.server = None

    async def asyncTearDown(self):
        self.process.is_alive.return_value = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        await self.pool.aclose()

    async def test_stream_order_and_audio_roundtrip(self):
        async def handle(reader, writer):
            request = await _receive(reader)
            self.assertEqual(request["op"], "transcribe")
            self.assertEqual(request["sample_rate"], 16000)
            np.testing.assert_equal(np.frombuffer(request["audio"], dtype="<f4"), [0.25, -0.25])
            await _send(writer, op="delta", sequence=0, text="你")
            await _send(writer, op="delta", sequence=1, text="好")
            await _send(writer, op="done", sequence=2)
            writer.close()
            await writer.wait_closed()

        self.server = await asyncio.start_unix_server(handle, self.path)
        output = [part async for part in self.pool.transcribe_stream(np.array([0.25, -0.25]), 16000)]
        self.assertEqual(output, ["你", "好"])
        self.assertEqual(self.pool._loads, [0])

    async def test_early_stream_close_disconnects_worker(self):
        disconnected = asyncio.Event()

        async def handle(reader, writer):
            await _receive(reader)
            await _send(writer, op="delta", sequence=0, text="first")
            await reader.read()
            disconnected.set()
            writer.close()
            await writer.wait_closed()

        self.server = await asyncio.start_unix_server(handle, self.path)
        stream = self.pool.transcribe_stream(np.ones(16), 16000)
        self.assertEqual(await anext(stream), "first")
        await stream.aclose()
        await asyncio.wait_for(disconnected.wait(), 1)
        self.assertEqual(self.pool._loads, [0])

    async def test_capacity_and_dead_worker_fail_without_queueing(self):
        self.pool._loads[0] = 2
        with self.assertRaises(EngineCapacityExceeded):
            await anext(self.pool.transcribe_stream(np.ones(16), 16000))
        self.pool._loads[0] = 0
        self.process.is_alive.return_value = False
        with self.assertRaises(EngineUnavailable):
            await anext(self.pool.transcribe_stream(np.ones(16), 16000))

    async def test_worker_disconnect_fails_pending_and_releases_slot(self):
        async def handle(reader, writer):
            await _receive(reader)
            writer.close()
            await writer.wait_closed()

        self.server = await asyncio.start_unix_server(handle, self.path)
        with self.assertRaises(EngineUnavailable):
            await anext(self.pool.transcribe_stream(np.ones(16), 16000))
        self.assertEqual(self.pool._loads, [0])

    async def test_reject_oversized_frame_before_reading_payload(self):
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack("!I", 16 * 1024 * 1024))
        with self.assertRaises(ValueError):
            await _receive(reader)


if __name__ == "__main__":
    unittest.main()
