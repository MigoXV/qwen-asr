import asyncio
import unittest

import grpc

from qwen_asr.servicer.servicer import ASRServicer
from tests.fakes import FakeAbortError, FakeContext


class ServicerCapacityTest(unittest.IsolatedAsyncioTestCase):
    async def test_capacity_includes_waiting_for_audio_and_releases_on_cancel(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        class WaitingServicer(ASRServicer):
            async def _recognize(self, request_iterator, context):
                entered.set()
                await release.wait()
                yield "ok"

        servicer = WaitingServicer(object(), max_concurrent_requests=1)
        first = servicer.StreamingRecognize(None, FakeContext())
        pending = asyncio.create_task(anext(first))
        await entered.wait()
        second = servicer.StreamingRecognize(None, FakeContext())
        with self.assertRaises(FakeAbortError) as error:
            await anext(second)
        self.assertEqual(error.exception.code, grpc.StatusCode.RESOURCE_EXHAUSTED)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await first.aclose()
        self.assertEqual(servicer.active_requests, 0)
        release.set()
        third = servicer.StreamingRecognize(None, FakeContext())
        self.assertEqual(await anext(third), "ok")
        await third.aclose()
        self.assertEqual(servicer.active_requests, 0)
