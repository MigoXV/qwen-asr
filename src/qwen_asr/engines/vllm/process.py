"""Single model owner with bounded, streamed msgpack over a Unix socket.

One worker owns one vLLM Engine Core. API concurrency and max_num_seqs never
create additional models. A disconnected IPC stream cancels that request.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import multiprocessing as mp
import os
import signal
import struct
import tempfile
from collections.abc import AsyncIterator
from contextlib import aclosing, suppress
from pathlib import Path

import msgpack
import numpy as np

from qwen_asr.engines.errors import EngineCapacityExceeded, EngineUnavailable

logger = logging.getLogger(__name__)
MAX_FRAME = 8 * 1024 * 1024
PROTOCOL_VERSION = 1


async def _receive(reader: asyncio.StreamReader) -> dict:
    size = struct.unpack("!I", await reader.readexactly(4))[0]
    if not 0 < size <= MAX_FRAME:
        raise ValueError("Invalid engine frame size")
    message = msgpack.unpackb(await reader.readexactly(size), raw=False)
    if not isinstance(message, dict) or message.get("version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported engine protocol")
    return message


async def _send(writer: asyncio.StreamWriter, **message) -> None:
    payload = msgpack.packb({"version": PROTOCOL_VERSION, **message}, use_bin_type=True)
    if len(payload) > MAX_FRAME:
        raise ValueError("Engine frame exceeds capacity")
    writer.write(struct.pack("!I", len(payload)) + payload)
    await writer.drain()


async def _serve_worker(path: str, config: dict, ready, parent_pid: int) -> None:
    from qwen_asr.inferencers.vllm.inferencer import VLLMInferencer

    engine = VLLMInferencer.LLM(**config)
    tasks: set[asyncio.Task] = set()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        disconnected = None
        try:
            request = await asyncio.wait_for(_receive(reader), timeout=30)
            if request.get("op") != "transcribe":
                raise ValueError("Unknown engine operation")
            audio = np.frombuffer(request["audio"], dtype="<f4")
            sample_rate = request["sample_rate"]
            if not isinstance(sample_rate, int) or sample_rate <= 0 or not 0 < len(audio) <= sample_rate * 30:
                raise ValueError("Audio must be nonempty and at most 30 seconds")

            async def watch_disconnect():
                await reader.read(1)
                task.cancel()

            disconnected = asyncio.create_task(watch_disconnect())
            stream = engine.transcribe_stream(
                audio=audio, sample_rate=sample_rate,
                context=request["context"], language=request["language"],
            )
            sequence = 0
            async with aclosing(stream):
                async for text in stream:
                    await _send(writer, op="delta", sequence=sequence, text=text)
                    sequence += 1
            await _send(writer, op="done", sequence=sequence)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        except Exception as exc:
            logger.exception("Engine request failed")
            with suppress(ConnectionError):
                await _send(writer, op="error", message=str(exc))
            if engine.errored:
                stop.set()
        finally:
            if disconnected:
                disconnected.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnected
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            tasks.discard(task)

    async def supervise_engine():
        while not stop.is_set():
            await asyncio.sleep(1)
            if engine.errored:
                logger.error("vLLM Engine Core failed; stopping worker")
                stop.set()

    supervisor = asyncio.create_task(supervise_engine())
    try:
        server = await asyncio.start_unix_server(handle, path=path)
        ready.send_bytes(json.dumps({"ready": True, "pid": os.getpid()}).encode())
        ready.close()
        async with server:
            await stop.wait()
    finally:
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.aclose()


def _worker_main(path: str, config: dict, ready, parent_pid: int) -> None:
    # Linux parent death signal also protects launches outside systemd.
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent_pid:
        return
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(_serve_worker(path, config, ready, parent_pid))
    except BaseException as exc:
        with suppress(Exception):
            ready.send_bytes(json.dumps({"ready": False, "error": str(exc)}).encode())
        raise


class VLLMEngineClient:
    """A single engine; max_inflight bounds all pending IPC streams."""

    def __init__(self, *, max_inflight: int, engine_config: dict, request_timeout: float = 180):
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self.max_inflight = max_inflight
        self.engine_config = engine_config
        self.request_timeout = request_timeout
        self._directory = tempfile.TemporaryDirectory(prefix="qwen-asr-engine-")
        self._processes = []
        self._paths = []
        self._loads = [0]
        self._closed = False
        self._writers = set()

    async def start(self, timeout: float = 600) -> None:
        if self._closed or self._processes:
            raise EngineUnavailable("The single engine is already started or closed")
        ctx = mp.get_context("spawn")
        receivers = []
        waiters = []

        async def wait_ready(index, receiver, process):
            deadline = asyncio.get_running_loop().time() + timeout
            while not receiver.poll():
                if not process.is_alive():
                    raise EngineUnavailable(f"Worker {index} exited before readiness")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"Worker {index} startup timed out")
                await asyncio.sleep(0.1)
            status = json.loads(receiver.recv_bytes())
            if not status.get("ready"):
                raise EngineUnavailable(f"Worker {index}: {status}")
            logger.info("Worker %d ready, pid=%d", index, process.pid)

        try:
            for index in range(1):
                path = str(Path(self._directory.name) / f"worker-{index}.sock")
                receiver, sender = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=_worker_main,
                    args=(path, self.engine_config, sender, os.getpid()),
                    name=f"qwen-asr-worker-{index}",
                )
                process.start()
                sender.close()
                self._processes.append(process)
                self._paths.append(path)
                receivers.append(receiver)
                waiters.append(asyncio.create_task(wait_ready(index, receiver, process)))
            await asyncio.gather(*waiters)
        except BaseException:
            await self.aclose()
            raise
        finally:
            for task in waiters:
                task.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            for receiver in receivers:
                receiver.close()

    async def wait_failure(self):
        while not self._closed:
            await asyncio.sleep(1)
            for process in self._processes:
                if not process.is_alive():
                    raise EngineUnavailable(f"{process.name} exited: {process.exitcode}")

    async def transcribe_stream(self, audio, sample_rate, context="", language=None) -> AsyncIterator[str]:
        if sample_rate <= 0 or not 0 < len(audio) <= sample_rate * 30:
            raise ValueError("Audio must be nonempty and at most 30 seconds")
        if self._closed or any(not p.is_alive() for p in self._processes):
            raise EngineUnavailable("Engine process is unavailable")
        if sum(self._loads) >= self.max_inflight:
            raise EngineCapacityExceeded("Engine inflight limit reached")
        index = 0
        self._loads[index] += 1
        deadline = asyncio.get_running_loop().time() + self.request_timeout
        started = asyncio.get_running_loop().time()
        completed = False
        writer = None
        try:
            reader, writer = await asyncio.open_unix_connection(self._paths[index])
            self._writers.add(writer)
            await _send(
                writer, op="transcribe", audio=np.asarray(audio, dtype="<f4").tobytes(),
                sample_rate=sample_rate, context=context, language=language,
            )
            sequence = 0
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Engine request deadline exceeded")
                try:
                    message = await asyncio.wait_for(_receive(reader), remaining)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError("Engine request deadline exceeded") from exc
                if message["op"] == "error":
                    raise EngineUnavailable(message["message"])
                if message.get("sequence") != sequence:
                    raise EngineUnavailable("Engine stream sequence mismatch")
                if message["op"] == "done":
                    completed = True
                    return
                if message["op"] != "delta":
                    raise EngineUnavailable("Unknown engine response")
                sequence += 1
                yield message["text"]
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            raise EngineUnavailable("Engine connection lost") from exc
        finally:
            self._loads[index] -= 1
            logger.info(
                "Engine request worker=%d elapsed_s=%.3f completed=%s inflight=%d loads=%s",
                index, asyncio.get_running_loop().time() - started, completed,
                sum(self._loads), self._loads,
            )
            if writer:
                self._writers.discard(writer)
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in list(self._writers):
            writer.close()
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            await asyncio.to_thread(process.join, 15)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
            process.close()
        self._directory.cleanup()
