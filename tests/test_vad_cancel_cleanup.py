"""在 VAD 容器内验证取消清理；普通 ASR 测试不导入 VAD 依赖。"""

import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


@unittest.skipUnless(os.environ.get("QWEN_ASR_TEST_VAD_PATCH") == "1", "requires patched VAD image")
class VadCancelCleanupTest(unittest.TestCase):
    def make_client(self):
        from fireredvad_ncnn.engine.client import EngineClient
        client = EngineClient.__new__(EngineClient)
        client._condition = threading.Condition()
        client._sessions = {("session", "epoch"): 7}
        client._session_names = {"session": "epoch"}
        request = SimpleNamespace(fields={"session_id": "session", "epoch": "epoch"},
                                  native_id=11, cancel_sent=False)
        client._pending = {11: request}
        return client

    def test_native_cancel_already_retired_session_does_not_kill_bridge(self):
        from fireredvad_ncnn.native import NativeError
        client = self.make_client()
        engine = Mock()
        engine.session_close.side_effect = NativeError(4, "unknown stream session")
        client._drop_session(engine, ("session", "epoch"))
        self.assertEqual(client._sessions, {})
        self.assertEqual(client._session_names, {})
        self.assertTrue(client._pending[11].cancel_sent)
        engine.cancel.assert_called_once_with(11)
        client._drop_session(engine, ("session", "epoch"))
        engine.session_close.assert_called_once_with(7)

    def test_other_native_errors_remain_fatal(self):
        from fireredvad_ncnn.native import NativeError
        for status, message in ((4, "engine closed"), (5, "unknown stream session")):
            client = self.make_client()
            engine = Mock()
            engine.session_close.side_effect = NativeError(status, message)
            with self.assertRaises(NativeError):
                client._drop_session(engine, ("session", "epoch"))

    def test_live_session_still_closes_once(self):
        client = self.make_client()
        engine = Mock()
        client._drop_session(engine, ("session", "epoch"))
        client._drop_session(engine, ("session", "epoch"))
        engine.session_close.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
