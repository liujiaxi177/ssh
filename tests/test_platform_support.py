from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssh_mcp import platform_support
from ssh_mcp.platform_support import (
    IS_WINDOWS,
    PtyBackendError,
    child_environment,
    reap,
    resolve_backend_name,
    spawn_local_pty,
    terminate_process,
)
from ssh_mcp.ssh import SessionManager, ValidationError


def collect(pty, *, max_wait: float = 10.0) -> bytes:
    """Read from *pty* until end-of-stream, with a wall-clock guard."""
    import time

    chunks: list[bytes] = []
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        data = pty.read(4096)
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks)


class BackendSelectionTests(unittest.TestCase):
    def test_backend_name_is_known(self) -> None:
        self.assertIn(resolve_backend_name(), {"posix", "conpty", "pipe"})

    def test_windows_defaults_to_conpty_when_pywinpty_is_installed(self) -> None:
        if not IS_WINDOWS:
            self.skipTest("Windows-only behaviour")
        expected = "conpty" if platform_support.conpty_available() else "pipe"
        with mock.patch.dict(os.environ, {"SSH_MCP_PTY_BACKEND": "auto"}):
            self.assertEqual(resolve_backend_name(), expected)

    def test_pipe_can_be_forced(self) -> None:
        if not IS_WINDOWS:
            self.skipTest("Windows-only behaviour")
        with mock.patch.dict(os.environ, {"SSH_MCP_PTY_BACKEND": "pipe"}):
            self.assertEqual(resolve_backend_name(), "pipe")

    def test_unknown_backend_is_rejected(self) -> None:
        if not IS_WINDOWS:
            self.skipTest("Windows-only behaviour")
        with mock.patch.dict(os.environ, {"SSH_MCP_PTY_BACKEND": "nonsense"}):
            with self.assertRaises(PtyBackendError):
                resolve_backend_name()

    def test_conpty_request_without_pywinpty_raises(self) -> None:
        if not IS_WINDOWS:
            self.skipTest("Windows-only behaviour")
        with mock.patch.dict(os.environ, {"SSH_MCP_PTY_BACKEND": "conpty"}):
            with mock.patch.object(platform_support, "conpty_available", return_value=False):
                with self.assertRaises(PtyBackendError):
                    resolve_backend_name()


class ChildEnvironmentTests(unittest.TestCase):
    def test_explicit_term_is_preserved(self) -> None:
        env = child_environment({"TERM": "screen-256color"})
        self.assertEqual(env["TERM"], "screen-256color")

    def test_missing_or_dumb_term_is_replaced_on_windows(self) -> None:
        for value in ("", "dumb", "unknown"):
            env = child_environment({"TERM": value})
            if IS_WINDOWS:
                self.assertEqual(env["TERM"], "xterm-256color")
            else:
                self.assertEqual(env["TERM"], value)

    def test_does_not_mutate_caller_mapping(self) -> None:
        original = {"TERM": "dumb"}
        child_environment(original)
        self.assertEqual(original, {"TERM": "dumb"})


class PtySpawnTests(unittest.TestCase):
    """Exercises whichever backend this platform uses."""

    def test_reads_child_output_and_exit_code(self) -> None:
        pty = spawn_local_pty(
            [sys.executable, "-c", "import sys; print('pty-hello'); sys.exit(3)"]
        )
        try:
            output = collect(pty)
        finally:
            pty.close()
        self.assertIn(b"pty-hello", output)
        self.assertEqual(reap(pty.process), 3)

    def test_write_reaches_child_stdin(self) -> None:
        code = "import sys; line = sys.stdin.readline(); print('got:' + line.strip())"
        pty = spawn_local_pty([sys.executable, "-c", code])
        try:
            self.assertEqual(pty.write(b"ping\n"), 5)
            output = collect(pty)
        finally:
            pty.close()
        self.assertIn(b"got:ping", output)

    def test_terminate_process_stops_a_running_child(self) -> None:
        pty = spawn_local_pty([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.assertIsNone(pty.process.poll())
            terminate_process(pty.process, force=False)
            deadline = 10.0
            step = 0.05
            waited = 0.0
            while pty.process.poll() is None and waited < deadline:
                import time as _time

                _time.sleep(step)
                waited += step
            self.assertIsNotNone(pty.process.poll())
        finally:
            terminate_process(pty.process, force=True)
            pty.close()

    def test_close_is_idempotent(self) -> None:
        pty = spawn_local_pty([sys.executable, "-c", "pass"])
        collect(pty)
        pty.close()
        pty.close()
        self.assertTrue(pty.closed)


class TranscriptFidelityTests(unittest.TestCase):
    """Sessions write what the terminal produced -- no newline translation."""

    def _start(self, manager: SessionManager, code: str):
        return manager.start(
            session_name=None,
            target="fake",
            argv=[sys.executable, "-c", code],
            remote_command=None,
            observer_mode="transcript",
        )

    def test_transcript_matches_what_the_session_returned(self) -> None:
        """The transcript must be byte-identical to the session's own output.

        Asserting on a literal CRLF count would be platform-specific: a POSIX pty
        rewrites LF to CRLF on its own (ONLCR), so a child that writes CRLF
        legitimately comes back as CRCRLF there.  Comparing the on-disk bytes
        with the value the tool hands back catches what actually matters -- that
        writing the transcript does not rewrite line endings a second time
        (``newline=None`` on Windows would turn CRLF into CRCRLF).
        """
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(state_dir=Path(tmp))
            try:
                session = self._start(
                    manager,
                    # os.write bypasses Python's own newline translation, which
                    # would otherwise rewrite CRLF depending on whether the
                    # child's stdout happens to be a console or a pipe.
                    "import os; os.write(1, b'alpha\\r\\nbeta\\r\\n')",
                )
                import time

                deadline = 10.0
                waited = 0.0
                while session.process.poll() is None and waited < deadline:
                    time.sleep(0.05)
                    waited += 0.05
                output = ""
                # Drain until the stream settles; the reader thread appends the
                # final chunk right after the child exits.
                for _ in range(5):
                    chunk = session.read(wait_seconds=0.4, max_output_chars=65536)["output"]
                    if not chunk and output:
                        break
                    output += chunk
            finally:
                manager.close()
            raw = (Path(tmp) / session.session_id / "transcript.log").read_bytes()

        self.assertIn("alpha", output)
        self.assertIn("beta", output)
        self.assertEqual(raw.decode("utf-8"), output)
        # Whatever the terminal produced, no line feed was left on its own.
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))

    def test_session_write_and_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(state_dir=Path(tmp))
            try:
                session = self._start(
                    manager,
                    "import sys\nfor line in sys.stdin:\n"
                    "    sys.stdout.write('echo:' + line)\n    sys.stdout.flush()\n",
                )
                session.read(wait_seconds=0.5, max_output_chars=65536)
                result = session.write(
                    input_text="hello session\n", wait_seconds=3.0, max_output_chars=65536
                )
                self.assertEqual(result["bytes_written"], len(b"hello session\n"))
                output = result["output"]
                if "echo:hello session" not in output:
                    output += session.read(
                        wait_seconds=3.0, max_output_chars=65536
                    )["output"]
                self.assertIn("echo:hello session", output)
            finally:
                manager.close()

    def test_stop_terminates_child_and_closes_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(state_dir=Path(tmp))
            try:
                session = self._start(manager, "import time; time.sleep(60)")
                session.read(wait_seconds=0.3, max_output_chars=1024)
                result = session.stop(force=True, wait_seconds=5.0, max_output_chars=1024)
            finally:
                manager.close()
        self.assertTrue(result["was_running"])
        self.assertFalse(result["running"])

    def test_write_to_exited_session_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(state_dir=Path(tmp))
            try:
                session = self._start(manager, "pass")
                session.read(wait_seconds=1.0, max_output_chars=1024)
                import time

                deadline = 5.0
                waited = 0.0
                while session.process.poll() is None and waited < deadline:
                    time.sleep(0.05)
                    waited += 0.05
                with self.assertRaises(ValidationError):
                    session.write(input_text="x", wait_seconds=0.2, max_output_chars=1024)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
