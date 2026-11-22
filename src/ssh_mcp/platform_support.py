"""Operating-system integration points for ssh-mcp.

The tools themselves are portable Python, but interactive SSH needs a *local*
terminal, and POSIX and Windows disagree about nearly everything involved:

* POSIX hands out a PTY with ``os.openpty()``.  Windows has no PTY in the
  standard library at all -- it has ConPTY, a Win32 API with no stdlib binding
  (reached here through the optional ``pywinpty`` package).
* POSIX ends a process tree with ``os.killpg()``.  Windows has neither process
  groups nor signals; the tree has to be terminated explicitly.
* A Windows console program started by a console-less parent (an MCP server
  launched by a GUI client) pops up its own console window unless the caller
  asks for it not to.

Everything platform-specific about spawning, killing, and talking to those
children lives here, so ``ssh.py`` stays platform-agnostic.
"""

from __future__ import annotations

import errno
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

IS_WINDOWS = os.name == "nt"

# Size handed to a freshly created local PTY.  Windows defaults to 80x24 and
# POSIX PTYs start at 0x0 until something sets them, either of which makes
# command output wrap badly and breaks full-screen programs; 40x120 is a better
# starting point for the way agents read output.
DEFAULT_PTY_ROWS = 40
DEFAULT_PTY_COLS = 120
DEFAULT_TERM = "xterm-256color"

#: Overrides automatic backend selection.  "auto" (default), "conpty", "pipe".
PTY_BACKEND_ENV = "SSH_MCP_PTY_BACKEND"

_ALLOWED_BACKENDS = {"auto", "conpty", "pipe"}

#: How long a ConPTY read may block before it re-checks whether the child is
#: still alive, and how long to wait for a final flush once it is gone.
_CONPTY_READ_POLL_SECONDS = 0.2
_CONPTY_EOF_GRACE_SECONDS = 1.0

# Tri-state cache for the pywinpty import probe: None = not probed yet.
_conpty_importable: bool | None = None
_fallback_warned = False


class PtyBackendError(RuntimeError):
    """Raised when no usable local-terminal backend can be created."""


# ---------------------------------------------------------------------------
# Child-process plumbing
# ---------------------------------------------------------------------------


def child_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a spawned ``ssh``/``scp``/``rsync`` child.

    Windows has no ``TERM``.  ``ssh`` forwards whatever it inherits to the
    remote pty, so without a value the remote side ends up with ``TERM=dumb``
    (or an empty TERM) and turns off color, readline editing, and full-screen
    output.  Inheriting an explicit user setting is respected as-is.
    """
    env = dict(os.environ if environment is None else environment)
    if IS_WINDOWS:
        term = env.get("TERM", "").strip().lower()
        if not term or term in {"dumb", "unknown"}:
            env["TERM"] = DEFAULT_TERM
    return env


def spawn_kwargs() -> dict[str, Any]:
    """Popen keyword arguments for a detached, windowless child.

    POSIX: own session, so the whole tree can be signalled later.
    Windows: own process group (signalable with a console break event) and no
    console window -- the client that started the MCP server is usually a GUI
    app, and a console-less child would otherwise flash a console per command.
    """
    if IS_WINDOWS:
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        }
    return {"start_new_session": True}


def _taskkill(pid: int) -> bool:
    """Forcefully terminate *pid* and its children.  Windows only."""
    taskkill = shutil.which("taskkill") or os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe"
    )
    try:
        subprocess.run(
            [taskkill, "/F", "/T", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            **spawn_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def terminate_process(process: Any, *, force: bool) -> None:
    """Ask a child process (tree) to die.

    POSIX maps ``force`` onto ``SIGKILL``/``SIGTERM`` for the process group.
    Windows has no graceful signal that survives a piped/console-less parent,
    so the first request is already a ``TerminateProcess``; a forced request
    additionally asks ``taskkill`` to reap any grandchildren.
    """
    if IS_WINDOWS:
        if force:
            _taskkill(process.pid)
        # `terminate()` is TerminateProcess under the hood: Windows has no
        # graceful signal that reaches a console-less child, so the polite and
        # the forceful path look the same from here.
        try:
            process.terminate()
        except Exception:
            pass
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


# ---------------------------------------------------------------------------
# Local terminal backends
# ---------------------------------------------------------------------------


class LocalPty:
    """A child process attached to a local terminal.

    ``read`` blocks until data or end-of-stream is available.  Callers run it in
    a dedicated reader thread and treat ``b""`` as EOF.
    """

    kind = "abstract"

    def __init__(self, process: Any) -> None:
        self.process = process
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def read(self, size: int = 4096) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def write(self, data: bytes) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        self._closed = True


class PosixPty(LocalPty):
    """A real PTY from ``os.openpty()``."""

    kind = "posix"

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        master_fd, slave_fd = os.openpty()
        try:
            process = subprocess.Popen(
                argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                cwd=cwd,
                env=env,
                **spawn_kwargs(),
            )
        except BaseException:
            os.close(master_fd)
            raise
        finally:
            os.close(slave_fd)
        super().__init__(process)
        self._master_fd = master_fd

    def read(self, size: int = 4096) -> bytes:
        try:
            return os.read(self._master_fd, size)
        except OSError as exc:
            # EIO is how a Linux/macOS PTY reports "the other side hung up".
            if exc.errno in {errno.EIO, errno.EBADF}:
                return b""
            raise

    def write(self, data: bytes) -> int:
        written = 0
        view = memoryview(data)
        while written < len(view):
            chunk = os.write(self._master_fd, view[written:])
            if chunk <= 0:
                raise OSError(errno.EIO, "short write to PTY")
            written += chunk
        return written

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._master_fd)
        except OSError:
            pass


class _WinptyProcessHandle:
    """``subprocess.Popen`` look-alike for a ``pywinpty.PtyProcess``.

    ``ssh.py`` only needs ``poll()``/``returncode``/``pid``/``terminate()`` from
    the child object, so a thin adapter keeps the ConPTY backend from leaking
    into the rest of the module.
    """

    def __init__(self, process: Any) -> None:
        self._process = process
        self.pid = int(getattr(process, "pid", 0) or 0)

    def poll(self) -> int | None:
        try:
            if self._process.isalive():
                return None
            status = self._process.exitstatus
        except Exception:
            return -1
        # pywinpty cannot always report a status for a killed child; -1 keeps
        # the caller from mistaking "unknown" for a clean exit.
        return -1 if status is None else int(status)

    @property
    def returncode(self) -> int | None:
        return self.poll()

    def terminate(self) -> None:
        try:
            self._process.terminate(force=True)
        except Exception:
            pass

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.poll()
            if status is not None:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.05)


class ConPtyPty(LocalPty):
    """A ConPTY-backed terminal (Windows 10 1809+), via ``pywinpty``.

    The child gets a real console, so ``ssh``'s own prompts (password, unknown
    host key) work and the window size is forwarded to the remote pty.
    """

    kind = "conpty"

    #: pywinpty's conpty layer writes this sentinel for window-size queries and
    #: strips it in ``PtyProcess.read()``.  We read the raw stream, so the
    #: filtering has to happen here -- including when it straddles two reads.
    _IGNORE_MARKER = b"0011Ignore"

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        rows: int = DEFAULT_PTY_ROWS,
        cols: int = DEFAULT_PTY_COLS,
    ) -> None:
        from winpty import PtyProcess  # imported lazily: optional dependency

        winpty_process = PtyProcess.spawn(
            list(argv), cwd=cwd, env=dict(env) if env is not None else None, dimensions=(rows, cols)
        )
        super().__init__(_WinptyProcessHandle(winpty_process))
        self._winpty = winpty_process
        # pywinpty surfaces ConPTY output on a loopback socket.  Reading it
        # directly yields raw bytes; PtyProcess.read() instead decodes UTF-8
        # and can stall on binary output, so prefer the socket when present.
        self._output_socket = getattr(winpty_process, "fileobj", None)
        self._carry = b""
        self._dead_since: float | None = None
        if self._output_socket is not None:
            # A recv() on this socket blocks forever if the child dies without
            # closing the ConPTY output side (a process parked in a console
            # read does exactly that), so poll it instead of blocking.
            self._output_socket.settimeout(_CONPTY_READ_POLL_SECONDS)

    def _raw_read(self, size: int) -> bytes | None:
        """One read from the pty.  ``None`` means "nothing yet, try again"."""
        if self._output_socket is not None:
            try:
                return self._output_socket.recv(size)
            except socket.timeout:
                return None
            except OSError:
                return b""
        try:
            text = self._winpty.read(size)
        except EOFError:
            return b""
        return text.encode("utf-8", "replace") if text else b""

    def read(self, size: int = 4096) -> bytes:
        while True:
            chunk = self._raw_read(size)
            if chunk is None:
                # Silence is only end-of-stream once the child is gone: an
                # interactive session can sit idle for minutes, and a session
                # that has not printed its banner yet is not finished either.
                # Once the child has exited, allow a short grace period for
                # the last flush and then report EOF, so a reader thread can
                # never be parked on a dead session.
                if self.process.poll() is None:
                    self._dead_since = None
                    continue
                now = time.monotonic()
                if self._dead_since is None:
                    self._dead_since = now
                elif now - self._dead_since >= _CONPTY_EOF_GRACE_SECONDS:
                    carry, self._carry = self._carry, b""
                    return carry
                continue
            if not chunk:
                carry, self._carry = self._carry, b""
                return carry
            self._dead_since = None
            data = self._carry + chunk
            self._carry = b""
            cleaned = data.replace(self._IGNORE_MARKER, b"")
            # ConPTY re-renders the console screen buffer, so a line that the
            # child ended with CRLF comes back as CR CR LF.  A terminal hides
            # that; transcripts and tool output should not.
            cleaned = cleaned.replace(b"\r\r\n", b"\r\n")
            # Keep a trailing partial marker, or a trailing CR CR (which may be
            # the start of the sequence above), buffered for the next read.
            for cut in range(len(self._IGNORE_MARKER) - 1, 0, -1):
                if cleaned.endswith(self._IGNORE_MARKER[:cut]):
                    self._carry = cleaned[-cut:]
                    cleaned = cleaned[:-cut]
                    break
            else:
                if cleaned.endswith(b"\r\r"):
                    self._carry = cleaned[-2:]
                    cleaned = cleaned[:-2]
            if cleaned:
                return cleaned

    def write(self, data: bytes) -> int:
        # A Windows console delivers Enter as CR, while callers send the LF a
        # remote pty expects.  Without this translation a line never submits:
        # ssh's own password/host-key prompts hang, and so does any child
        # reading from the console.
        text = data.decode("utf-8", "replace").replace("\r\n", "\r").replace("\n", "\r")
        try:
            self._winpty.write(text)
        except (EOFError, OSError) as exc:
            raise OSError(errno.EPIPE, "the pseudo-terminal is closed") from exc
        return len(data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # pywinpty marks itself closed as soon as the child exits, and its own
        # close() then returns early -- leaking the two loopback sockets behind
        # the ConPTY channel.  Reset the flag so the real cleanup runs.
        try:
            self._winpty.closed = False
        except Exception:
            pass
        try:
            self._winpty.close(force=True)
        except Exception:
            pass


class PipePty(LocalPty):
    """Dependency-free fallback: pipes locally, PTY on the remote side.

    Interactive transports are always invoked with ``-tt``, which forces the
    *remote* to allocate a pty even when the client has no local terminal --
    and the remote pty is what interactive programs actually need.  What is
    lost compared with a real local terminal: ssh's own password/host-key
    prompts have no console to draw on, and no window size is forwarded (the
    remote pty starts at 0x0 until the program sets one).
    """

    kind = "pipe"

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Merge stderr into the stream, matching what a single pty does for
            # the other backends; it also keeps an undrained stderr pipe from
            # blocking the child.
            stderr=subprocess.STDOUT,
            bufsize=0,
            cwd=cwd,
            env=env,
            **spawn_kwargs(),
        )
        super().__init__(process)
        assert process.stdout is not None  # for type checkers
        self._output_fd = process.stdout.fileno()

    def read(self, size: int = 4096) -> bytes:
        try:
            return os.read(self._output_fd, size)
        except OSError:
            return b""

    def write(self, data: bytes) -> int:
        stdin = self.process.stdin
        if stdin is None:
            raise OSError(errno.EPIPE, "the pipe to ssh is closed")
        try:
            written = stdin.write(data)
            stdin.flush()
        except (OSError, ValueError) as exc:
            raise OSError(errno.EPIPE, "the pipe to ssh is closed") from exc
        return len(data) if written is None else int(written)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                pass


def conpty_available() -> bool:
    """True when ConPTY can be used (Windows with ``pywinpty`` installed)."""
    global _conpty_importable
    if not IS_WINDOWS:
        return False
    if _conpty_importable is None:
        try:
            import winpty  # noqa: F401

            _conpty_importable = True
        except Exception:
            _conpty_importable = False
    return _conpty_importable


def requested_backend() -> str:
    """Backend name from the environment, normalized; ``"auto"`` if unset."""
    raw = os.environ.get(PTY_BACKEND_ENV, "").strip().lower()
    if not raw:
        return "auto"
    if raw not in _ALLOWED_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_BACKENDS))
        raise PtyBackendError(f"{PTY_BACKEND_ENV} must be one of: {allowed} (got {raw!r}).")
    return raw


def resolve_backend_name(requested: str | None = None) -> str:
    """Name of the backend that would be used: ``posix``, ``conpty``, ``pipe``."""
    if not IS_WINDOWS:
        return "posix"
    name = (requested or requested_backend()).strip().lower()
    if name in {"", "auto"}:
        return "conpty" if conpty_available() else "pipe"
    if name not in _ALLOWED_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_BACKENDS))
        raise PtyBackendError(f"{PTY_BACKEND_ENV} must be one of: {allowed} (got {name!r}).")
    if name == "conpty" and not conpty_available():
        raise PtyBackendError(
            "ConPTY was requested but 'pywinpty' is not installed. "
            'Install it with: pip install pywinpty (or reinstall with: pip install ssh-mcp-win)'
        )
    return name


def _warn_fallback(reason: Any) -> None:
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    print(
        f"ssh-mcp: ConPTY backend unavailable ({reason}); falling back to pipes. "
        "Interactive sessions still work, but terminal size is not forwarded.",
        file=sys.stderr,
        flush=True,
    )


def spawn_local_pty(
    argv: list[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    rows: int = DEFAULT_PTY_ROWS,
    cols: int = DEFAULT_PTY_COLS,
) -> LocalPty:
    """Start *argv* attached to the best local terminal available."""
    environment = child_environment(env)
    name = requested_backend() if IS_WINDOWS else "posix"
    if IS_WINDOWS:
        resolved = resolve_backend_name(name)
        if resolved == "conpty":
            try:
                return ConPtyPty(argv, cwd=cwd, env=environment, rows=rows, cols=cols)
            except Exception as exc:  # ConPTY missing/blocked on this host
                if name == "conpty":
                    raise PtyBackendError(f"Failed to create a ConPTY terminal: {exc}") from exc
                _warn_fallback(exc)
        return PipePty(argv, cwd=cwd, env=environment)
    return PosixPty(argv, cwd=cwd, env=environment)


def wait_for_exit(process: Any, timeout_seconds: float) -> bool:
    """Wait up to *timeout_seconds* for *process* to exit; True if it did."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return process.poll() is not None


def reap(process: Any, timeout: float = 2.0) -> int | None:
    """Collect the child's exit status, tolerating a wedged child.

    A backend can report end-of-stream a moment before Windows has finished
    tearing the process down, in which case ``returncode`` is still unset; the
    caller needs a real status to report, so give the child a short grace
    period to be reaped.
    """
    code = process.poll()
    if code is not None:
        return code
    wait = getattr(process, "wait", None)
    if wait is None:
        return process.poll()
    try:
        return wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return process.poll()
    except Exception:
        return process.poll()
