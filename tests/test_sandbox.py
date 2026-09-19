import json
import os
import time

import pytest

from rhg.env.sandbox import OUTPUT_CAP_BYTES, run_python


def run(script, timeout_s=10, mem_mb=1024, **kw):
    return run_python(script, timeout_s=timeout_s, mem_mb=mem_mb, **kw)


def test_ok_run_returns_stdout_and_status():
    r = run("print('hello')")
    assert r.status == "ok" and r.returncode == 0 and r.stdout.strip() == "hello"
    assert r.wall_s > 0


def test_stdin_is_delivered_then_closed():
    r = run("import sys\ndata = sys.stdin.read()\nprint(len(data))\nprint(repr(sys.stdin.read()))", stdin_data="abc" * 100000)
    assert r.status == "ok"
    assert r.stdout.split()[0] == "300000"
    assert "''" in r.stdout


def test_timeout_kills_infinite_loop_quickly():
    t0 = time.monotonic()
    r = run("while True:\n    pass", timeout_s=1.0)
    elapsed = time.monotonic() - t0
    assert r.status == "timeout"
    assert elapsed < 1.0 + 2.0


def test_timeout_kills_child_process_tree():
    marker_free = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    r = run(marker_free, timeout_s=1.5)
    assert r.status == "timeout"
    pid_line = r.stdout.strip().splitlines()
    if pid_line:
        pid = int(pid_line[0])
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.1)
        assert not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_output_is_capped():
    r = run("import sys\nsys.stdout.write('x' * 5_000_000)\nsys.stderr.write('y' * 5_000_000)")
    assert r.status == "ok"
    assert len(r.stdout.encode()) <= OUTPUT_CAP_BYTES and r.stdout_truncated
    assert len(r.stderr.encode()) <= OUTPUT_CAP_BYTES and r.stderr_truncated
    assert OUTPUT_CAP_BYTES == 64 * 1024


def test_endless_output_hits_timeout_with_bounded_memory():
    r = run("import sys\nwhile True:\n    sys.stdout.write('z' * 10000)", timeout_s=1.0)
    assert r.status == "timeout" and len(r.stdout.encode()) <= OUTPUT_CAP_BYTES


def test_env_is_cleared(monkeypatch):
    monkeypatch.setenv("FAKE_SECRET", "hunter2-do-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-do-not-leak")
    r = run("import os, json\nprint(json.dumps(dict(os.environ)))")
    assert r.status == "ok"
    env = json.loads(r.stdout)
    assert "FAKE_SECRET" not in env and "ANTHROPIC_API_KEY" not in env
    assert "hunter2" not in r.stdout and "sk-fake" not in r.stdout
    assert len(env) <= 8


def test_cwd_is_a_fresh_temp_dir_and_removed():
    r = run("import os\nprint(os.getcwd())\nopen('marker.txt', 'w').write('x')\nprint(os.listdir('.'))")
    assert r.status == "ok"
    cwd = r.stdout.splitlines()[0]
    assert "rhg-sbx-cwd-" in cwd
    assert not os.path.exists(cwd)
    assert os.getcwd() != cwd


def test_explicit_workdir_is_used_and_kept(tmp_path):
    r = run("open('kept.txt', 'w').write('x')", workdir=tmp_path)
    assert r.status == "ok" and (tmp_path / "kept.txt").exists()


def test_syntax_error_is_crash():
    r = run("def f(:\n    pass")
    assert r.status == "crash" and r.returncode != 0 and "SyntaxError" in r.stderr


def test_exception_and_nonzero_exit_are_crash():
    assert run("raise ValueError('boom')").status == "crash"
    assert run("import sys\nsys.exit(3)").returncode == 3


def test_sandbox_process_is_isolated_mode():
    r = run("import sys\nprint(sys.flags.isolated, sys.flags.utf8_mode)")
    assert r.stdout.split() == ["1", "1"]


@pytest.mark.skipif(os.name == "nt", reason="rlimits are POSIX only")
def test_memory_limit_is_oom_or_memoryerror():
    r = run("x = bytearray(900 * 1024 * 1024)\nprint('allocated')", mem_mb=256)
    assert "allocated" not in r.stdout
    assert r.status in ("oom", "crash")


@pytest.mark.skipif(os.name != "nt", reason="warning is Windows only")
def test_windows_emits_one_time_warning(caplog):
    import rhg.env.sandbox as sbx

    sbx._windows_warned = False
    with caplog.at_level("WARNING", logger="rhg.env.sandbox"):
        run("pass")
        run("pass")
    assert len([m for m in caplog.messages if "only the wall-clock timeout" in m]) == 1
