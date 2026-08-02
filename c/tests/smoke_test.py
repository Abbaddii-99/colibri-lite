#!/usr/bin/env python3
"""Smoke tests for the GLM binary — no model required.

Tests:
  1. --help flag prints usage and exits 0
  2. No SNAP env var prints usage and exits 1
  3. Missing SNAP directory exits non-zero
  4. --help mentions core env vars
"""

import subprocess, sys, os, platform

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.normpath(os.path.join(HERE, "..", "glm"))
if platform.system() == "Windows":
    BINARY += ".exe"

failed = 0

def test(name, fn):
    global failed
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

def run(*args, extra_env=None):
    env = os.environ.copy()
    if extra_env is not None:
        env.update(extra_env)
    # Prevent OMP tuning re-exec which would mask our env control
    env["COLI_NO_OMP_TUNE"] = "1"
    return subprocess.run(
        [BINARY] + list(args),
        capture_output=True, timeout=30, env=env,
    )

# ── Test 1: --help ──
def test_help():
    r = run("--help", extra_env={"SNAP": "."})
    assert r.returncode == 0, f"exit {r.returncode}"
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")
    assert "SNAP" in out, "help should mention SNAP"

# ── Test 2: no SNAP ──
def test_no_snap():
    minimal = {"COLI_NO_OMP_TUNE": "1", "PATH": os.environ.get("PATH", ""),
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    if platform.system() == "Windows":
        minimal["PATHEXT"] = os.environ.get("PATHEXT", "")
    r = subprocess.run([BINARY], capture_output=True, timeout=30, env=minimal)
    assert r.returncode == 1, f"exit {r.returncode} (expected 1)"

# ── Test 3: missing SNAP dir ──
def test_bad_snap():
    r = run(extra_env={"SNAP": "/nonexistent/glm_model"})
    assert r.returncode != 0, "should fail with bad SNAP"

# ── Test 4: --help mentions key env vars ──
def test_help_vars():
    r = run("--help")
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")
    for var in ("SNAP", "TEMP", "NGEN", "SEED", "CUDA_DEVICES"):
        assert var in out, f"help should mention {var}"

# ── Test 5: --help mentions SERVE mode env vars ──
def test_help_serve_vars():
    r = run("--help")
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")
    for var in ("SERVE", "KV_SLOTS", "CTX"):
        assert var in out, f"help should mention {var}"

# ── Test 6: --help still works with SNAP set ──
def test_help_with_snap():
    r = run("--help", extra_env={"SNAP": "."})
    assert r.returncode == 0, f"exit {r.returncode}"
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")
    assert "SNAP" in out

# ── Test 7: LLaMA synthetic model loads and runs forward pass ──
def test_llama_synthetic():
    snap_dir = os.path.normpath(os.path.join(HERE, "..", "snap_llama_test"))
    score_inp = os.path.join(snap_dir, "score_input.txt")
    if not os.path.exists(snap_dir):
        subprocess.run([sys.executable, os.path.join(HERE, "gen_llama.py")], check=True, cwd=HERE)
    if not os.path.exists(score_inp):
        with open(score_inp, "w") as f:
            f.write("2 3 10 20 30 40 50\n")
    r = run(extra_env={"SNAP": snap_dir, "SCORE": score_inp})
    assert r.returncode == 0, f"exit {r.returncode}"
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")
    assert "layers=2" in out, "should report 2 layers"
    assert "experts=0" in out, "LLaMA has no experts"
    assert "resident dense:" in out, "should report resident bytes"
    lines = out.strip().splitlines()
    score_lines = [l for l in lines if l.strip() and l.strip()[0] in "-0123456789" and " " in l]
    assert len(score_lines) > 0, f"expected score output, got: {lines}"
    parts = score_lines[0].split()
    assert len(parts) == 3, f"expected 3 fields, got {parts}"
    float(parts[0])  # logprob
    int(parts[1])    # contlen
    int(parts[2])    # greedy

# ── Test 8: GLM synthetic model loads and runs forward pass ──
def test_glm_synthetic():
    snap_dir = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    score_inp = os.path.join(snap_dir, "score_input.txt")
    if not os.path.exists(snap_dir):
        subprocess.run([sys.executable, os.path.join(HERE, "gen_glm.py")], check=True, cwd=HERE)
    if not os.path.exists(score_inp):
        with open(score_inp, "w") as f:
            f.write("2 3 10 20 30 40 50\n")
    r = run(extra_env={"SNAP": snap_dir, "SCORE": score_inp})
    assert r.returncode == 0, f"exit {r.returncode}"
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace")
    assert "layers=2" in out, "should report 2 layers"
    assert "experts=4" in out, "GLM should have 4 experts"
    assert "resident dense:" in out, "should report resident bytes"
    lines = out.strip().splitlines()
    score_lines = [l for l in lines if l.strip() and l.strip()[0] in "-0123456789" and " " in l]
    assert len(score_lines) > 0, f"expected score output, got: {lines}"
    parts = score_lines[0].split()
    assert len(parts) == 3, f"expected 3 fields, got {parts}"
    float(parts[0]); int(parts[1]); int(parts[2])

# ── Run ──
if __name__ == "__main__":
    print(f"Smoke tests for {BINARY}")
    test("--help exits 0", test_help)
    test("no SNAP exits 1", test_no_snap)
    test("bad SNAP exits non-zero", test_bad_snap)
    test("--help mentions core vars", test_help_vars)
    test("--help mentions SERVE vars", test_help_serve_vars)
    test("--help with SNAP set", test_help_with_snap)
    test("LLaMA synthetic loads + forward pass", test_llama_synthetic)
    test("GLM synthetic loads + forward pass", test_glm_synthetic)
    sys.exit(failed)
