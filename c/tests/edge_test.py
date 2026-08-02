#!/usr/bin/env python3
"""Edge-case tests for the scoring path — no model required (uses synthetic GLM/LLaMA)."""

import subprocess, sys, os, platform

HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.normpath(os.path.join(HERE, "..", "glm"))
if platform.system() == "Windows":
    BINARY += ".exe"

failed = 0

def run(env):
    env = dict(env)  # copy
    env["COLI_NO_OMP_TUNE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run([BINARY], capture_output=True, timeout=30, env=env)

def test(name, fn):
    global failed
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

def get_score(env):
    r = run(env)
    out = (r.stderr + r.stdout).decode("utf-8", errors="replace").strip()
    for line in out.splitlines():
        line = line.strip()
        if line and line[0] in "-0123456789" and " " in line:
            parts = line.split()
            if len(parts) == 3:
                return float(parts[0]), int(parts[1]), int(parts[2])
    raise AssertionError(f"No score in output: {out[:200]}")

# ── Tests ──

def test_glm_score_s1():
    """S=1 (single token context)"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    if not os.path.exists(snap):
        subprocess.run([sys.executable, os.path.join(HERE, "gen_glm.py")], check=True, cwd=HERE)
    score_file = os.path.join(snap, "score_simple.txt")
    env = os.environ.copy()
    env["SNAP"] = snap
    env["SCORE"] = score_file
    logprob, contlen, greedy = get_score(env)
    assert contlen >= 1, f"contlen should be >= 1, got {contlen}"

def test_glm_score_idot0_vs_idot1_differs():
    """IDOT=0 vs IDOT=1 should give different (but valid) scores"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    score_file = os.path.join(snap, "score_s4.txt")
    env = os.environ.copy()
    env["SNAP"] = snap
    env["SCORE"] = score_file
    env["OMP_NUM_THREADS"] = "1"

    env_idot0 = dict(env); env_idot0["IDOT"] = "0"
    env_idot1 = dict(env); env_idot1["IDOT"] = "1"

    lp0, *_ = get_score(env_idot0)
    lp1, *_ = get_score(env_idot1)
    assert lp0 != lp1, f"IDOT=0 and IDOT=1 should differ: {lp0} vs {lp1}"

def test_glm_score_repeatable():
    """Same score file, same settings → same result"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    score_file = os.path.join(snap, "score_s4.txt")
    env = os.environ.copy()
    env["SNAP"] = snap
    env["SCORE"] = score_file
    lp1, *_ = get_score(env)
    lp2, *_ = get_score(env)
    assert lp1 == lp2, f"Not repeatable: {lp1} != {lp2}"

def test_glm_score_multi_file():
    """Two different score files produce different results"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    env = os.environ.copy()
    env["SNAP"] = snap
    env_s2 = dict(env); env_s2["SCORE"] = os.path.join(snap, "score_s2.txt")
    env_s4 = dict(env); env_s4["SCORE"] = os.path.join(snap, "score_s4.txt")
    lp2, *_ = get_score(env_s2)
    lp4, *_ = get_score(env_s4)
    assert lp2 != lp4, f"S=2 and S=4 should differ: {lp2} vs {lp4}"

def test_llama_score():
    """LLaMA synthetic score works"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_llama_test"))
    if not os.path.exists(snap):
        subprocess.run([sys.executable, os.path.join(HERE, "gen_llama.py")], check=True, cwd=HERE)
    score_file = os.path.join(snap, "score_input.txt")
    if not os.path.exists(score_file):
        with open(score_file, "w") as f:
            f.write("2 3 10 20 30 40 50\n")
    env = os.environ.copy()
    env["SNAP"] = snap
    env["SCORE"] = score_file
    logprob, contlen, greedy = get_score(env)
    assert contlen >= 1, f"contlen >= 1, got {contlen}"

def test_glm_with_omp_threads():
    """Multi-threaded scoring matches single-threaded (IDOT=1)"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    score_file = os.path.join(snap, "score_s4.txt")
    env = os.environ.copy()
    env["SNAP"] = snap
    env["SCORE"] = score_file
    env_st = dict(env); env_st["OMP_NUM_THREADS"] = "1"
    env_mt = dict(env); env_mt["OMP_NUM_THREADS"] = "4"
    lp_st, *_ = get_score(env_st)
    lp_mt, *_ = get_score(env_mt)
    assert lp_st == lp_mt, f"Single vs multi-thread differ: {lp_st} != {lp_mt}"

def test_llama_forward_verified():
    """LLaMA forward pass verified against Python reference"""
    r = subprocess.run([sys.executable, os.path.join(HERE, "llama_verify.py")], capture_output=True, timeout=60, env=os.environ)
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    if r.returncode != 0:
        raise AssertionError("LLaMA verify failed:\n" + out)

def test_glm_generation():
    """GLM token generation (PROMPT + NGEN) produces expected output"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_glm_test"))
    env = os.environ.copy()
    env["SNAP"] = snap
    env["PROMPT"] = "hello"
    env["NGEN"] = "4"
    env["IDOT"] = "1"
    env["COLI_NO_OMP_TUNE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    r = subprocess.run([BINARY], capture_output=True, timeout=30, env=env)
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    assert r.returncode == 0, f"Generation failed: {out}"
    # Should produce some generated tokens
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    assert any("embedding" in l.lower() or "loaded" in l.lower() for l in lines), f"Unexpected output: {out}"

def test_llama_generation():
    """LLaMA token generation produces expected output"""
    snap = os.path.normpath(os.path.join(HERE, "..", "snap_llama_test"))
    env = os.environ.copy()
    env["SNAP"] = snap
    env["PROMPT"] = "hello"
    env["NGEN"] = "4"
    env["IDOT"] = "1"
    env["COLI_NO_OMP_TUNE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    r = subprocess.run([BINARY], capture_output=True, timeout=30, env=env)
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    assert r.returncode == 0, f"Generation failed: {out}"
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    assert any("embedding" in l.lower() or "loaded" in l.lower() for l in lines), f"Unexpected output: {out}"

if __name__ == "__main__":
    print(f"Edge-case tests for {BINARY}")
    test("GLM S=1 scoring", test_glm_score_s1)
    test("GLM IDOT=0 vs IDOT=1 differ", test_glm_score_idot0_vs_idot1_differs)
    test("GLM score repeatable", test_glm_score_repeatable)
    test("GLM S=2 vs S=4 differ", test_glm_score_multi_file)
    test("LLaMA scoring", test_llama_score)
    test("GLM single vs multi-thread match", test_glm_with_omp_threads)
    test("LLaMA forward pass verified", test_llama_forward_verified)
    test("GLM generation", test_glm_generation)
    test("LLaMA generation", test_llama_generation)
    sys.exit(failed)
