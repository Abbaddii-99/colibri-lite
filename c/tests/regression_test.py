#!/usr/bin/env python3
"""Regression test: compare modular engine output against the original monolithic glm.c.

Requires the original monolithic binary (glm_debug_new.exe / glm_orig) to be compiled
at a known path.  On CI this is built in the "Compile original monolithic glm.c" step.

Usage: python tests/regression_test.py [SNAP=<path>]
"""

import subprocess, sys, os, platform

HERE = os.path.dirname(os.path.abspath(__file__))
MODULAR = os.path.normpath(os.path.join(HERE, "..", "glm"))
ORIG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "colibri-main", "c"))
ORIG = os.path.join(ORIG_DIR, "glm_debug_new.exe") if platform.system() == "Windows" else os.path.join(ORIG_DIR, "glm_orig")
if not os.path.exists(ORIG):
    ORIG = os.path.join(ORIG_DIR, "glm_orig.exe")
if not os.path.exists(ORIG):
    ORIG = os.path.join(ORIG_DIR, "glm_orig")
if not os.path.exists(ORIG):
    ORIG = os.path.join(ORIG_DIR, "glm")

if platform.system() == "Windows":
    if not MODULAR.endswith(".exe"):
        MODULAR += ".exe"
    if not ORIG.endswith(".exe"):
        ORIG += ".exe"

failed = 0

def run(binary, env):
    return subprocess.run([binary], capture_output=True, timeout=30, env=env)

def test(name, fn):
    global failed
    try:
        fn()
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1

def compare_score(score_file, config_name="default", extra_env=None):
    snap = os.environ.get("SNAP", os.path.normpath(os.path.join(HERE, "..", "snap_glm_test")))
    if not os.path.exists(snap):
        subprocess.run([sys.executable, os.path.join(HERE, "gen_glm.py")], check=True, cwd=HERE)
    score_path = os.path.join(snap, score_file)
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Score file not found: {score_path}")

    env = os.environ.copy()
    env["SNAP"] = snap
    env["SCORE"] = score_path
    env["COLI_NO_OMP_TUNE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if extra_env:
        env.update(extra_env)

    r1 = run(MODULAR, env)
    r2 = run(ORIG, env)

    out1 = (r1.stderr + r1.stdout).decode("utf-8", errors="replace")
    out2 = (r2.stderr + r2.stdout).decode("utf-8", errors="replace")

    lines1 = [l.strip() for l in out1.splitlines() if l.strip() and l.strip()[0] in "-0123456789" and " " in l]
    lines2 = [l.strip() for l in out2.splitlines() if l.strip() and l.strip()[0] in "-0123456789" and " " in l]

    if not lines1 or not lines2:
        raise AssertionError(f"No score output: mod={lines1}, orig={lines2}")

    if lines1[0] != lines2[0]:
        raise AssertionError(f"Mismatch: mod={lines1[0]} orig={lines2[0]}")
    print(f"  PASS  {score_file} ({config_name}): {lines1[0]}")

def test_glm_score_idot0():
    compare_score("score_s4.txt", "IDOT=0", {"IDOT": "0"})

def test_glm_score_idot1():
    compare_score("score_s4.txt", "IDOT=1")

def test_glm_score_simple():
    compare_score("score_simple.txt", "IDOT=1")

def test_glm_s2():
    compare_score("score_s2.txt", "S=2")

def test_glm_s3():
    compare_score("score_s3.txt", "S=3")

if __name__ == "__main__":
    if not os.path.exists(ORIG):
        print(f"SKIP  regression: original binary not found at {ORIG}")
        sys.exit(0)

    print(f"Regression: {MODULAR} vs {ORIG}")

    test("score_s4.txt (IDOT=0)", test_glm_score_idot0)
    test("score_s4.txt (IDOT=1, OMP=1)", test_glm_score_idot1)
    test("score_simple.txt", test_glm_score_simple)
    test("score_s2.txt (S=2)", test_glm_s2)
    test("score_s3.txt (S=3)", test_glm_s3)

    sys.exit(failed)
