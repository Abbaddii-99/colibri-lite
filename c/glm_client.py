#!/usr/bin/env python3
"""Python client for the Colibri-Lite inference engine.

Communicates with the C engine via stdin/stdout using the serve protocol.

Usage:
  # Start the engine:
  SNAP=/path/to/model python glm_client.py

  # Or connect to a running engine:
  python glm_client.py --connect /tmp/glm.sock

Commands:
  Type your message and press Enter to generate a response.
  :reset  — clear conversation history
  :quit   — exit
"""

import subprocess, sys, os, json, argparse, time


class GlmClient:
    """Client for the GLM engine's serve protocol."""

    def __init__(self, snap=None, binary="glm", ebits=8, dbits=8, cache=64):
        self.proc = None
        self.binary = binary
        if snap or "SNAP" not in os.environ:
            os.environ["SNAP"] = snap or os.environ.get("SNAP", "")
        os.environ["COLI_NO_OMP_TUNE"] = "1"
        args = [binary]
        if cache:
            args.append(str(cache))
        if ebits:
            args.append(str(ebits))
        if dbits:
            args.append(str(dbits))
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        # Read startup banner
        self._read_until(b"READY")

    def _read_until(self, marker):
        buf = b""
        while marker not in buf:
            ch = self.proc.stdout.read(1)
            if not ch:
                break
            buf += ch
        return buf

    def generate(self, text):
        """Send text, yield detokenized tokens as they arrive."""
        self.proc.stdin.write(text.encode("utf-8") + b"\n")
        self.proc.stdin.flush()
        buf = b""
        end_marker = b"\x01\x01END\x01\x01\n"
        while True:
            ch = self.proc.stdout.read(1)
            if not ch:
                break
            buf += ch
            if buf.endswith(end_marker):
                break
        result = buf[: -len(end_marker)].decode("utf-8", errors="replace")
        return result

    def reset(self):
        self.proc.stdin.write(b"\x01\x01RESET\n")
        self.proc.stdin.flush()

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
            self.proc = None


def main():
    parser = argparse.ArgumentParser(description="GLM inference client")
    parser.add_argument("--snap", help="Model snapshot directory")
    parser.add_argument("--binary", default="glm", help="Path to GLM binary")
    parser.add_argument("--ebits", type=int, default=8)
    parser.add_argument("--dbits", type=int, default=8)
    parser.add_argument("--cache", type=int, default=64)
    parser.add_argument("--prompt", "-p", help="Single prompt (non-interactive)")
    parser.add_argument("--ngen", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()

    client = GlmClient(args.snap, args.binary, args.ebits, args.dbits, args.cache)

    if args.prompt:
        os.environ["NGEN"] = str(args.ngen)
        result = client.generate(args.prompt)
        print(result)
        client.close()
        return

    print("Chat ready. Type your message. :reset to clear, :quit to exit.")
    try:
        while True:
            line = input("> ").strip()
            if line == ":quit":
                break
            if line == ":reset":
                client.reset()
                print("[reset]")
                continue
            if not line:
                continue
            t0 = time.time()
            result = client.generate(line)
            dt = time.time() - t0
            print(result)
            toks = len(result.split())
            print(f"[{toks} tok, {dt:.2f}s, {toks/dt:.1f} tok/s]")
    except (EOFError, KeyboardInterrupt):
        pass
    client.close()


if __name__ == "__main__":
    main()
