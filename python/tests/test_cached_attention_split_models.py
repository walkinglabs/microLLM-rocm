#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/single_gpu/compare_cached_attention_split_models.py"


FAKE = r'''#!/usr/bin/env python3
import array
import json
import sys

a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
b = int(a["--batch"])
n = int(a["--new-tokens"])
steps = int(a["--steps"])
warmup = int(a["--warmup"])
splits = int(a["--cached-attention-splits"])
minimum = int(a["--cached-attention-minimum-sequence"])
tokens = [int(value) for value in a["--tokens"].split(",")]
values = [float(index) / 32.0 + (1.0e-6 if splits else 0.0)
          for index in range(b * 16)]
with open(a["--cache-logits-output"], "wb") as stream:
    array.array("f", values).tofile(stream)
record = {
    "parameter_count": 1234,
    "token_count": len(tokens),
    "batch": b,
    "decode_tokens": n,
    "warmup": warmup,
    "steps": steps,
    "cache_mode": "cached",
    "decode_mode": "steady",
    "decode_step_semantics": "one_model_forward_per_measured_token",
    "kv_cache_dtype": a["--kv-cache-dtype"],
    "kv_cache_capacity_tokens": int(a["--cache-capacity"]),
    "kv_cache_active_tokens": len(tokens) + n,
    "kv_cache_actual_bytes": 4096,
    "kv_cache_active_bytes": 4096,
    "cached_attention_splits": splits,
    "cached_attention_minimum_sequence": minimum,
    "measured_tokens": b * n * steps,
    "measured_forward_steps": b * n * steps,
    "generated_tokens": list(range(n)),
    "decode_tokens_per_second": 150.0 if splits else 100.0,
    "engine_peak_bytes": 1020 if splits else 1000,
    "engine_allocation_calls": 130 if splits else 100,
    "engine_backend_allocation_calls": 10,
}
print(json.dumps(record))
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "config.json"
        config.write_text(json.dumps({"vocab_size": 16}), encoding="utf-8")
        weights = root / "weights.bin"
        weights.write_bytes(b"fixture")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "models": [{
                "name": "fixture",
                "revision": "fixture-r1",
                "parameter_count": 1234,
                "config": str(config),
                "weights": str(weights),
                "inference": {"token_ids": [1, 2, 3]},
            }],
        }), encoding="utf-8")
        fake = root / "fake_model.py"
        fake.write_text(FAKE, encoding="utf-8")
        os.chmod(fake, 0o755)
        output = root / "output"
        completed = subprocess.run([
            sys.executable, str(RUNNER), "--manifest", str(manifest),
            "--binary", str(fake), "--output-directory", str(output),
            "--model", "fixture", "--context", "8", "--batch", "2",
            "--decode-tokens", "4", "--cache-dtype", "bf16",
            "--splits", "4", "--minimum-sequence", "4",
            "--warmup", "1", "--steps", "2", "--runs", "3",
        ], text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        raw = (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        pairs = (output / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        chart = (output / "comparison.svg").read_text(encoding="utf-8")
        assert summary["status"] == "pass"
        assert summary["process_rows"] == 6
        assert summary["pair_rows"] == 3
        assert summary["median_throughput_speedup"] == 1.5
        assert summary["all_generated_tokens_equal"] is True
        assert summary["accuracy_gate_passed"] is True
        assert summary["performance_gate_passed"] is True
        assert summary["median_peak_bytes_delta"] == 20
        assert summary["median_allocation_calls_delta"] == 30
        assert summary["median_backend_allocation_calls_delta"] == 0
        assert summary["maximum_logit_error"] < 1.1e-6
        assert len(raw) == 6
        assert len(pairs) == 3
        assert "Official model · current vs split cached Attention" in chart
        assert "median speedup  1.5000x" in chart
    print("cached Attention split model comparison contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
