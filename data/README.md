# Data registry

Generated smoke data belongs in the repository when small. Downloaded corpora are
not committed. Every real dataset must have a registry entry with source, license,
version/hash, boundaries, split method, tokenizer, token counts, and evidence state.

See [registry.toml](registry.toml).

Download an immutable one-megabyte validation prefix for loader/training smoke:

```bash
./scripts/fetch_tinystories_smoke.sh
```

This prefix is not a validation benchmark and may end mid-document. Reference training
must use the official complete train/validation files at the recorded revision.

Pinned official model fixtures are described in
[model_fixtures.toml](model_fixtures.toml). Prepare a local benchmark manifest without
committing model payloads:

```bash
python3 tools/prepare_hf_fixture.py prepare \
  --download-root /absolute/path/to/models \
  --manifest /absolute/path/to/hf-models.local.json \
  --evidence /tmp/hf-fixture-evidence.json

python3 tools/prepare_hf_fixture.py validate \
  --manifest /absolute/path/to/hf-models.local.json
```

Download mode requires the optional `huggingface_hub` package. Existing model and
tokenizer directories can instead be supplied with repeated `--model-source` and
`--tokenizer-source` arguments. The tool parses complete safetensors headers and checks
parameter/Tensor counts, config, vocab and merges before writing the manifest. Registry
`parameter_count` means values physically stored in safetensors. A tied model may also
declare `runtime_parameter_count`; generated manifests retain both counts while their
legacy `parameter_count` field follows the runtime model expected by C++ and PyTorch.
