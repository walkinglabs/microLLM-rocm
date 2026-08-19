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
