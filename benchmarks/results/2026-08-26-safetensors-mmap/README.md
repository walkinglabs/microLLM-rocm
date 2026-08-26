# Read-only mmap safetensors visits

On POSIX hosts, `visit_safetensors` maps the file read-only and gives each callback a
bounded payload span. The report exposes mapping use, tensor count, and payload bytes.
If mapping is unavailable, the previous one-buffer streaming fallback remains.

![mmap visit](mmap-visit.svg)
