# 2026-08-19 — pure C++ training CLI and real-text smoke

## CLI

`microllm_train` accepts text path, tiny/Model-S/Model-M, CPU/HIP, seed, steps, batch,
context, learning rate, save checkpoint, and resume checkpoint. It records model/data
summaries, step/loss/gradient/cursor JSONL, and restores optimizer/experiment cursor.

CPU CTest fixtures run a two-step save followed by a one-step resume on tracked
generated text. Core training and evaluation remain C++ with no Python frontend.

## Model-S TinyStories HIP smoke

An immutable one-megabyte prefix of the official TinyStories train file was byte-
tokenized and used for 10 sequential B1/context-8 steps on MI300X:

```text
step 1  loss 9.435515404
step 2  loss 9.163564682
step 3  loss 9.368831635
step 4  loss 8.902405739
step 5  loss 9.481015205
step 6  loss 7.261631012
step 7  loss 7.996677399
step 8  loss 8.599428177
step 9  loss 7.950082779
step 10 loss 8.482085228
```

Total wall time was about 5.159 seconds. Different sequential windows make loss
non-monotonic; this is real-text connectivity smoke, not a convergence curve. No
validation loss, generation quality, or full-dataset token count is claimed.

Raw JSONL and a dataset/model/environment manifest are committed under
`benchmarks/results`.
