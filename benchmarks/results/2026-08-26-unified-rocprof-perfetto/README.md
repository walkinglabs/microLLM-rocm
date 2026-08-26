# Unified ROCTX and GPU Perfetto timeline

rocprof captures two ROCTX ranges and two GPU kernels. One correlation ID is shared
by `microllm.test.finished` and the HIP add kernel. The exporter emits six events:
two marker spans, two kernel spans, and one start/finish correlation flow.

Open `unified-perfetto.json` in Perfetto UI.

![Unified timeline](unified-timeline.svg)
