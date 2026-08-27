#!/usr/bin/env python3
"""Audit official Qwen3 loss, parameters, moments and step after three steps."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
try:
    from .qwen3_training_all_parameter_audit import compare_files, run_json, shaped_manifest
    from .qwen3_training_adamw_state_audit import compare as compare_moments
except ImportError:
    from qwen3_training_all_parameter_audit import compare_files, run_json, shaped_manifest
    from qwen3_training_adamw_state_audit import compare as compare_moments

LIMITS={"fp32":{"loss":1e-5,"parameter_max":3e-4,"parameter_rms":3e-6,"moment_max":3e-3,"moment_rms":3e-6},"bf16":{"loss":5e-3,"parameter_max":2e-4,"parameter_rms":2e-6,"moment_max":1e-2,"moment_rms":1e-4}}

def options():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","micro-binary","pytorch-python","pytorch-runner","output-directory"): p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--precision",choices=("fp32","bf16"),required=True); p.add_argument("--model",default="qwen3-0.6b"); p.add_argument("--context",type=int,default=32); p.add_argument("--steps",type=int,default=3); p.add_argument("--learning-rate",type=float,default=1e-5); p.add_argument("--timeout-seconds",type=int,default=2400); p.add_argument("--allow-amdsmi-fallback",action="store_true")
    a=p.parse_args()
    if a.steps<2 or (a.output_directory.exists() and any(a.output_directory.iterdir())): p.error("steps must be >=2 and output directory empty")
    return a

def main():
    a=options(); a.output_directory.mkdir(parents=True,exist_ok=True)
    model,manifest=shaped_manifest(a.manifest,a.model,a.context,a.learning_rate); model["training"]["steps"]=a.steps; manifest["models"][0]["training"]["steps"]=a.steps
    mp=a.output_directory/"micro-parameters.safetensors"; mm=a.output_directory/"micro-moments.safetensors"; tp=a.output_directory/"pytorch-parameters.safetensors"; tm=a.output_directory/"pytorch-moments.safetensors"; ml=a.output_directory/"micro-losses.json"; shaped=a.output_directory/"manifest.json"; shaped.write_text(json.dumps(manifest,indent=2)+"\n")
    micro=[str(a.micro_binary),"--config",model["config"],"--weights",model["weights"],"--tokens",model["training"]["tokens"],"--device","hip","--learning-rate",str(a.learning_rate),"--warmup","0","--steps",str(a.steps),"--batch","1","--linear-precision",a.precision,"--bf16-weight-mirrors","true" if a.precision=="bf16" else "false","--adamw-moment-precision","fp32","--all-parameters-output",str(mp),"--all-moments-output",str(mm),"--loss-trajectory-output",str(ml)]
    torch=[str(a.pytorch_python),str(a.pytorch_runner),"--manifest",str(shaped),"--device","cuda","--dtype","bf16_amp" if a.precision=="bf16" else "fp32","--worker-model",model["name"],"--worker-mode","train","--all-parameters-output",str(tp),"--all-moments-output",str(tm)]
    if a.allow_amdsmi_fallback: torch.append("--allow-amdsmi-fallback")
    mw=run_json(micro,a.timeout_seconds); tw=run_json(torch,a.timeout_seconds)
    losses_m=json.loads(ml.read_text())["losses"]; losses_t=tw["loss_trajectory"]
    if len(losses_m)!=a.steps or len(losses_t)!=a.steps or mw["all_moment_step"]!=a.steps or tw["all_moment_step"]!=a.steps: raise RuntimeError("multi-step count contract failed")
    pr,params,_=compare_files(mp,tp,"parameter"); mr,moments=compare_moments(mm,tm)
    loss_diffs=[abs(float(x)-float(y)) for x,y in zip(losses_m,losses_t)]; lim=LIMITS[a.precision]
    gates={"loss":max(loss_diffs)<=lim["loss"],"parameter_max":params["maximum_absolute_difference"]<=lim["parameter_max"],"parameter_rms":params["rms_difference"]<=lim["parameter_rms"],"moment_max":moments["maximum_absolute_difference"]<=lim["moment_max"],"moment_rms":moments["rms_difference"]<=lim["moment_rms"],"step":mw["all_moment_step"]==tw["all_moment_step"]==a.steps}
    files=(mp,mm,tp,tm); export_bytes=sum(x.stat().st_size for x in files)
    for x in files: x.unlink()
    summary={"schema_version":1,"record_type":"qwen3_training_multistep_state_audit","status":"pass" if all(gates.values()) else "precision_mismatch","model":model["name"],"precision":a.precision,"steps":a.steps,"losses":{"microllm":losses_m,"pytorch":losses_t,"absolute_differences":loss_diffs,"maximum_absolute_difference":max(loss_diffs)},"parameters":params,"moments":moments,"limits":lim,"gates":gates,"temporary_export_bytes":export_bytes,"temporary_exports_removed":all(not x.exists() for x in files),"boundary":"three consecutive steps at B1/T32; longer trajectories, checkpoint restart and BF16 moment storage remain separate"}
    (a.output_directory/"raw.jsonl").write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in [*pr,*mr])); (a.output_directory/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (a.output_directory/"workers.json").write_text(json.dumps({"microllm":mw,"pytorch":tw},indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,sort_keys=True)); return 0 if summary["status"]=="pass" else 2
if __name__=="__main__": raise SystemExit(main())
