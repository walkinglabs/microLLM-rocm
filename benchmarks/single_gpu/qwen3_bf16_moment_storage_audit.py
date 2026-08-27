#!/usr/bin/env python3
"""Measure the additional error of BF16 AdamW moments on official Qwen3."""
from __future__ import annotations
import argparse,json
from pathlib import Path
try:
 from .qwen3_training_all_parameter_audit import compare_files,run_json,shaped_manifest
 from .qwen3_training_adamw_state_audit import compare as compare_moments
except ImportError:
 from qwen3_training_all_parameter_audit import compare_files,run_json,shaped_manifest
 from qwen3_training_adamw_state_audit import compare as compare_moments
LIMITS={'loss':0.05,'parameter_max':2e-4,'parameter_rms':1e-5,'moment_max':0.05,'moment_rms':1e-3}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('manifest','micro-binary','output-directory'):p.add_argument(f'--{n}',type=Path,required=True)
 p.add_argument('--model',default='qwen3-0.6b');p.add_argument('--context',type=int,default=32);p.add_argument('--steps',type=int,default=3);p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--timeout-seconds',type=int,default=2400);a=p.parse_args()
 if a.output_directory.exists() and any(a.output_directory.iterdir()):p.error('output directory must be empty')
 a.output_directory.mkdir(parents=True,exist_ok=True);m,_=shaped_manifest(a.manifest,a.model,a.context,a.learning_rate)
 def base():return [str(a.micro_binary),'--config',m['config'],'--weights',m['weights'],'--tokens',m['training']['tokens'],'--device','hip','--learning-rate',str(a.learning_rate),'--warmup','0','--steps',str(a.steps),'--batch','1','--linear-precision','bf16','--bf16-weight-mirrors','true']
 rp=a.output_directory/'fp32-moment-parameters.safetensors';rm=a.output_directory/'fp32-moments.safetensors';rl=a.output_directory/'fp32-loss.json';cp=a.output_directory/'bf16-moment-parameters.safetensors';cm=a.output_directory/'bf16-moments.safetensors';cl=a.output_directory/'bf16-loss.json'
 ref=run_json(base()+['--adamw-moment-precision','fp32','--all-parameters-output',str(rp),'--all-moments-output',str(rm),'--loss-trajectory-output',str(rl)],a.timeout_seconds)
 cand=run_json(base()+['--adamw-moment-precision','bf16','--adamw-bf16-multi-tensor-threshold','auto','--all-parameters-output',str(cp),'--all-moments-output',str(cm),'--loss-trajectory-output',str(cl)],a.timeout_seconds)
 pr,params,_=compare_files(cp,rp,'parameter');mr,moments=compare_moments(cm,rm);lr=json.loads(rl.read_text())['losses'];lc=json.loads(cl.read_text())['losses'];ld=[abs(x-y) for x,y in zip(lc,lr)]
 gates={'loss':max(ld)<=LIMITS['loss'],'parameter_max':params['maximum_absolute_difference']<=LIMITS['parameter_max'],'parameter_rms':params['rms_difference']<=LIMITS['parameter_rms'],'moment_max':moments['maximum_absolute_difference']<=LIMITS['moment_max'],'moment_rms':moments['rms_difference']<=LIMITS['moment_rms'],'moment_bytes_half':cand['adamw_moment_state_bytes']*2==ref['adamw_moment_state_bytes']}
 files=(rp,rm,cp,cm);temp=sum(x.stat().st_size for x in files)
 for x in files:x.unlink()
 s={'schema_version':1,'record_type':'qwen3_bf16_moment_storage_audit','status':'pass' if all(gates.values()) else 'precision_mismatch','model':m['name'],'steps':a.steps,'reference_policy':'BF16 forward + FP32 moments','candidate_policy':'BF16 forward + BF16 moments','losses':{'reference':lr,'candidate':lc,'absolute_differences':ld,'maximum_absolute_difference':max(ld)},'parameters':params,'moments':moments,'moment_state_bytes':{'reference':ref['adamw_moment_state_bytes'],'candidate':cand['adamw_moment_state_bytes']},'multi_tensor':{'threshold':cand['adamw_bf16_multi_tensor_threshold'],'tensors':cand['adamw_bf16_multi_tensor_tensors'],'elements':cand['adamw_bf16_multi_tensor_elements']},'limits':LIMITS,'gates':gates,'temporary_export_bytes':temp,'temporary_exports_removed':all(not x.exists() for x in files),'boundary':'additional BF16-moment error inside an already rejected cross-framework BF16 forward path; no PyTorch parity claim'}
 (a.output_directory/'raw.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in [*pr,*mr]));(a.output_directory/'summary.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n');(a.output_directory/'workers.json').write_text(json.dumps({'reference':ref,'candidate':cand},indent=2,sort_keys=True)+'\n');print(json.dumps(s,sort_keys=True));return 0 if s['status']=='pass' else 2
if __name__=='__main__':raise SystemExit(main())
