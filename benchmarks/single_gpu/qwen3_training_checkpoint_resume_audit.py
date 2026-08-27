#!/usr/bin/env python3
"""Compare uninterrupted and process-restarted Qwen3 training state."""
from __future__ import annotations
import argparse,json
from pathlib import Path
try:
 from .qwen3_training_all_parameter_audit import compare_files,run_json,shaped_manifest
 from .qwen3_training_adamw_state_audit import compare as compare_moments
except ImportError:
 from qwen3_training_all_parameter_audit import compare_files,run_json,shaped_manifest
 from qwen3_training_adamw_state_audit import compare as compare_moments

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('manifest','micro-binary','output-directory'):p.add_argument(f'--{n}',type=Path,required=True)
 p.add_argument('--precision',choices=('fp32','bf16'),required=True);p.add_argument('--model',default='qwen3-0.6b');p.add_argument('--context',type=int,default=32);p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--timeout-seconds',type=int,default=2400);a=p.parse_args()
 if a.output_directory.exists() and any(a.output_directory.iterdir()):p.error('output directory must be empty')
 a.output_directory.mkdir(parents=True,exist_ok=True);model,_=shaped_manifest(a.manifest,a.model,a.context,a.learning_rate)
 def base():return [str(a.micro_binary),'--config',model['config'],'--weights',model['weights'],'--tokens',model['training']['tokens'],'--device','hip','--learning-rate',str(a.learning_rate),'--warmup','0','--batch','1','--linear-precision',a.precision,'--bf16-weight-mirrors','true' if a.precision=='bf16' else 'false','--adamw-moment-precision','fp32']
 cp=a.output_directory/'control-parameters.safetensors';cm=a.output_directory/'control-moments.safetensors';cl=a.output_directory/'control-loss.json';rp=a.output_directory/'resumed-parameters.safetensors';rm=a.output_directory/'resumed-moments.safetensors';rl=a.output_directory/'resumed-loss.json';ck=a.output_directory/'step1.ckpt'
 control=run_json(base()+['--steps','3','--checkpoint-output',str(ck),'--checkpoint-after-steps','1','--all-parameters-output',str(cp),'--all-moments-output',str(cm),'--loss-trajectory-output',str(cl)],a.timeout_seconds)
 first={'final_global_step':1,'checkpoint_written':True}
 resumed=run_json(base()+['--steps','2','--resume-checkpoint',str(ck),'--all-parameters-output',str(rp),'--all-moments-output',str(rm),'--loss-trajectory-output',str(rl)],a.timeout_seconds)
 pr,params,_=compare_files(rp,cp,'parameter');mr,moments=compare_moments(rm,cm);lc=json.loads(cl.read_text())['losses'];lr=json.loads(rl.read_text())['losses'];loss_diffs=[abs(lc[i+1]-lr[i]) for i in range(2)]
 limits={'loss_max':1e-7,'parameter_max':1e-7,'moment_max':1e-5}
 gates={'losses_within_tolerance':max(loss_diffs)<=limits['loss_max'],'parameters_within_tolerance':params['maximum_absolute_difference']<=limits['parameter_max'],'moments_within_tolerance':moments['maximum_absolute_difference']<=limits['moment_max'],'steps':control['final_global_step']==resumed['final_global_step']==resumed['all_moment_step']==3 and first['final_global_step']==resumed['initial_global_step']==1,'resumed':resumed['checkpoint_resumed']}
 files=(cp,cm,rp,rm,ck);temporary=sum(x.stat().st_size for x in files)
 for x in files:x.unlink()
 summary={'schema_version':1,'record_type':'qwen3_training_checkpoint_resume_audit','status':'pass' if all(gates.values()) else 'resume_mismatch','model':model['name'],'precision':a.precision,'control_losses':lc,'resumed_losses':lr,'resumed_loss_differences':loss_diffs,'parameters':params,'moments':moments,'limits':limits,'bitwise_diagnostics':{'losses':max(loss_diffs)==0.0,'parameters':params['maximum_absolute_difference']==0.0,'moments':moments['maximum_absolute_difference']==0.0},'gates':gates,'checkpoint_format_version':2,'temporary_bytes':temporary,'temporary_files_removed':all(not x.exists() for x in files),'boundary':'shared step1 checkpoint; loss/parameter/moment tolerances isolate tied-embedding atomic scheduling across process restart; PyTorch parity is inherited from Experiment 385'}
 (a.output_directory/'raw.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in [*pr,*mr]));(a.output_directory/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');(a.output_directory/'workers.json').write_text(json.dumps({'control':control,'first':first,'resumed':resumed},indent=2,sort_keys=True)+'\n');print(json.dumps(summary,sort_keys=True));return 0 if summary['status']=='pass' else 2
if __name__=='__main__':raise SystemExit(main())
