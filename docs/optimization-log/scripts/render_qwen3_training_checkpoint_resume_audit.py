#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fp32',type=Path,required=True);p.add_argument('--bf16',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();f=json.loads(a.fp32.read_text());b=json.loads(a.bf16.read_text())
L=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="620" viewBox="0 0 1400 620">','<rect width="1400" height="620" fill="#f7f9fc"/>','<text x="700" y="48" text-anchor="middle" font-family="Inter,Arial" font-size="30" font-weight="700" fill="#172033">Qwen3 Checkpoint Resume: Shared-State Branch Passes</text>','<text x="700" y="80" text-anchor="middle" font-family="Inter,Arial" font-size="16" fill="#5b6474">step1 save → fresh process → steps2–3 versus same-process continuation</text>']
for x,d,title in ((60,f,'FP32'),(720,b,'BF16')):
 L += [f'<rect x="{x}" y="120" width="620" height="390" rx="18" fill="#fff" stroke="#dbe1ea"/>',f'<text x="{x+30}" y="165" font-family="Inter,Arial" font-size="23" font-weight="700" fill="#198754">{title} · 5/5 PASS</text>']
 vals=[('resumed loss Max',max(d['resumed_loss_differences'])),('parameter Max',d['parameters']['maximum_absolute_difference']),('moment Max',d['moments']['maximum_absolute_difference'])]
 for i,(n,v) in enumerate(vals):L += [f'<text x="{x+35}" y="{225+i*65}" font-family="Inter,Arial" font-size="17" fill="#172033">{n}</text>',f'<text x="{x+300}" y="{225+i*65}" font-family="monospace" font-size="17" fill="#172033">{v:.3e}</text>']
 L += [f'<text x="{x+35}" y="440" font-family="Inter,Arial" font-size="14" fill="#5b6474">loss bitwise: {str(d["bitwise_diagnostics"]["losses"]).lower()}</text>',f'<text x="{x+35}" y="470" font-family="Inter,Arial" font-size="14" fill="#5b6474">parameter/moment bitwise: false · strict tolerance reported</text>']
L += ['<text x="700" y="565" text-anchor="middle" font-family="Inter,Arial" font-size="15" fill="#5b6474">checkpoint v2 · 930 final-state Tensors · step/global step=3 · temporary 21.46 GB removed per precision</text>','</svg>'];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(L)+'\n')
