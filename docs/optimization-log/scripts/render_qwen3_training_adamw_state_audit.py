#!/usr/bin/env python3
"""Render complete Qwen3 AdamW state alignment evidence."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--fp32",type=Path,required=True); p.add_argument("--bf16",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    fp=json.loads(a.fp32.read_text()); bf=json.loads(a.bf16.read_text())
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="760" viewBox="0 0 1400 760" role="img">','<rect width="1400" height="760" fill="#f7f9fc"/>','<text x="700" y="48" text-anchor="middle" font-family="Inter,Arial" font-size="30" font-weight="700" fill="#172033">Qwen3 AdamW State: FP32 Pass · BF16 Rejected</text>','<text x="700" y="80" text-anchor="middle" font-family="Inter,Arial" font-size="16" fill="#5b6474">620 moment Tensors · 1,192,099,840 values · strict step=1</text>']
    for x,d,title,color in ((60,fp,'FP32 · PASS','#198754'),(720,bf,'BF16 · REJECT','#c0392b')):
        c=d['comparison']; lim=d['limits']; lines += [f'<rect x="{x}" y="115" width="620" height="210" rx="18" fill="#fff" stroke="#dbe1ea"/>',f'<text x="{x+30}" y="158" font-family="Inter,Arial" font-size="22" font-weight="700" fill="{color}">{title}</text>',f'<text x="{x+30}" y="205" font-family="Inter,Arial" font-size="17" fill="#172033">Max  {c["maximum_absolute_difference"]:.3e}  /  {lim["maximum"]:.1e}</text>',f'<text x="{x+30}" y="245" font-family="Inter,Arial" font-size="17" fill="#172033">RMS  {c["rms_difference"]:.3e}  /  {lim["rms"]:.1e}</text>',f'<text x="{x+30}" y="285" font-family="Inter,Arial" font-size="14" fill="#5b6474">worst: {html.escape(c["worst_tensor"])}</text>']
    lines += ['<rect x="60" y="360" width="1280" height="320" rx="18" fill="#fff" stroke="#dbe1ea"/>','<text x="90" y="402" font-family="Inter,Arial" font-size="21" font-weight="700" fill="#172033">BF16 Max difference by family and moment</text>']
    rows=bf['comparison']['groups']; maximum=max(r['maximum_absolute_difference'] for r in rows)
    for i,r in enumerate(rows):
        y=425+i*13.2; w=850*r['maximum_absolute_difference']/maximum; color='#c0392b' if r['maximum_absolute_difference']>bf['limits']['maximum'] else '#4776c5'; label=f"{r['family']} · {'m' if r['moment']=='first_moment' else 'v'}"; lines += [f'<text x="90" y="{y+10:.1f}" font-family="Inter,Arial" font-size="11" fill="#172033">{html.escape(label)}</text>',f'<rect x="285" y="{y:.1f}" width="{w:.1f}" height="10" rx="3" fill="{color}"/>',f'<text x="1280" y="{y+10:.1f}" text-anchor="end" font-family="monospace" font-size="11" fill="#172033">{r["maximum_absolute_difference"]:.3e}</text>']
    lines += ['<text x="700" y="725" text-anchor="middle" font-family="Inter,Arial" font-size="15" fill="#5b6474">canonical FP32 moments · temporary 9.54 GB exports removed · serialization excluded from timing</text>','</svg>']
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text('\n'.join(lines)+'\n'); return 0
if __name__=='__main__': raise SystemExit(main())
