#!/usr/bin/env python3
"""Offline evaluation. The default reproduces the released frozen scores."""
from pathlib import Path
import argparse,json,hashlib,statistics,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from openperov.evaluation import aggregate_component_rows,anchored_pro_score,score_answer_exact,score_external_mcq,human_family_agreement

def readl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def reproduce(root):
    psm=aggregate_component_rows(readl(root/'results/psm_bench/components.jsonl'))
    expected=json.loads((root/'results/psm_bench/frozen_aggregate_comparison.json').read_text(encoding='utf-8'))
    for row in expected:
        observed=psm['systems'][row['model']]
        if round(observed['huber'],4)!=row['huber'] or round(observed['arithmetic'],4)!=row['arithmetic']:
            raise ValueError('Published PSM aggregate mismatch: '+row['model'])
    s20=[]
    saved=json.loads((root/'results/expert_comparison/summary.json').read_text(encoding='utf-8'))
    for condition in saved['conditions']:
        name=condition['condition'];base=readl(root/f'results/expert_comparison/{name}_frozen_baseline_scores.jsonl')
        flash=statistics.mean(r['score_0_100'] for r in base if r['system']=='OpenPerov Flash')
        value=anchored_pro_score(flash,readl(root/f'results/expert_comparison/{name}_paired_scores.jsonl'))
        if round(value['anchored_pro_score'],4)!=condition['published_scores']['OpenPerov Pro']:
            raise ValueError('Published S20 aggregate mismatch: '+name)
        s20.append(value)
    human=human_family_agreement(readl(root/'results/psm_bench/components.jsonl'),readl(root/'results/human_assessment/ratings.jsonl'))
    if round(human['pearson_r'],3)!=0.707 or round(human['spearman_rho'],3)!=0.729:raise ValueError('Human agreement mismatch')
    return {'human_agreement':{k:human[k] for k in ('groups','pearson_r','spearman_rho')},'status':'PASS','scope':'Frozen component scores and archived S20 judgments; no inference or new judging','psm':psm,'expert_comparison':s20}

def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command')
    p=subs.add_parser('reproduce');p.add_argument('--root',type=Path,default=ROOT)
    p=subs.add_parser('components');p.add_argument('--predictions',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--source-support',type=Path,required=True,help='Private/user-supplied original support JSONL; not distributed');p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('mcq');p.add_argument('--questions',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--allow-different-upstream-hash',action='store_true',help='Explicitly allow a reformatted/modified input and report its actual hash')
    args=parser.parse_args()
    if args.command in (None,'reproduce'):
        result=reproduce(getattr(args,'root',ROOT))
    elif args.command=='components':
        refs={str(r.get('benchmark_id') or r.get('id')):r for r in readl(args.references)}
        supports={str(r.get('benchmark_id') or r.get('id')):r for r in readl(args.source_support)}
        predictions=readl(args.predictions);rows=[]
        for r in predictions:
            bid=str(r.get('benchmark_id') or r.get('id'))
            if bid not in refs or bid not in supports: raise ValueError('Missing reference/support for '+bid)
            rows.append(score_answer_exact(r,refs[bid],supports[bid]))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        result=aggregate_component_rows(rows)
    else:
        raw=args.questions.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        expected='cf0c043eaa79234cad5ce011d4f8966b6118a364fbfc7927f72557584953bc27'
        if digest!=expected and not args.allow_different_upstream_hash:
            raise ValueError('Input hash differs from the recorded 49-question upstream file. Use the explicit override only for a reviewed alternate encoding/version.')
        result=score_external_mcq(json.loads(raw.decode('utf-8-sig')),readl(args.predictions));result['upstream_sha256']=digest;result['matches_recorded_upstream']=digest==expected
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
