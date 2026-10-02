"""Meaningful offline checks against frozen public evaluation records."""
import unittest,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from openperov.evaluation import COMPONENT_FIELDS,V43_FIELD,scientific_mastery_v43,aggregate_component_rows,score_answer_exact,anchored_pro_score,score_external_mcq,human_family_agreement
from openperov.statistics import huber,paired_bootstrap,study_paired_bootstrap

def readl(path):return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.components=readl(ROOT/'results/psm_bench/components.jsonl')
    def test_entire_frozen_board(self):
        result=aggregate_component_rows(self.components)
        expected=json.loads((ROOT/'results/psm_bench/frozen_aggregate_comparison.json').read_text(encoding='utf-8'))
        self.assertEqual(len(self.components),11200)
        self.assertEqual(len(result['systems']),14)
        for r in expected:
            value=result['systems'][r['model']]
            self.assertEqual(value['items'],800)
            self.assertEqual(round(value['huber'],4),r['huber'])
            self.assertEqual(round(value['arithmetic'],4),r['arithmetic'])
    def test_human_family_agreement(self):
        result=human_family_agreement(self.components,readl(ROOT/'results/human_assessment/ratings.jsonl'))
        self.assertEqual(result['groups'],16)
        self.assertEqual(round(result['pearson_r'],3),0.707)
        self.assertEqual(round(result['spearman_rho'],3),0.729)
    def test_cap_floor_and_zero_mad(self):
        r=dict(zip(COMPONENT_FIELDS,[100,1,1,1,1,4]))
        self.assertEqual(scientific_mastery_v43(r),80.0)
        self.assertEqual(huber([100,100,100,0])['location'],75.0)
        with self.assertRaises(ValueError):scientific_mastery_v43({**r,'missing_key_count':1.5})
    def test_missing_source_support_is_rejected(self):
        ref=readl(ROOT/'benchmarks/psm_bench/references.jsonl')[0]
        with self.assertRaisesRegex(ValueError,'source-support'):
            score_answer_exact({'benchmark_id':ref['benchmark_id'],'answer':'Example'},ref,None)
    def test_prediction_integrity_and_alignment(self):
        components={(r['benchmark_id'],r['model']) for r in self.components}
        predictions=readl(ROOT/'results/psm_bench/predictions.jsonl')
        self.assertEqual(len(predictions),11200)
        self.assertEqual({(r['benchmark_id'],r['model']) for r in predictions},components)
        for r in predictions:self.assertEqual(hashlib.sha256(r['answer'].encode()).hexdigest(),r['answer_sha256'])
    def test_human_counts(self):
        rows=readl(ROOT/'results/human_assessment/ratings.jsonl')
        self.assertEqual(len(rows),3200);self.assertEqual(len({r['response_id'] for r in rows}),3200)
        self.assertEqual(len({r['benchmark_id'] for r in rows}),800)
        self.assertEqual(sum(r[k] is not None for r in rows for k in ['human_C1','human_C2','human_C3','human_C4']),12800)
        self.assertFalse(any('reviewer_id' in r or 'candidate_label' in r for r in rows))
    def test_s20_anchored_scores_and_separation(self):
        summary=json.loads((ROOT/'results/expert_comparison/summary.json').read_text(encoding='utf-8'))
        allrows=[]
        for condition in summary['conditions']:
            name=condition['condition'];rows=readl(ROOT/f'results/expert_comparison/{name}_paired_scores.jsonl');allrows+=rows
            self.assertEqual(len(rows),40);self.assertEqual(len({r['study_id'] for r in rows}),10)
            result=anchored_pro_score(condition['frozen_flash_score'],rows)
            self.assertEqual(round(result['anchored_pro_score'],4),condition['published_scores']['OpenPerov Pro'])
            answers=readl(ROOT/f'results/expert_comparison/{name}_answers.jsonl')
            self.assertEqual(len(answers),120);self.assertEqual(sum(r['system']=='Expert + AI' for r in answers),40)
        with self.assertRaises(ValueError):anchored_pro_score(0,allrows)
        with self.assertRaises(ValueError):study_paired_bootstrap(allrows,40)
    def test_external_adapter_is_exact_match(self):
        q=[{'correct_option':'A'} for _ in range(49)]
        p=[{'source_index':i,'predicted_option':'A' if i<=48 else 'B'} for i in range(1,50)]
        self.assertEqual(score_external_mcq(q,p)['correct'],48)
        with self.assertRaises(ValueError):score_external_mcq(q*4,p)
        with self.assertRaises(ValueError):score_external_mcq(q,p[:-1]+[p[0]])
    def test_paired_bootstrap_pairing(self):
        result=paired_bootstrap([5,6,7],[2,3,4],40,123)
        self.assertEqual(result['difference'],3)
        for bound in result['ci95']:self.assertAlmostEqual(bound,3)

if __name__=='__main__':unittest.main()

