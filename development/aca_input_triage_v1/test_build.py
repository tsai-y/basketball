import copy
import unittest
from .build import FOLDER, REVIEW, evidence_names, read, validate


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.obs=read(FOLDER/'observations.json');self.cases=read(REVIEW/'review_template.json')['cases']

    def test_exact_24_validation_cases(self):
        before=copy.deepcopy(self.cases);validate(self.obs,self.cases);self.assertEqual(before,self.cases)

    def test_duplicate_or_missing_review_id_rejected(self):
        for bad in (self.obs['cases'][:-1],self.obs['cases'][:-1]+[self.obs['cases'][0]]):
            with self.assertRaises(AssertionError):validate({**self.obs,'cases':bad},self.cases)

    def test_test_split_rejected(self):
        self.cases[0]['split']='test'
        with self.assertRaises(AssertionError):validate(self.obs,self.cases)

    def test_three_uncertain_only_and_control_included(self):
        uncertain=[c['review_id'] for c in self.obs['cases'] if c['category']=='uncertain']
        self.assertEqual(uncertain,['R11','R14','R21'])
        control=next(c for c in self.cases if c['review_id']=='R14')
        self.assertEqual(control['prediction'],control['label'])

    def test_evidence_names_are_safe_and_deduplicated(self):
        for row in self.obs['cases']:
            names=evidence_names(row);self.assertEqual(len(names),len(set(names)))
            self.assertIn('aca50_01.png',names);self.assertIn('key_comparison.png',names)
        self.obs['cases'][0]['extra_viewed_files']=['../test.png']
        with self.assertRaises(AssertionError):validate(self.obs,self.cases)

    def test_clear_is_not_all_pass_and_known_user_answers_preserved(self):
        self.assertEqual([c['review_id'] for c in self.obs['cases'] if c['category']=='clear'],['R07','R12','R17'])
        for rid in ('R02','R24'):
            row=next(c for c in self.obs['cases'] if c['review_id']==rid)
            self.assertNotIn('question',row)
        self.assertEqual(next(c for c in self.cases if c['review_id']=='R24')['human_review']['target_attempt_index'],2)


if __name__=='__main__':unittest.main()
