import unittest

from development.temporal_selection_v1.event_review_v5 import excluded_spans,validate_assessment,QUESTIONS


class V5EventReviewTests(unittest.TestCase):
    def assessment(self):
        return {'sample_id':'A','status':'acceptable','questions':{k:True for k in QUESTIONS},
                'source_phase_evidence_brackets':{'preparation':[40,46],'release':[46,52],'rim':[62,70],'result':[72,96]},
                'human_ground_truth':False,'training_use_approved':False}

    def row(self):return {'sample_id':'A','frame_count':146,'candidate_interval':[37,142]}

    def test_half_open_excluded_spans_and_no_candidate(self):
        self.assertEqual(excluded_spans(146,(37,142)),[[0,37],[142,146]])
        self.assertEqual(excluded_spans(146,(0,146)),[])
        self.assertEqual(excluded_spans(146,None),[])

    def test_complete_retained_AI_evidence_does_not_promote_to_training(self):
        a=self.assessment();validate_assessment(a,self.row())
        self.assertFalse(a['training_use_approved']);self.assertFalse(a['human_ground_truth'])

    def test_unknown_result_or_question_cannot_pass(self):
        a=self.assessment();a['source_phase_evidence_brackets']['result']=None
        with self.assertRaises(ValueError):validate_assessment(a,self.row())
        a=self.assessment();a['questions']['same_labeled_event_preserved']=None
        with self.assertRaises(ValueError):validate_assessment(a,self.row())

    def test_outside_release_rejected_but_explicit_negative_review_allowed(self):
        a=self.assessment();a['source_phase_evidence_brackets']['release']=[30,40]
        with self.assertRaises(ValueError):validate_assessment(a,self.row())
        a['status']='unacceptable';a['questions']['preparation_and_release_preserved']=False
        validate_assessment(a,self.row())

    def test_no_interval_or_human_truth_cannot_be_acceptable(self):
        row=self.row();row['candidate_interval']=None
        with self.assertRaises(ValueError):validate_assessment(self.assessment(),row)
        a=self.assessment();a['human_ground_truth']=True
        with self.assertRaises(ValueError):validate_assessment(a,self.row())

    def test_invalid_bracket_bool_and_unfounded_rejection(self):
        a=self.assessment();a['source_phase_evidence_brackets']['release']=[True,52]
        with self.assertRaises(ValueError):validate_assessment(a,self.row())
        a=self.assessment();a['status']='unacceptable'
        with self.assertRaises(ValueError):validate_assessment(a,self.row())


if __name__=='__main__':unittest.main()
