import unittest
from development.temporal_selection_v1.event_retention_v1 import bracket_retention


class RetentionTests(unittest.TestCase):
    def test_unknown_is_not_pass(self):
        r = bracket_retention([0,84], {'release':[8,16], 'rim':None, 'result':None}, 146)
        self.assertEqual(r['status'], 'unresolved')
        self.assertFalse(r['training_use_approved'])

    def test_half_open_and_all_bracket_containment(self):
        r = bracket_retention([44,142], {'release':[50,56], 'rim':[70,75], 'result':[90,142]}, 146)
        self.assertEqual(r['status'], 'all_supplied_brackets_retained')
        self.assertFalse(r['independent_event_verification'])
        r = bracket_retention([44,142], {'release':[43,51], 'rim':[70,75], 'result':[90,143]}, 146)
        self.assertEqual(r['status'], 'evidence_outside_candidate')

    def test_invalid_brackets_rejected(self):
        for bounds in ([8,8], [-1,5], [0,147], [True,5]):
            with self.assertRaises(ValueError):
                bracket_retention([0,84], {'release':bounds, 'rim':None, 'result':None}, 146)
        with self.assertRaises(ValueError):
            bracket_retention([0,84], {'release':None}, 146)


if __name__ == '__main__':
    unittest.main()
