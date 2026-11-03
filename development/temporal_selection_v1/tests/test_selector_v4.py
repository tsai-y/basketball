import unittest
from development.temporal_selection_v1.selector_v4 import refine_analysis, refine_regions, all_segments


def base(n,regions,cuts=()):
    return {'per_frame':[{'frame_index':i,'histogram_distance':.02,'gray_difference':.01,
        'flash_boundary_suppressed':False} for i in range(n)],
        'gradual_change_regions':[{'start_frame':a,'end_frame_exclusive':b} for a,b in regions],
        'hard_cut_candidates':list(cuts),'raw_hard_cut_candidates':list(cuts),'exposure_flash_candidates':[]}


class BoundaryTests(unittest.TestCase):
    def test_isolated_cut_replaces_padded_exclusion(self):
        b=base(100,[(40,55)])
        b['per_frame'][48].update(histogram_distance=.6,gray_difference=.25)
        cuts,regions,d=refine_regions(b)
        self.assertEqual(cuts,[48]);self.assertEqual(regions,[])
        self.assertFalse(d[0]['semantic_graphic_verified'])

    def test_multiple_strong_edges_do_not_collapse_to_one_cut(self):
        b=base(100,[(20,60)])
        for i in (27,53):b['per_frame'][i].update(histogram_distance=.6,gray_difference=.25)
        cuts,regions,_=refine_regions(b)
        self.assertEqual(cuts,[]);self.assertEqual(regions,[(27,53)])

    def test_return_frame_not_excluded_and_right_censoring_explicit(self):
        b=base(100,[(20,60)])
        for i in range(27,54):b['per_frame'][i].update(histogram_distance=.2,gray_difference=.08)
        self.assertEqual(refine_regions(b)[1],[(27,53)])
        b=base(100,[(90,100)])
        for i in (97,98,99):b['per_frame'][i].update(histogram_distance=.2,gray_difference=.08)
        _,regions,d=refine_regions(b)
        self.assertEqual(regions,[(97,100)]);self.assertTrue(d[0]['right_censored'])

    def test_active_gap_can_join_but_stable_gap_never_merges_by_length(self):
        b=base(100,[(20,40),(45,65)])
        for i in range(27,59):b['per_frame'][i].update(histogram_distance=.2,gray_difference=.08)
        self.assertEqual(len(refine_regions(b)[2]),1)
        for i in range(40,45):b['per_frame'][i].update(histogram_distance=.02,gray_difference=.01)
        self.assertEqual(len(refine_regions(b)[2]),2)

    def test_no_evidence_retains_coarse_and_flash_does_not_localize_cut(self):
        b=base(100,[(20,40)])
        b['per_frame'][30].update(histogram_distance=.9,gray_difference=.5,flash_boundary_suppressed=True)
        c,r,d=refine_regions(b)
        self.assertEqual(c,[]);self.assertEqual(r,[(20,40)])
        self.assertEqual(d[0]['kind'],'unresolved_keep_coarse')

    def test_existing_cut_on_region_right_edge_is_localized(self):
        b=base(100,[(3,10)],cuts=[10])
        b['per_frame'][10].update(histogram_distance=.6,gray_difference=.25)
        self.assertEqual(refine_regions(b)[:2],([10],[]))

    def test_no_target_selection_and_all_short_scenes_logged(self):
        b=base(100,[],cuts=[5,95]);r=refine_analysis(b)
        self.assertFalse(r['target_event_selected']);self.assertFalse(r['automatic_training_use_enabled'])
        self.assertEqual([x['length'] for x in r['all_segments_including_short']],[5,90,5])
        self.assertEqual(sum(x['length'] for x in all_segments(100,[5,95],[(40,60)])),100)


if __name__=='__main__':unittest.main()
