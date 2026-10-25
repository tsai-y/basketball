"""Optional roles must not change accepted detection annotations or infer possession."""
from copy import deepcopy
import unittest

from development.yolo_audit_v1.role_metadata import (
    ROLE_PROTOCOL, add_role_metadata, validate_role_metadata,
)
from development.yolo_audit_v1 import target_label_pilot as pilot


def fixture():
    return {'schema':'yolo_target_label_review_v1','pilot_id':'roles',
        'protocol':pilot.PROTOCOL,'classes':pilot.CLASS_NAMES,
        'images':[{'image_id':'L001','image_sha256':'pixels','status':'reviewed',
            'scene':'court_play','annotation_complete':True,'notes':'Accepted object annotations',
            'annotations':[
                {'id':'A1','class_id':0,'xyxy':[10,10,30,100],'source_proposal_id':None},
                {'id':'A2','class_id':1,'xyxy':[25,40,30,45],'source_proposal_id':None},
                {'id':'A3','class_id':2,'xyxy':[50,20,65,26],'source_proposal_id':None}],
            'proposals':[],'proposal_states':{},
            'class_visibility':{'active_player':'present','ball':'present','rim':'present'}}]}


class RoleMetadataTests(unittest.TestCase):
    def test_legacy_missing_role_is_unreviewed_and_still_complete(self):
        data=fixture()
        self.assertEqual(validate_role_metadata(data),{'unreviewed':1,'yes':0,'no':0,'uncertain':0})
        self.assertEqual(pilot.validate_review(data,data)['training_ready_ids'],['L001'])

    def test_migration_is_non_mutating_idempotent_and_does_not_infer_holder(self):
        base=fixture()
        before=deepcopy(base)
        result=add_role_metadata(base)
        self.assertEqual(base,before)
        self.assertIsNone(result['images'][0]['annotations'][0]['holding_ball'])
        self.assertEqual(result['role_metadata_protocol'],ROLE_PROTOCOL)
        self.assertEqual(add_role_metadata(result),result)
        stripped=deepcopy(result)
        stripped.pop('role_metadata_protocol')
        stripped['images'][0]['annotations'][0].pop('holding_ball')
        self.assertEqual(stripped,before)
        self.assertEqual(pilot.validate_review(base,result)['training_ready_ids'],['L001'])

    def test_existing_roles_preserved_without_guessing_or_resetting_objects(self):
        for value in ('yes','no','uncertain',None):
            with self.subTest(value=value):
                base=fixture()
                base['images'][0]['annotations'][0]['holding_ball']=value
                result=add_role_metadata(base)
                self.assertEqual(result['images'][0]['annotations'][0]['holding_ball'],value)
                self.assertEqual(pilot.validate_review(fixture(),result)['training_ready_images'],1)

    def test_invalid_values_fail_in_both_migration_and_full_review_validation(self):
        for value in ('holder','',True,False,0,1,[],{}):
            with self.subTest(value=value):
                data=fixture()
                data['images'][0]['annotations'][0]['holding_ball']=value
                with self.assertRaisesRegex(ValueError,'holding_ball'):
                    add_role_metadata(data)
                with self.assertRaisesRegex(ValueError,'holding_ball'):
                    pilot.validate_review(fixture(),data)

    def test_ball_and_rim_cannot_have_the_player_attribute_even_if_null(self):
        for index in (1,2):
            data=fixture()
            data['images'][0]['annotations'][index]['holding_ball']=None
            with self.subTest(index=index), self.assertRaisesRegex(ValueError,'only allowed'):
                pilot.validate_review(fixture(),data)

    def test_unknown_role_protocol_rejected(self):
        data=fixture()
        data['role_metadata_protocol']='dribble-possession-v2'
        with self.assertRaisesRegex(ValueError,'protocol'):
            pilot.validate_review(fixture(),data)

    def test_null_no_uncertain_are_distinct_in_summary(self):
        data=fixture()
        player=data['images'][0]['annotations'][0]
        for i,value in enumerate((None,'yes','no','uncertain')):
            a=deepcopy(player)
            a['id']=f'P{i}'
            a['holding_ball']=value
            data['images'][0]['annotations'].append(a)
        self.assertEqual(validate_role_metadata(data),{'unreviewed':2,'yes':1,'no':1,'uncertain':1})


if __name__=='__main__':
    unittest.main()
