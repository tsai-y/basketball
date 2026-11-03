from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from development.temporal_selection_v1.batch_review_v1 import make_focus, render_review


class BatchReviewTests(unittest.TestCase):
    def fixture(self, root):
        run=root/'run';(run/'contacts').mkdir(parents=True)
        canvas=np.zeros((1056,1280,3),np.uint8)
        for i in range(16):canvas[(i//4)*264:(i//4+1)*264,(i%4)*320:(i%4+1)*320]=i
        cv2.imwrite(str(run/'contacts/S001.png'),canvas)
        (run/'manifest.json').write_text(json.dumps({'videos':[{'sample_id':'S001','contact_pages':['contacts/S001.png']}]}))
        return run

    def test_focus_exact_native_tiles_and_never_overwrites(self):
        with TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root)
            with patch('development.temporal_selection_v1.batch_review_v1.ARTIFACTS',root):
                pages=make_focus(run,['S001'])
                img=cv2.imread(str(run/pages[0]))
                self.assertEqual(img.shape,(264,1920,3))
                for j,i in enumerate((0,2,5,8,12,15)):
                    self.assertTrue(np.all(img[:,j*320:(j+1)*320]==i))
                with self.assertRaises(ValueError):make_focus(run,['S001'])

    def test_foreign_or_unknown_focus_rejected(self):
        with TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root)
            with patch('development.temporal_selection_v1.batch_review_v1.ARTIFACTS',root/'elsewhere'):
                with self.assertRaises(ValueError):make_focus(run,['S001'])
            with patch('development.temporal_selection_v1.batch_review_v1.ARTIFACTS',root):
                with self.assertRaises(ValueError):make_focus(run,['S999'])
                self.assertFalse((run/'focus').exists())

    def test_review_mismatched_manifest_cannot_write_approval(self):
        with TemporaryDirectory() as d:
            root=Path(d);run=self.fixture(root);evidence=run/'findings.json'
            evidence.write_text(json.dumps({'schema':'AI_fixed_v4_train100_review_v1','manifest_sha256':'wrong'}))
            with patch('development.temporal_selection_v1.batch_review_v1.ARTIFACTS',root):
                with self.assertRaises(ValueError):render_review(run,evidence)
            self.assertFalse((run/'AI_review.json').exists())


if __name__=='__main__':unittest.main()
