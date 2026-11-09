from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core

class ArtifactContracts(unittest.TestCase):
    def test_source_and_artifact_digests(self):
        manifest = json.loads((ROOT / "results/manifest.json").read_text())
        for rel, digest in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), digest, rel)
        for rel, digest in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "results" / rel).read_bytes()).hexdigest(), digest, rel)

    def test_generator_reproduces_committed_inputs(self):
        self.assertEqual(core.generate(), json.loads((ROOT / "data/dataset.json").read_text()))

import copy
import numpy as np

class FareContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=core.generate(); cls.train=[r for r in cls.data["rows"] if r["split"]=="train"]
        cls.model=core.fit(cls.train,1)
        cls.row=next(r for r in cls.data["rows"] if r["split"]=="test")

    def test_safe_features_do_not_read_later_outcomes(self):
        altered=dict(self.row,paid_fare=999999,post_booking_fare=999999)
        self.assertEqual(core.predict(self.model,[self.row]),core.predict(self.model,[altered]))
        self.assertNotIn("post_booking_fare_UNSAFE",self.model["features"])

    def test_quote_expiry_is_end_exclusive(self):
        self.assertTrue(np.isfinite(core.quote(self.model,self.row,self.row["quoted_at"])))
        with self.assertRaises(ValueError): core.quote(self.model,self.row,self.row["expires_at"])
        with self.assertRaises(ValueError): core.quote(self.model,self.row,self.row["quoted_at"]-1)

    def test_unsafe_diagnostic_cannot_serve(self):
        leaked=core.fit(self.train,1,leaked=True)
        with self.assertRaises(ValueError): core.quote(leaked,self.row,self.row["quoted_at"])

    def test_future_feature_is_rejected(self):
        row=dict(self.row,feature_observed_at=self.row["quoted_at"]+1)
        with self.assertRaises(ValueError): core.vector(row)

    def test_nonfinite_or_bool_time_rejected(self):
        for value in [True,float("nan"),float("inf"),"1"]:
            with self.subTest(value=value),self.assertRaises(ValueError): core.quote(self.model,self.row,value)

    def test_model_artifact_reproduces_raw_predictions(self):
        model=json.loads((ROOT/"results/model.json").read_text())
        raw=json.loads((ROOT/"results/decisions.json").read_text())
        test=[r for r in self.data["rows"] if r["split"]=="test"]
        np.testing.assert_allclose(core.predict(model,test),[r["ready_prediction"] for r in raw],atol=1e-10,rtol=1e-12)

    def test_normalization_uses_only_fitted_rows(self):
        expected=np.array([core.vector(r) for r in self.train]).mean(0)
        np.testing.assert_allclose(self.model["mean"],expected)

    def test_split_boundaries_are_chronological_and_disjoint(self):
        days={s:{r["day"] for r in self.data["rows"] if r["split"]==s} for s in ["train","validation","test"]}
        self.assertLess(max(days["train"]),min(days["validation"]))
        self.assertLess(max(days["validation"]),min(days["test"]))
        with self.assertRaises(ValueError): core.vector(dict(self.row,route=True))
