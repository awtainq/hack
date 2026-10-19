"""Standalone-model data boundaries, target integrity and model serialization."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import v65_standalone as model


class StandaloneModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg,cls.ks=model.load_sources()

    def test_only_raw_csvs_are_read(self):
        original=pd.read_csv
        allowed={model.REG.resolve(),model.KS.resolve()}
        seen=[]
        def guarded(path,*a,**kw):
            p=Path(path).resolve()
            self.assertIn(p,allowed)
            seen.append(p)
            return original(path,*a,**kw)
        with patch.object(model.pd,"read_csv",side_effect=guarded):
            model.load_sources()
        self.assertEqual(set(seen),allowed)

    def test_missing_intervening_seasons_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Missing intervening"):
            model.make_dataset(self.reg,self.ks,2025)

    def test_validation_labels_do_not_enter_features_or_schedule(self):
        year=2023
        full=model.make_dataset(self.reg,self.ks,year)
        r=self.reg[self.reg.date.dt.year<year]
        k=self.ks[self.ks.date.dt.year<year]
        hidden=model.make_dataset(r,k,year)
        pd.testing.assert_frame_equal(full[model.CALENDAR+model.LAGS],hidden[model.CALENDAR+model.LAGS])
        pd.testing.assert_frame_equal(full.loc[full.year<year,model.TARGETS],
                                      hidden.loc[hidden.year<year,model.TARGETS])
        np.testing.assert_array_equal(model.closed_mask(full[full.year<year],full[full.year==year]),
                                      model.closed_mask(hidden[hidden.year<year],hidden[hidden.year==year]))

    def test_auxiliary_labels_preserve_all_registrations(self):
        g=model.make_dataset(self.reg,self.ks,2022)
        np.testing.assert_array_equal(g.budget+g.contract,g.total)
        np.testing.assert_array_equal(g[["bak","mag","spo","other"]].sum(axis=1),g.total)

    def test_prediction_is_direct_first_model_output_and_survives_serialization(self):
        year=2017;g=model.make_dataset(self.reg,self.ks,year)
        tr,te=g[g.year<year],g[g.year==year].reset_index(drop=True)
        config=model.Config(mode="multi",depth=3,iterations=15)
        fitted=model.fit_model(tr,year,config)
        predicted=model.predict_log(fitted,te,config)
        raw=fitted.predict(te[model.features(config)])
        np.testing.assert_array_equal(predicted,np.maximum(0,raw[:,0]/config.total_scale))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"one_model.cbm"
            fitted.save_model(str(path));restored=CatBoostRegressor();restored.load_model(str(path))
            np.testing.assert_array_equal(predicted,model.predict_log(restored,te,config))


if __name__=="__main__":unittest.main()
