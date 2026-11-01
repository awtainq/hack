"""Protect forecast data boundaries and the zero-day training correction."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import v66_calendar_model as model


class CalendarModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg=model.load_source()

    def test_forecast_features_and_schedule_do_not_use_held_out_labels(self):
        full=model.make_grid(self.reg,2023)
        hidden=model.make_grid(self.reg[self.reg.date.dt.year<2023],2023)
        columns=['date','year','dow']+model.NUMERIC+model.CATS+['era']
        pd.testing.assert_frame_equal(full[columns],hidden[columns])
        pd.testing.assert_series_equal(full.loc[full.year<2023,'y'],hidden.loc[hidden.year<2023,'y'])
        np.testing.assert_array_equal(model.closed_mask(full[full.year<2023],full[full.year==2023]),
                                      model.closed_mask(hidden[hidden.year<2023],hidden[hidden.year==2023]))

    def test_zero_targets_are_fitted_with_positive_weight(self):
        grid=model.make_grid(self.reg,2023);tr=grid[grid.year<2023]
        cfg=model.Config(iterations=10,depth=3)
        with patch.object(model.CatBoostRegressor,'fit',autospec=True) as fit:
            model.fit_model(tr,2023,cfg)
        labels=np.asarray(fit.call_args.args[2])
        weights=np.asarray(fit.call_args.kwargs['sample_weight'])
        self.assertGreater(np.sum((labels==0)&(weights>0)),0)
        self.assertNotIn('y',fit.call_args.args[1].columns)

    def test_one_model_prediction_survives_serialization(self):
        g=model.make_grid(self.reg,2022);tr=g[g.year<2022];te=g[g.year==2022]
        cfg=model.Config(iterations=10,depth=3)
        fitted=model.fit_model(tr,2022,cfg)
        pred=model.predict_log(fitted,tr,te,cfg)
        expected=np.where(model.closed_mask(tr,te),0.,np.maximum(0.,fitted.predict(model.feature_frame(te,cfg))))
        np.testing.assert_array_equal(pred,expected)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.joblib';joblib.dump(fitted,path)
            np.testing.assert_array_equal(pred,model.predict_log(joblib.load(path),tr,te,cfg))

    def test_only_raw_registration_source_is_read(self):
        original=pd.read_csv;seen=[]
        def guarded(path,*args,**kwargs):
            self.assertEqual(Path(path).resolve(),model.REG.resolve())
            seen.append(Path(path).resolve())
            return original(path,*args,**kwargs)
        with patch.object(model.pd,'read_csv',side_effect=guarded):
            model.load_source()
        self.assertEqual(seen,[model.REG.resolve()])


if __name__=='__main__':unittest.main()
