"""Composition leakage and soft aggregate reconciliation checks."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import v67_volume_prior as model


class VolumePriorTests(unittest.TestCase):
    def test_held_out_labels_do_not_change_degree_shares(self):
        r=model.load_registrations()
        _,dates,full=model.fit_shares(r,2023)
        _,dates_hidden,hidden=model.fit_shares(r[r.date.dt.year<2023],2023)
        pd.testing.assert_series_equal(dates.date,dates_hidden.date)
        np.testing.assert_array_equal(full,hidden)
        np.testing.assert_allclose(full.sum(axis=1),1.)

    def test_soft_reconciliation_preserves_closure_and_unconstrained_group(self):
        d=pd.DataFrame({'bachelor':[100.,0.,200.],'master_miep':[20.,0.,30.],'other':[5.,0.,10.]})
        priors={'relative_uncertainty':.15,'max_prior_weight':.75,
                'totals':{'bachelor':400.,'master_miep':40.}}
        result,summary=model.reconcile(d,priors,dict.fromkeys(model.GROUPS,.2))
        self.assertTrue(300<result.bachelor.sum()<400)
        self.assertTrue(40<result.master_miep.sum()<50)
        np.testing.assert_array_equal(result.other,d.other)
        np.testing.assert_array_equal(result.sum(axis=1)==0,d.sum(axis=1)==0)
        self.assertTrue((summary.prior_weight<=.75).all())

    def test_matching_priors_do_not_change_the_forecast(self):
        d=pd.DataFrame({'bachelor':[100.,200.],'master_miep':[20.,30.],'other':[5.,10.]})
        priors={'relative_uncertainty':.15,'max_prior_weight':.75,
                'totals':{'bachelor':300.,'master_miep':50.}}
        result,_=model.reconcile(d,priors,dict.fromkeys(model.GROUPS,.2))
        pd.testing.assert_frame_equal(result,d)


if __name__=='__main__':unittest.main()
