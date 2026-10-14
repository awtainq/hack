import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from src.preprocess import FEATURE_COLS
from src.metrics import rmsle, mae

class AdmissionsForecaster:
    def __init__(self, use_ensemble=True, random_state=42):
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        self.lgb_model = None
        self.hgb_model = None
        self.categorical_features = ['ob_osnova_cat']
        
    def fit(self, X, y):
        """
        Fit model on features X and targets y.
        y is expected to be raw non-negative counts.
        Model trains internally on log1p(y).
        """
        y_log = np.log1p(np.asarray(y))
        X_df = X[FEATURE_COLS].copy()
        
        # Primary LightGBM model
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            random_state=self.random_state,
            verbosity=-1
        )
        self.lgb_model.fit(X_df, y_log)
        
        # Secondary model for ensemble (HistGradientBoosting)
        if self.use_ensemble:
            X_encoded = X_df.copy()
            # Convert categorical to integer codes for HistGradientBoosting
            X_encoded['ob_osnova_cat'] = X_encoded['ob_osnova_cat'].cat.codes
            self.hgb_model = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=10,
                categorical_features=['ob_osnova_cat'],
                random_state=self.random_state
            )
            self.hgb_model.fit(X_encoded, y_log)
        return self

    def predict(self, X):
        """
        Predict application counts for features X.
        Returns non-negative counts.
        """
        X_df = X[FEATURE_COLS].copy()
        pred_log_lgb = self.lgb_model.predict(X_df)
        
        if self.use_ensemble and self.hgb_model is not None:
            X_encoded = X_df.copy()
            X_encoded['ob_osnova_cat'] = X_encoded['ob_osnova_cat'].cat.codes
            pred_log_hgb = self.hgb_model.predict(X_encoded)
            # 70% LightGBM + 30% HistGradientBoosting
            pred_log = 0.70 * pred_log_lgb + 0.30 * pred_log_hgb
        else:
            pred_log = pred_log_lgb
            
        pred = np.clip(np.expm1(pred_log), 0.0, None)
        
        # Post-processing: on Sundays, admissions commission is closed
        is_sunday = (X['is_sunday'] == 1).values if 'is_sunday' in X.columns else (X['day_of_week'] == 6).values
        pred[is_sunday] = 0.0
        
        return pred

    def get_feature_importances(self):
        if self.lgb_model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': self.lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

