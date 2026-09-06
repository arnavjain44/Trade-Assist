import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

# Attempt to import LightGBM or XGBoost
HAS_LIGHTGBM = False
HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass


class TradeSignalClassifier:
    """
    Model-family neutral classifier for Phase 3 experiments.
    Supports Logistic Regression, Random Forest, and LightGBM / XGBoost.
    
    Guarantees:
    - Train-only feature scaling (StandardScaler for Logistic Regression).
    - Model-family neutral class-weight translation (scale_pos_weight vs class_weight={0:1, 1:w}).
    - Out-Of-Fold (OOF) purged chronological walk-forward predictions for calibration fitting.
    - Calibrated probability output (Isotonic or Sigmoid/Platt).
    """

    FEATURE_COLS = [
        "rsi",
        "obv",
        "bollinger_position",
        "macd",
        "macd_signal",
        "macd_diff",
        "price_vs_vwap",
        "price_vs_ema5",
        "direction",
    ]

    def __init__(
        self,
        model_family: str = "lightgbm",
        pos_weight: float = 1.0,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None,
    ):
        """
        model_family: 'logistic_regression', 'random_forest', or 'lightgbm' / 'xgboost'
        pos_weight: Weight assigned to positive class (TARGET_HIT)
        feature_cols: Optional list of feature column names (defaults to FEATURE_COLS)
        """
        self.model_family = model_family.lower()
        self.pos_weight = float(pos_weight)
        self.random_state = random_state
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(self.FEATURE_COLS)

        self.scaler: Optional[StandardScaler] = None
        self.model: Any = None
        self.calibrator: Any = None
        self.calibration_method: str = "none"

        self._build_model()

    def _build_model(self):
        """Instantiates the underlying model with neutral class weighting."""
        if self.model_family in ["logistic_regression", "lr"]:
            self.model_family = "logistic_regression"
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                class_weight={0: 1.0, 1: self.pos_weight},
                random_state=self.random_state,
                max_iter=1000
            )

        elif self.model_family in ["random_forest", "rf"]:
            self.model_family = "random_forest"
            self.scaler = None
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight={0: 1.0, 1: self.pos_weight},
                random_state=self.random_state,
                n_jobs=-1
            )

        elif self.model_family in ["lightgbm", "lgbm", "boosting"]:
            if HAS_LIGHTGBM:
                self.model_family = "lightgbm"
                self.scaler = None
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                    scale_pos_weight=self.pos_weight,
                    random_state=self.random_state,
                    verbose=-1,
                    n_jobs=-1
                )
            elif HAS_XGBOOST:
                self.model_family = "xgboost"
                self.scaler = None
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                    scale_pos_weight=self.pos_weight,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=-1
                )
            else:
                raise RuntimeError(
                    "Neither LightGBM nor XGBoost is installed in this Python environment. "
                    "Cannot instantiate gradient boosting model family."
                )
        elif self.model_family == "xgboost":
            if HAS_XGBOOST:
                self.scaler = None
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=6,
                    scale_pos_weight=self.pos_weight,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=-1
                )
            else:
                raise RuntimeError("XGBoost is not installed.")
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fits scaler (if LR) and underlying model on Training data.
        """
        X_mat = X[self.feature_cols].values
        if self.scaler is not None:
            X_mat = self.scaler.fit_transform(X_mat)

        self.model.fit(X_mat, y)
        return self

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts raw uncalibrated positive class probabilities P(label=1).
        """
        X_mat = X[self.feature_cols].values
        if self.scaler is not None:
            X_mat = self.scaler.transform(X_mat)

        probs = self.model.predict_proba(X_mat)
        if probs.shape[1] > 1:
            return probs[:, 1]
        return probs[:, 0]

    def fit_oof_purged_predictions(
        self,
        df_train: pd.DataFrame,
        target_col: str = "label",
        n_splits: int = 4,
        horizon_minutes: int = 240
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates Out-Of-Fold (OOF) predictions across chronological walk-forward folds
        within the Training set, applying 240m purging between folds.

        Returns (valid_oof_indices, oof_raw_probabilities).
        """
        df_sorted = df_train.sort_values("timestamp").reset_index(drop=True)
        unique_dates = df_sorted["timestamp"].dt.date.unique()
        unique_dates = np.sort(unique_dates)

        if len(unique_dates) < n_splits + 1:
            n_splits = max(2, len(unique_dates) - 1)

        date_chunks = np.array_split(unique_dates, n_splits + 1)

        oof_indices = []
        oof_probs = []

        for i in range(1, len(date_chunks)):
            val_dates = date_chunks[i]
            train_dates = np.concatenate(date_chunks[:i])

            val_start_ts = pd.Timestamp(val_dates[0])
            if val_start_ts.tzinfo is None and df_sorted["timestamp"].dt.tz is not None:
                val_start_ts = val_start_ts.tz_localize(df_sorted["timestamp"].dt.tz)

            # Purge train observations within horizon_minutes prior to val_start_ts
            purge_cutoff = val_start_ts - pd.Timedelta(minutes=horizon_minutes)

            train_mask = (df_sorted["timestamp"].dt.date.isin(train_dates)) & (df_sorted["timestamp"] <= purge_cutoff)
            val_mask = df_sorted["timestamp"].dt.date.isin(val_dates)

            df_fold_train = df_sorted[train_mask]
            df_fold_val = df_sorted[val_mask]

            if df_fold_train.empty or df_fold_val.empty or df_fold_train[target_col].nunique() < 2:
                continue

            fold_clf = TradeSignalClassifier(
                model_family=self.model_family,
                pos_weight=self.pos_weight,
                random_state=self.random_state,
                feature_cols=self.feature_cols,
            )
            fold_clf.fit(df_fold_train[self.feature_cols], df_fold_train[target_col].values)

            val_probs = fold_clf.predict_proba_raw(df_fold_val[self.feature_cols])
            oof_indices.extend(df_fold_val.index.tolist())
            oof_probs.extend(val_probs.tolist())

        return np.array(oof_indices), np.array(oof_probs)

    def fit_calibrator(
        self,
        raw_probs: np.ndarray,
        y_true: np.ndarray,
        method: str = "isotonic"
    ):
        """
        Fits probability calibrator on OOF or validation raw probabilities.
        method: 'none', 'isotonic', or 'sigmoid' (Platt scaling).
        """
        self.calibration_method = method.lower()

        if self.calibration_method == "none":
            self.calibrator = None

        elif self.calibration_method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(raw_probs, y_true)
            self.calibrator = cal

        elif self.calibration_method in ["sigmoid", "platt"]:
            self.calibration_method = "sigmoid"
            cal = LogisticRegression(C=1.0, max_iter=1000)
            cal.fit(raw_probs.reshape(-1, 1), y_true)
            self.calibrator = cal

        else:
            raise ValueError(f"Unsupported calibration method: {method}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts calibrated (or raw if calibration_method='none') probabilities.
        """
        raw_p = self.predict_proba_raw(X)

        if self.calibrator is None or self.calibration_method == "none":
            return np.clip(raw_p, 0.0, 1.0)

        if self.calibration_method == "isotonic":
            calib_p = self.calibrator.transform(raw_p)
            return np.clip(calib_p, 0.0, 1.0)

        elif self.calibration_method == "sigmoid":
            probs = self.calibrator.predict_proba(raw_p.reshape(-1, 1))
            if probs.shape[1] > 1:
                return np.clip(probs[:, 1], 0.0, 1.0)
            return np.clip(probs[:, 0], 0.0, 1.0)

        return np.clip(raw_p, 0.0, 1.0)
