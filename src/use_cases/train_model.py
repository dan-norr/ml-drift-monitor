"""
use_cases/train_model.py — Fraud detection model training pipeline.

Orchestrates data loading, preprocessing, SMOTE, XGBoost training,
threshold optimisation, and artefact persistence.  All I/O is handled
by injected repository ports — the use case itself has no file system calls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..domain.entities import TrainingResult
from ..domain.ports import IDataRepository, IModelRepository

logger = logging.getLogger(__name__)


class TrainModelUseCase:
    def __init__(
        self,
        data_repo: IDataRepository,
        model_repo: IModelRepository,
        cfg: dict[str, Any],
    ) -> None:
        self._data = data_repo
        self._model = model_repo
        self._cfg = cfg

    def execute(self) -> TrainingResult:
        df = self._data.load_raw(self._cfg["data"]["raw_path"])
        logger.info("Dataset loaded: %d rows, fraud rate=%.4f%%", len(df), df["Class"].mean() * 100)

        X_train, X_test, y_train, y_test = self._split(df)

        if self._cfg["smote"]["enabled"]:
            X_train, y_train = self._apply_smote(X_train, y_train)

        model = self._build_model(smote_applied=self._cfg["smote"]["enabled"])
        logger.info("Training XGBoost …")
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        result = self._evaluate(model, X_test, y_test)
        if result.f1 < 0.85:
            logger.warning("F1=%.4f is below the 0.85 target.", result.f1)

        fi = dict(zip(X_test.columns.tolist(), model.feature_importances_.tolist()))
        fi_sorted = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))
        self._model.save(model, fi_sorted, result)

        ref = X_test.copy()
        ref["target"] = y_test.values
        ref["prediction"] = model.predict(X_test)
        ref["prediction_proba"] = model.predict_proba(X_test)[:, 1]
        self._data.save_reference(ref)

        logger.info("Training complete — F1=%.4f, ROC-AUC=%.4f", result.f1, result.roc_auc)
        return result

    # ── private helpers ──────────────────────────────────────────────────────

    def _split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        feature_cols = [c for c in df.columns if c != "Class"]
        X, y = df[feature_cols], df["Class"]
        m = self._cfg["model"]
        return train_test_split(
            X, y,
            test_size=m["test_size"],
            random_state=m["random_state"],
            stratify=y,
        )

    def _apply_smote(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        cfg = self._cfg["smote"]
        sm = SMOTE(random_state=cfg["random_state"], k_neighbors=cfg["k_neighbors"])
        X_res, y_res = sm.fit_resample(X_train, y_train)
        logger.info("After SMOTE: %d samples", len(X_res))
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name="Class")

    def _build_model(self, smote_applied: bool) -> XGBClassifier:
        m = self._cfg["model"]
        spw = 1 if smote_applied else m["scale_pos_weight"]
        return XGBClassifier(
            n_estimators=m["n_estimators"],
            max_depth=m["max_depth"],
            learning_rate=m["learning_rate"],
            scale_pos_weight=spw,
            random_state=m["random_state"],
            eval_metric="logloss",
            tree_method="hist",
        )

    def _evaluate(
        self, model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
    ) -> TrainingResult:
        y_prob = model.predict_proba(X_test)[:, 1]

        precision_c, recall_c, thresholds_pr = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_c, precision_c)
        roc_auc = roc_auc_score(y_test, y_prob)

        denom = precision_c[:-1] + recall_c[:-1]
        f1_scores = np.where(denom > 0, 2 * precision_c[:-1] * recall_c[:-1] / denom, 0.0)
        best_idx = int(np.argmax(f1_scores))
        threshold = float(thresholds_pr[best_idx])

        y_pred = (y_prob >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(
            "Threshold=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f  ROC-AUC=%.4f",
            threshold, f1_score(y_test, y_pred),
            report["1"]["precision"], report["1"]["recall"], roc_auc,
        )

        return TrainingResult(
            f1=float(f1_score(y_test, y_pred)),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=float(report["1"]["precision"]),
            recall=float(report["1"]["recall"]),
            threshold=threshold,
        )
