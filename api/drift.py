"""
Rapport de drift Evidently : compare les prédictions récentes
à la distribution de référence issue du training.
"""

import os
import tempfile

import pandas as pd
from evidently import Report
from evidently.metrics import ValueDrift, DriftedColumnsCount

from db import get_connection

# Baseline training : ~200 images/classe, confiance moyenne observée
REFERENCE_DATA = pd.DataFrame({
    "confidence": [0.88] * 209 + [0.91] * 210 + [0.85] * 207 + [0.87] * 209 + [0.89] * 210,
    "class_id": [0] * 209 + [1] * 210 + [2] * 207 + [3] * 209 + [4] * 210,
})

CLASS_MAP = {"Catfish": 0, "Goldfish": 1, "Mudfish": 2, "Mullet": 3, "Snakehead": 4}


def get_recent_predictions(limit: int = 200) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT predicted_class, confidence FROM predictions ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=["confidence", "class_id"])

    df = pd.DataFrame(rows)
    df["class_id"] = df["predicted_class"].map(CLASS_MAP).fillna(-1).astype(int)
    return df[["confidence", "class_id"]]


def _build_snapshot(current: pd.DataFrame):
    return Report([
        DriftedColumnsCount(),
        ValueDrift(column="confidence"),
        ValueDrift(column="class_id"),
    ]).run(reference_data=REFERENCE_DATA, current_data=current)


def generate_drift_report() -> dict:
    current = get_recent_predictions()

    if len(current) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Seulement {len(current)} prédictions disponibles. Minimum 10 requis.",
            "prediction_count": len(current),
        }

    snapshot = _build_snapshot(current)
    metrics = {m["metric_name"]: m["value"] for m in snapshot.dict()["metrics"]}

    drifted_count = metrics.get("DriftedColumnsCount(drift_share=0.5)", {})
    drift_detected = drifted_count.get("share", 0) >= 0.5

    return {
        "status": "drift_detected" if drift_detected else "ok",
        "drift_detected": drift_detected,
        "drifted_columns": drifted_count.get("count", 0),
        "total_columns": 2,
        "prediction_count": len(current),
        "avg_confidence": round(float(current["confidence"].mean()), 4),
        "class_distribution": get_recent_predictions(200)
            .merge(
                pd.DataFrame(list(CLASS_MAP.items()), columns=["predicted_class", "class_id"]),
                on="class_id", how="left"
            )["predicted_class"].value_counts().to_dict() if len(current) > 0 else {},
    }


def generate_drift_html() -> str:
    current = get_recent_predictions()
    # Si pas assez de données, on utilise la référence comme courante (rapport vide mais valide)
    current_data = current if len(current) >= 10 else REFERENCE_DATA.sample(50)

    snapshot = _build_snapshot(current_data)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        fname = f.name
    snapshot.save_html(fname)
    with open(fname) as hf:
        html = hf.read()
    os.unlink(fname)
    return html
