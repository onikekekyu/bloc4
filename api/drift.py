"""
Génère un rapport de drift Evidently en comparant les prédictions récentes
à la distribution de référence issue du training.
"""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

from db import get_connection

# Distribution de référence (training set : ~200 images par classe, confiance ~0.87 moyenne)
REFERENCE_DATA = pd.DataFrame({
    "predicted_class": (
        ["Catfish"] * 209 + ["Goldfish"] * 210 + ["Mudfish"] * 207 +
        ["Mullet"] * 209 + ["Snakehead"] * 210
    ),
    "confidence": (
        [0.88] * 209 + [0.91] * 210 + [0.85] * 207 +
        [0.87] * 209 + [0.89] * 210
    ),
})
REFERENCE_DATA["class_id"] = REFERENCE_DATA["predicted_class"].map(
    {"Catfish": 0, "Goldfish": 1, "Mudfish": 2, "Mullet": 3, "Snakehead": 4}
)


def get_recent_predictions(limit: int = 200) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT predicted_class, confidence FROM predictions ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=["predicted_class", "confidence", "class_id"])

    df = pd.DataFrame(rows)
    df["class_id"] = df["predicted_class"].map(
        {"Catfish": 0, "Goldfish": 1, "Mudfish": 2, "Mullet": 3, "Snakehead": 4}
    )
    return df


def generate_drift_report() -> dict:
    current = get_recent_predictions()

    if len(current) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Seulement {len(current)} prédictions disponibles. Minimum 10 requis.",
            "prediction_count": len(current),
        }

    ref = REFERENCE_DATA[["confidence", "class_id"]].copy()
    cur = current[["confidence", "class_id"]].copy()

    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="confidence"),
        ColumnDriftMetric(column_name="class_id"),
    ])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    metrics = result.get("metrics", [])
    dataset_drift = next(
        (m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {}
    )
    drift_detected = dataset_drift.get("result", {}).get("dataset_drift", False)
    drift_share = dataset_drift.get("result", {}).get("share_of_drifted_columns", 0)

    return {
        "status": "drift_detected" if drift_detected else "ok",
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "prediction_count": len(current),
        "avg_confidence": round(float(current["confidence"].mean()), 4),
        "class_distribution": current["predicted_class"].value_counts().to_dict(),
    }


def generate_drift_html() -> str:
    current = get_recent_predictions()

    ref = REFERENCE_DATA[["confidence", "class_id"]].copy()
    cur = current[["confidence", "class_id"]].copy() if len(current) >= 10 else ref.copy()

    report = Report(metrics=[
        DataDriftPreset(),
    ])
    report.run(reference_data=ref, current_data=cur)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        report.save_html(f.name)
        f.flush()
        with open(f.name) as html_file:
            html = html_file.read()
    os.unlink(f.name)
    return html
