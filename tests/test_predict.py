"""
Tests d'intégration pour l'endpoint /predict de l'API.

Usage:
    pytest tests/test_predict.py
    python tests/test_predict.py --image /chemin/vers/poisson.jpg
"""

import argparse
import sys
from pathlib import Path
import requests


API_URL = "http://localhost:8000"


def send_prediction(image_path: str, url: str = f"{API_URL}/predict") -> dict:
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=10)
    resp.raise_for_status()
    return resp.json()


def test_api_root():
    resp = requests.get(f"{API_URL}/", timeout=5)
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_api_metrics():
    resp = requests.get(f"{API_URL}/metrics", timeout=5)
    assert resp.status_code == 200
    assert b"fish_predictions_total" in resp.content


def test_predict_with_sample_image():
    candidates = list(Path("minio_data/dataset-fish").rglob("*.jpg"))[:1]
    if not candidates:
        import pytest
        pytest.skip("Aucune image de test disponible localement")

    result = send_prediction(str(candidates[0]))
    assert "prediction" in result
    assert "confidence" in result
    assert result["confidence"] > 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", "-i", required=False)
    p.add_argument("--url", "-u", default=f"{API_URL}/predict")
    args = p.parse_args()

    if not args.image:
        candidates = list(Path("minio_data/dataset-fish").rglob("*.jpg"))
        if not candidates:
            print("Aucune image trouvee. Passez --image /chemin/vers/image.jpg")
            sys.exit(1)
        args.image = str(candidates[0])
        print(f"Image utilisee : {args.image}")

    try:
        result = send_prediction(args.image, args.url)
        print(f"Prediction : {result['prediction']}")
        print(f"Confiance  : {result['confidence']}%")
    except requests.exceptions.ConnectionError:
        print(f"Impossible de se connecter a {args.url}")
        sys.exit(1)
