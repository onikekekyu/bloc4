"""
Script de réentraînement automatisé.
Peut être déclenché manuellement, par la CI/CD, ou sur détection de drift.

Usage:
    python retrain/retrain.py
    python retrain/retrain.py --minio-endpoint localhost:9000 --mysql-host localhost
"""

import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Déclenche le pipeline de réentraînement")
    p.add_argument("--minio-endpoint", default="minio:9000")
    p.add_argument("--minio-key", default="admin-user")
    p.add_argument("--minio-secret", default="admin-password")
    p.add_argument("--mysql-host", default="mysql")
    p.add_argument("--mysql-password", default="root")
    p.add_argument("--force", action="store_true", help="Forcer même sans drift détecté")
    return p.parse_args()


def check_drift_threshold(mysql_host: str, mysql_password: str) -> bool:
    """
    Vérifie si le drift dépasse le seuil justifiant un réentraînement.
    Retourne True si le réentraînement est nécessaire.
    """
    try:
        import pymysql
        conn = pymysql.connect(
            host=mysql_host, user="root",
            password=mysql_password, database="mlops"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(confidence) as avg_conf,
                   COUNT(*) as total
            FROM predictions
            WHERE created_at >= NOW() - INTERVAL 7 DAY
        """)
        row = cursor.fetchone()
        conn.close()

        if row and row[1] and row[1] >= 50:
            avg_confidence = row[0]
            print(f"Confiance moyenne (7 derniers jours) : {avg_confidence:.2%} sur {row[1]} prédictions")
            if avg_confidence < 0.70:
                print("Drift détecté : confiance moyenne < 70%, réentraînement recommandé")
                return True
            print("Pas de drift significatif détecté")
            return False

        print("Pas assez de prédictions récentes pour évaluer le drift (< 50)")
        return False

    except Exception as e:
        print(f"Impossible de vérifier le drift : {e}")
        return False


def run_pipeline(args):
    env = os.environ.copy()
    env.update({
        "MINIO_ENDPOINT": args.minio_endpoint,
        "MINIO_ACCESS_KEY": args.minio_key,
        "MINIO_SECRET_KEY": args.minio_secret,
        "MYSQL_HOST": args.mysql_host,
        "MYSQL_PASSWORD": args.mysql_password,
    })

    print("Étape 1/2 : extraction des métadonnées depuis MinIO → MySQL")
    result = subprocess.run(
        [sys.executable, "src/extraction_creation_sql.py"],
        env=env, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Erreur extraction : {result.stderr}")
        sys.exit(1)

    print("Étape 2/2 : entraînement du modèle")
    result = subprocess.run(
        [sys.executable, "src/train_model.py"],
        env=env, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Erreur training : {result.stderr}")
        sys.exit(1)

    print(f"Réentraînement terminé à {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    args = parse_args()

    should_retrain = args.force or check_drift_threshold(args.mysql_host, args.mysql_password)

    if not should_retrain:
        print("Réentraînement non nécessaire. Utilisez --force pour forcer.")
        sys.exit(0)

    run_pipeline(args)


if __name__ == "__main__":
    main()
