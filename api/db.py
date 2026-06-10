import os
import pymysql

MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_USER = "root"
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DB = "mlops"


def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor,
    )


def init_predictions_table():
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    predicted_class VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        conn.close()
        print("Table predictions prête")
    except Exception as e:
        print(f"Impossible d'initialiser la table predictions : {e}")


def log_prediction(predicted_class: str, confidence: float):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO predictions (predicted_class, confidence) VALUES (%s, %s)",
                (predicted_class, confidence)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erreur log prédiction : {e}")
