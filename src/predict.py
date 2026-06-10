# ===============================================
# Script : predict.py
# Objectif : Faire des prédictions avec le modèle entraîné
# Auteur : Kirsten
# ===============================================

import io
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from minio import Minio
import pymysql

# ============================================================
# 1️⃣ Configuration
# ============================================================

# MinIO configuration
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "admin-user"
MINIO_SECRET_KEY = "admin-password"
BUCKET_NAME = "dataset-fish"
MODEL_BUCKET = "models"
MODEL_PATH = "model_v1_1761836094.pt"  # Modèle entraîné récemment (84.21% val acc)

# MySQL configuration
MYSQL_HOST = "mysql"
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_DB = "mlops"

# Classes de poissons (dans l'ordre du training)
CLASSES = ['Catfish', 'Gold Fish', 'Mudfish', 'Mullet', 'Snakehead']

# ============================================================
# 2️⃣ Connexion MinIO
# ============================================================
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# ============================================================
# 3️⃣ Téléchargement du modèle depuis MinIO
# ============================================================
print("📥 Téléchargement du modèle depuis MinIO...")
minio_client.fget_object(MODEL_BUCKET, MODEL_PATH, MODEL_PATH)
print(f"✅ Modèle '{MODEL_PATH}' téléchargé.")

# ============================================================
# 4️⃣ Chargement du modèle
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"✅ Modèle chargé et prêt sur {device}")

# ============================================================
# 5️⃣ Fonction de prédiction
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes):
    """Prédit la classe d'une image à partir de ses bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return CLASSES[predicted.item()], confidence.item()

# ============================================================
# 6️⃣ Test sur quelques images depuis MySQL
# ============================================================
print("\n🔍 Test de prédiction sur des images de la base de données...")

# Connexion MySQL
conn = pymysql.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DB
)
cursor = conn.cursor()

# Récupérer 10 images aléatoires du SET DE TEST (pas train!)
query = "SELECT id, species_label, file_name, url_s3 FROM fish_data WHERE split = 'test' ORDER BY RAND() LIMIT 10"
cursor.execute(query)
results = cursor.fetchall()

print("\n📊 Résultats des prédictions :\n")
print(f"{'ID':<5} {'Vrai Label':<15} {'Prédit':<15} {'Confiance':<10} {'Fichier':<30}")
print("=" * 85)

correct = 0
for row in results:
    img_id, true_label, file_name, url_s3 = row
    
    # Construire le chemin MinIO depuis l'URL
    # url_s3 = "http://localhost:9000/dataset-fish/train/Catfish/img_001.jpg"
    object_path = "/".join(url_s3.split("/")[4:])  # train/Catfish/img_001.jpg
    
    # Télécharger l'image depuis MinIO
    try:
        response = minio_client.get_object(BUCKET_NAME, object_path)
        image_bytes = response.read()
        response.close()
        
        # Faire la prédiction
        predicted_label, confidence = predict_image(image_bytes)
        
        # Vérifier si correct
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{img_id:<5} {true_label:<15} {predicted_label:<15} {confidence*100:<9.2f}% {file_name:<30} {status}")
        
    except Exception as e:
        print(f"{img_id:<5} {true_label:<15} ERROR: {str(e)[:40]}")

cursor.close()
conn.close()

accuracy = (correct / len(results)) * 100
print("=" * 85)
print(f"\n🎯 Précision sur cet échantillon : {accuracy:.2f}% ({correct}/{len(results)})")
print("\n✅ Test de prédiction terminé !")
