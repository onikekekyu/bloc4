"""
Script pour uploader le modèle local vers MinIO
À exécuter APRÈS avoir démarré MinIO avec docker-compose
"""
from minio import Minio
import os

# Configuration
MINIO_ENDPOINT = "localhost:9000"  # localhost car on exécute hors Docker
MINIO_ACCESS_KEY = "admin-user"
MINIO_SECRET_KEY = "admin-password"
BUCKET_NAME = "models"
MODEL_FILE = "model_v1_1761836094.pt"

def main():
    print("🔌 Connexion à MinIO...")
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    # Créer le bucket s'il n'existe pas
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"✅ Bucket '{BUCKET_NAME}' créé")
    else:
        print(f"✅ Bucket '{BUCKET_NAME}' existe déjà")
    
    # Vérifier que le fichier local existe
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Erreur : Le fichier '{MODEL_FILE}' n'existe pas dans le répertoire courant")
        return
    
    # Upload du modèle
    print(f"📤 Upload de '{MODEL_FILE}' vers MinIO...")
    client.fput_object(
        BUCKET_NAME,
        MODEL_FILE,
        MODEL_FILE,
    )
    print(f"✅ Modèle uploadé avec succès vers MinIO !")
    print(f"   Bucket: {BUCKET_NAME}")
    print(f"   Objet: {MODEL_FILE}")
    
    # Vérifier
    objects = list(client.list_objects(BUCKET_NAME))
    print(f"\n📦 Contenu du bucket '{BUCKET_NAME}':")
    for obj in objects:
        print(f"   - {obj.object_name} ({obj.size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
