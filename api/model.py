import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from minio import Minio
from io import BytesIO

# ---------------------------
# Configuration du modèle
# ---------------------------
MODEL_FILENAME = "model_v1_1761836094.pt"
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
MINIO_BUCKET = "models"

# Classes du dataset
CLASSES = ["Catfish", "Goldfish", "Mudfish", "Mullet", "Snakehead"]

# ---------------------------
# Connexion à MinIO
# ---------------------------
minio_client = Minio(
    "minio:9000",
    access_key="admin-user",
    secret_key="admin-password",
    secure=False
)

# ---------------------------
# Chargement du modèle
# ---------------------------
def load_model():
    """
    Tente de charger le modèle depuis MinIO.
    Si non disponible, utilise le modèle local.
    """
    # 1️⃣ On vérifie d’abord le local
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"✅ Modèle trouvé localement : {LOCAL_MODEL_PATH}")
        model_bytes = open(LOCAL_MODEL_PATH, "rb").read()
    else:
        print("⚠️ Modèle local introuvable, tentative de récupération depuis MinIO...")
        try:
            response = minio_client.get_object(MINIO_BUCKET, MODEL_FILENAME)
            model_bytes = response.read()
            with open(LOCAL_MODEL_PATH, "wb") as f:
                f.write(model_bytes)
            print("✅ Modèle téléchargé depuis MinIO et sauvegardé localement.")
        except Exception as e:
            print(f"❌ Impossible de charger le modèle depuis MinIO : {e}")
            return DummyModel()  # Fallback

    try:
        loaded_obj = torch.load(BytesIO(model_bytes), map_location=torch.device("cpu"))
        if isinstance(loaded_obj, dict):
            print("⚙️ State_dict détecté, reconstruction de l'architecture ResNet18...")
            from torchvision import models
            
            # Créer l'architecture ResNet18 (même que dans train_model.py)
            model = models.resnet18(weights=None)  # pas de poids pré-entraînés
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(CLASSES))
            
            # Charger le state_dict
            model.load_state_dict(loaded_obj)
            print("✅ State_dict chargé dans le modèle ResNet18")
        elif hasattr(loaded_obj, "eval"):
            model = loaded_obj
            print("✅ Modèle complet chargé")
        else:
            raise RuntimeError(f"Format de modèle non reconnu : {type(loaded_obj)}")
        
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Erreur de chargement du modèle : {e}")
        import traceback
        traceback.print_exc()
        return DummyModel()


class DummyModel(nn.Module):
    """ Modèle factice (fallback) pour éviter les crashs sans vrai modèle """
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        batch = x.shape[0]
        return torch.zeros((batch, self.n_classes))


# Charger le modèle au démarrage
model = load_model()

# ---------------------------
# Préprocessing pour prédiction
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Fonction de prédiction
# ---------------------------
def predict(image: Image.Image):
    """
    Prend une image PIL, renvoie (label, confiance)
    """
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    return CLASSES[pred_idx], confidence
