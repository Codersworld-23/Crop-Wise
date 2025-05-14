import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import spacy
import multiprocessing

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Paths
DATA_DIR = "data/Train"
MODEL_PATH = "models/crop_disease_resnet18.pth"
CROP_ENCODER_PATH = "models/crop_label_encoder.pkl"
DISEASE_ENCODER_PATH = "models/disease_label_encoder.pkl"

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
IMG_SIZE = 224
LEARNING_RATE = 0.001

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("Warning: GPU not detected. Training will run on CPU, which is slower.")

# ---------- NLP-Enhanced Folder Parsing ---------- #
def parse_folder_name(folder_name):
    folder_name = folder_name.lower().strip()
    doc = nlp(folder_name)
    common_crops = {"cotton", "maize", "rice", "sugarcane", "wheat"}
    crop, disease = "unknown", "unknown"

    if " on " in folder_name:
        disease, crop = folder_name.split(" on ")
    elif " in " in folder_name:
        disease, crop = folder_name.split(" in ")
    else:
        tokens = [token.text for token in doc]
        for token in tokens:
            if token in common_crops:
                crop = token
                disease = folder_name.replace(crop, "").strip()
                break
        if crop == "unknown":
            disease = " ".join(tokens[:-1]) if len(tokens) > 1 else "unknown"
            crop = tokens[-1] if tokens else "unknown"

    if disease == "healthy":
        disease = "none"
    return crop, disease

# ---------- Custom Dataset ---------- #
class CropDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.transform = transform

        # Parse class labels into crop/disease
        crops = []
        diseases = []
        self.crop_labels = []
        self.disease_labels = []

        for path, class_idx in self.dataset.samples:
            class_name = self.dataset.classes[class_idx]
            crop, disease = parse_folder_name(class_name)
            crops.append(crop)
            diseases.append(disease)

        # Fit encoders
        self.crop_encoder = LabelEncoder()
        self.disease_encoder = LabelEncoder()
        self.crop_encoder.fit(crops)
        self.disease_encoder.fit(diseases)

        # Encode labels per sample
        for path, class_idx in self.dataset.samples:
            class_name = self.dataset.classes[class_idx]
            crop, disease = parse_folder_name(class_name)
            self.crop_labels.append(self.crop_encoder.transform([crop])[0])
            self.disease_labels.append(self.disease_encoder.transform([disease])[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.crop_labels[idx], self.disease_labels[idx]

# ---------- Custom Fully Connected Layer ---------- #
class CustomFC(nn.Module):
    def __init__(self, in_features, num_crops, num_diseases):
        super(CustomFC, self).__init__()
        self.crop = nn.Linear(in_features, num_crops)
        self.disease = nn.Linear(in_features, num_diseases)

    def forward(self, x):
        return {
            "crop": self.crop(x),
            "disease": self.disease(x)
        }

# ---------- Main Training Logic ---------- #
def main():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset and encoders
    dataset = CropDiseaseDataset(DATA_DIR, transform=transform)
    joblib.dump(dataset.crop_encoder, CROP_ENCODER_PATH)
    joblib.dump(dataset.disease_encoder, DISEASE_ENCODER_PATH)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Optimized data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Load model
    model = models.resnet18(weights='IMAGENET1K_V1')  
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    num_crops = len(dataset.crop_encoder.classes_)
    num_diseases = len(dataset.disease_encoder.classes_)
    model.fc = CustomFC(in_features, num_crops, num_diseases)  # Use custom FC layer

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Training
    print("\n🚀 Training started...\n")
    for epoch in range(NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        correct_crops = 0
        correct_diseases = 0

        for inputs, crop_labels, disease_labels in tqdm(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            crop_labels = crop_labels.to(device, non_blocking=True)
            disease_labels = disease_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            crop_output, disease_output = outputs["crop"], outputs["disease"]

            crop_loss = criterion(crop_output, crop_labels)
            disease_loss = criterion(disease_output, disease_labels)
            loss = crop_loss + disease_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_crops += (crop_output.argmax(1) == crop_labels).sum().item()
            correct_diseases += (disease_output.argmax(1) == disease_labels).sum().item()

        crop_acc = correct_crops / len(train_set)
        disease_acc = correct_diseases / len(train_set)
        print(f"📉 Loss: {running_loss:.4f} | 🌾 Crop Acc: {crop_acc:.4f} | 🦠 Disease Acc: {disease_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Encoders saved to: {CROP_ENCODER_PATH}, {DISEASE_ENCODER_PATH}")

# ---------- Entry Point ---------- #
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows support
    main()