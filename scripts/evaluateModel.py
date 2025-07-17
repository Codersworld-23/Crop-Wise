import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.models as models # Explicitly import models
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import os
# import spacy # Uncomment if you need it for parse_folder_name

# --- Configuration (MUST match your training setup) ---
VALIDATION_DATA_DIR = "data/Validation"
MODEL_PATH = "models/crop_disease_resnet18.pth"
CROP_ENCODER_PATH = "models/crop_label_encoder.pkl"
DISEASE_ENCODER_PATH = "models/disease_label_encoder.pkl"

BATCH_SIZE = 32
IMG_SIZE = 224

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("Warning: GPU not detected. Evaluation will run on CPU, which is slower.")

# --- NLP-Enhanced Folder Parsing (MUST be identical to your training script) ---
# Your original parse_folder_name relied on NLP and specific keywords.
# For evaluation, we need to ensure it extracts labels EXACTLY as they were
# extracted during training for the encoders.

def parse_folder_name(folder_name):
    folder_name = folder_name.lower().strip()
    
    # --- IMPORTANT: THIS IS THE CRITICAL PART TO DEBUG ---
    # Based on your training code, the `parse_folder_name` likely determined
    # crop and disease from folder names like 'thirps on cotton' or 'Apple___scab'.
    # You MUST ensure this function processes the names exactly as it did during training.

    crop = "unknown_crop" # Default value if parsing logic below doesn't find a clear crop
    disease = "unknown_disease" # Default value if parsing logic below doesn't find a clear disease

    # Example parsing logic - adjust this heavily based on your actual folder names
    # and what your *original* training script's parse_folder_name did.
    # It seems your folder names are like "disease on crop" or "crop___disease"
    
    if " on " in folder_name:
        parts = folder_name.split(" on ")
        disease = parts[0].strip()
        crop = parts[1].strip()
    elif " in " in folder_name:
        parts = folder_name.split(" in ")
        disease = parts[0].strip()
        crop = parts[1].strip()
    elif "___" in folder_name: # Common in PlantVillage, etc.
        parts = folder_name.split("___")
        crop = parts[0].strip()
        disease = parts[1].strip()
    elif "_" in folder_name and folder_name.count('_') >= 1 and not folder_name.startswith("unknown_"): # Try to split if single underscore and not already marked unknown
        # This is a heuristic, be careful with it.
        # If your diseases themselves have underscores (e.g., 'early_blight'), this needs more complex logic.
        # Simplest assumption: first word is crop, rest is disease.
        potential_parts = folder_name.split('_', 1) # Split only on first underscore
        if len(potential_parts) == 2:
            crop = potential_parts[0].strip()
            disease = potential_parts[1].strip()
        else: # Handle cases like 'healthy' or 'scab' being a top-level folder
            crop = "unknown_crop" # Default or handle specifically
            disease = folder_name
    else: # Fallback for simple names, e.g., 'healthy', 'cotton'
        # If your dataset has root folders like just "healthy" or just "cotton",
        # you need a lookup table or a more sophisticated NLP approach.
        # For 'thirps on cotton', this 'else' block would likely fail to parse it correctly
        # if the above 'on'/'in'/'___' logic didn't catch it.
        crop = "unknown_crop" # Indicate that parsing failed for crop
        disease = folder_name # Assume the whole thing is disease for now


    # Your training code had this specific mapping:
    if disease == "healthy":
        disease = "none"
        
    return crop, disease

# ---------- Custom Dataset ---------- #
class CropDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.transform = transform

        # Load encoders that were saved during training
        try:
            self.crop_encoder = joblib.load(CROP_ENCODER_PATH)
            self.disease_encoder = joblib.load(DISEASE_ENCODER_PATH)
            print("Successfully loaded pre-fitted encoders.")
        except FileNotFoundError:
            print(f"Error: Encoders not found at {CROP_ENCODER_PATH} or {DISEASE_ENCODER_PATH}.")
            print("Please ensure you run your training script first to create and save them.")
            raise # Re-raise error as we cannot proceed without encoders

        self.crop_labels = []
        self.disease_labels = []
        self.samples_with_valid_labels = [] # To store (img, crop_label_idx, disease_label_idx)

        # Process each sample
        for i, (path, class_idx) in enumerate(self.dataset.samples):
            class_name = self.dataset.classes[class_idx]
            crop, disease = parse_folder_name(class_name)
            
            # Debugging print
            # print(f"Processing folder: {class_name} -> Parsed: Crop='{crop}', Disease='{disease}'")

            try:
                crop_encoded = self.crop_encoder.transform([crop])[0]
                # Special handling for 'healthy' if it's mapped to 'none' in the encoder
                disease_encoded = self.disease_encoder.transform([disease])[0]

                self.samples_with_valid_labels.append((i, crop_encoded, disease_encoded))

            except ValueError as e:
                # This means 'crop' or 'disease' extracted by parse_folder_name was not in the encoder's classes
                print(f"Warning: Skipping sample '{path}'. Parsed labels '{crop}', '{disease}' not found in encoders. Error: {e}")
                # Do not add this sample to self.crop_labels/disease_labels

    def __len__(self):
        return len(self.samples_with_valid_labels) # Only return count of successfully processed samples

    def __getitem__(self, idx):
        original_idx, crop_label, disease_label = self.samples_with_valid_labels[idx]
        img, _ = self.dataset[original_idx] # Get image from original ImageFolder dataset
        return img, crop_label, disease_label

# ---------- Custom Fully Connected Layer (MUST be identical to your training script) ---
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

# --- Evaluation Logic ---
def evaluate():
    # Transforms (MUST be identical to your training script)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    try:
        val_dataset = CropDiseaseDataset(VALIDATION_DATA_DIR, transform=transform)
        # Check if val_dataset is empty after filtering
        if len(val_dataset) == 0:
            print(f"Error: No valid samples found in {VALIDATION_DATA_DIR} after parsing and encoding. Check folder names and encoder content.")
            return # Exit if no samples to evaluate
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True) # No shuffle for evaluation
        
    except Exception as e:
        print(f"Failed to load validation dataset or encoders: {e}")
        return # Exit if dataset loading fails

    # Initialize model architecture (identical to how it was saved after training)
    model = None # Initialize model to None for safety

    try:
        model = models.resnet18(weights=None) # Load ResNet18 architecture
        
        # Get number of classes from loaded encoders
        num_crops = len(val_dataset.crop_encoder.classes_)
        num_diseases = len(val_dataset.disease_encoder.classes_)
        
        in_features = model.fc.in_features
        model.fc = CustomFC(in_features, num_crops, num_diseases) # Replace with your custom head
        
        # Load your trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode (VERY IMPORTANT for correct results)
    except Exception as e:
        print(f"Failed to load or initialize model: {e}")
        return # Exit if model loading/init fails

    # Evaluation loop
    correct_crop_predictions = 0
    correct_disease_predictions = 0
    total_samples = 0

    print(f"\n🚀 Starting evaluation on {VALIDATION_DATA_DIR}...\n")
    with torch.no_grad(): # Disable gradient calculations for faster evaluation
        for inputs, crop_labels, disease_labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device, non_blocking=True)
            crop_labels = crop_labels.to(device, non_blocking=True)
            disease_labels = disease_labels.to(device, non_blocking=True)

            outputs = model(inputs)
            crop_output, disease_output = outputs["crop"], outputs["disease"]
            
            _, predicted_crops = torch.max(crop_output, 1)
            _, predicted_diseases = torch.max(disease_output, 1)
            
            total_samples += inputs.size(0)
            correct_crop_predictions += (predicted_crops == crop_labels).sum().item()
            correct_disease_predictions += (predicted_diseases == disease_labels).sum().item()

    crop_accuracy = (correct_crop_predictions / total_samples) * 100
    disease_accuracy = (correct_disease_predictions / total_samples) * 100

    print(f"\n--- Evaluation Results ---")
    print(f"Total Samples Evaluated: {total_samples}")
    print(f"Final Crop Prediction Accuracy: {crop_accuracy:.2f}%")
    print(f"Final Disease Prediction Accuracy: {disease_accuracy:.2f}%")
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate()