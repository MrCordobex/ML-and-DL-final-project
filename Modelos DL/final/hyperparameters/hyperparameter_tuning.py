import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
import optuna
from optuna.trial import TrialState
from ultralytics.data.augment import classify_augmentations, classify_transforms
from tqdm import tqdm

# --- Constants & Configuration ---
SEED = 42
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DATA_PATH = "../../Datos/images/SPLIT_MULTICLASE"  # Relative path as per notebook
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# --- Data Loading & Transforms ---

# Augmentation pipeline matches notebook
train_pipeline = classify_augmentations(
    size=IMG_SIZE,
    mean=MEAN,
    std=STD,
    scale=(0.5, 1.0),
    hflip=0.5,
    vflip=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.3,
)

val_test_pipeline = classify_transforms(
    size=IMG_SIZE,
    mean=MEAN,
    std=STD,
    interpolation="BILINEAR"
)

class SubsetConTransform(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)

def get_data_loaders(fold_idx, train_indices, val_indices, full_dataset, batch_size):
    train_subset = SubsetConTransform(full_dataset, train_indices, train_pipeline)
    val_subset = SubsetConTransform(full_dataset, val_indices, val_test_pipeline)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader

# --- Model Definition ---

def get_model(model_name, num_classes, trial):
    # Retrieve hyperparameters from trial
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    
    if model_name == "resnet18":
        # Add dropout suggestion
        dropout_rate = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze initial layers
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = model.fc.in_features
        # Add Dropout to classifier
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        model = model.to(DEVICE)
        
        optimizer = optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': lr_backbone},
            {'params': model.fc.parameters(), 'lr': lr_head}
        ], weight_decay=weight_decay)
        
    elif model_name == "densenet121":
        # Add dropout suggestion for DenseNet too
        dropout_rate = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Freeze initial layers
        for param in model.features.parameters():
            param.requires_grad = False
        # Unfreeze specific blocks
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
            
        num_ftrs = model.classifier.in_features
        # Add Dropout to classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        model = model.to(DEVICE)
        
        optimizer = optim.AdamW([
            {'params': model.features.denseblock4.parameters(), 'lr': lr_backbone},
            {'params': model.features.norm5.parameters(), 'lr': lr_backbone},
            {'params': model.classifier.parameters(), 'lr': lr_head}
        ], weight_decay=weight_decay)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model, optimizer

# --- Training & Evaluation ---

def train_one_epoch(model, optimizer, loader, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    # Kappa usually computed on validation, but keeping return signature similar
    return epoch_loss, epoch_f1

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_kappa = cohen_kappa_score(all_labels, all_preds)
    
    return val_loss, val_f1, val_kappa

# --- Objective Function ---

def create_objective(model_name, full_dataset, n_epochs=10):
    targets = full_dataset.targets
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Pre-calculate folds to be consistent across trials
    folds = list(skf.split(range(len(targets)), targets))
    
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        
        fold_f1_scores = []
        
        # We will run 5-fold CV
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"Trial {trial.number}: Fold {fold_idx+1}/{N_FOLDS}")
            
            train_loader, val_loader = get_data_loaders(
                fold_idx, train_idx, val_idx, full_dataset, batch_size
            )
            
            model, optimizer = get_model(model_name, len(full_dataset.classes), trial)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop for this fold
            best_fold_score = 0.0
            
            for epoch in range(n_epochs):
                train_loss, train_f1 = train_one_epoch(model, optimizer, train_loader, criterion)
                val_loss, val_f1, val_kappa = validate(model, val_loader, criterion)
                
                # Combined metric
                val_score = 0.5 * val_f1 + 0.5 * val_kappa
                
                if val_score > best_fold_score:
                    best_fold_score = val_score
                
                # Report intermediate result for pruning (average score)
                # Instead, we can report the val_f1 of the current fold at the current epoch
                # Optuna pruning is usually per-step.
                # Use current epoch Score for pruning check occasionally or at end of epoch?
                # Simple strategy: just track best score of fold.
                pass
                
            fold_f1_scores.append(best_fold_score)
            
            # Intermediate reporting after each fold
            # We report the mean Score of the folds completed so far
            current_avg_score = np.mean(fold_f1_scores)
            trial.report(current_avg_score, step=fold_idx)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return np.mean(fold_f1_scores)

    return objective

# --- Main Tuning Loop ---

def run_tuning(model_name, n_trials=20, n_epochs=10):
    print(f"Starting tuning for {model_name}...")
    
    # Load dataset
    full_dataset_path = os.path.join(DATA_PATH, "train")
    if not os.path.exists(full_dataset_path):
        # Fallback to verify path logic or assume executed from correct dir
        # In notebook it was "../../Datos...", let's try to handle absolute path if needed
        # But user script usually runs from current dir.
        pass

    full_train_dataset = datasets.ImageFolder(full_dataset_path)
    print(f"Dataset loaded: {len(full_train_dataset)} images, classes: {full_train_dataset.classes}")
    
    objective = create_objective(model_name, full_train_dataset, n_epochs=n_epochs)
    
    # Use SQLite for persistence and parallelism
    db_url = "sqlite:///hyperparameter_tuning.db"
    
    # Create study (load if exists allows resuming)
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"{model_name}_tuning",
        storage=db_url,
        load_if_exists=True
    )
    
    # Run optimization (user can run multiple separate processes of this script)
    print(f"Running {n_trials} trials... (Total trials in DB might differ)")
    
    # Loop for periodic reporting as requested
    for i in range(1, n_trials + 1):
        study.optimize(objective, n_trials=1)
        
        trial = study.best_trial
        print(f"\n[Progress {i}/{n_trials}] Current Best Score (0.5*F1+0.5*Kappa): {trial.value:.4f}")
        print("  Best Params so far:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    print("\n" + "="*40)
    print(f"Final Best trial for {model_name}:")
    trial = study.best_trial
    print(f"  Score (0.5*F1 + 0.5*Kappa): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*40 + "\n")
    
    return study

if __name__ == "__main__":
    # Ensure data path exists roughly where we expect, or warn
    if not os.path.exists(os.path.join(DATA_PATH, "train")):
        print(f"WARNING: Data path {DATA_PATH}/train not found. Check path.")
        
    # Run tuning for both models
    # Reduce n_epochs and n_trials for initial testing if needed
    # User asked to resume interrupted DenseNet121, but tuning is a fresh start for params.
    
    # Optional: Allow selecting model via args, but for now run both sequentially or comment out
    print("Select model to tune:")
    print("1. ResNet18")
    print("2. DenseNet121")
    # Hardcoding to run just a quick test or both depending on user preference? 
    # I'll default to running both with small trial count to show it works, then user can expand.
    # Actually, I'll allow running one by default to save time.
    
    # Uncomment to run:
    # run_tuning("resnet18", n_trials=10, n_epochs=10)
    # run_tuning("densenet121", n_trials=10, n_epochs=10)
    
    # By default, let's just print help, or users can modify the script
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "densenet121"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    
    args = parser.parse_args()
    
    # Only support ResNet18 for now as requested
    if args.model != "resnet18":
        print("Note: Focusing on ResNet18 tuning per user request.")
    
    run_tuning(args.model, n_trials=args.trials, n_epochs=args.epochs)
