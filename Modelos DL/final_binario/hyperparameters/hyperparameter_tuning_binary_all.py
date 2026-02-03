import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
import optuna
from optuna.trial import TrialState
from ultralytics.data.augment import classify_augmentations, classify_transforms
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import logging
import gc
import warnings

# --- Constants & Configuration ---
SEED = 42
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR is DAVID/final_binario/hyperparameters
# We want ROOT/Datos. 
# .../DAVID/final_binario/hyperparameters -> ../../../Datos
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "Datos/images/SPLIT_BINARIO")

TRAIN_DIR = os.path.join(BASE_DATA_PATH, "train")
FOLDS_DIR = os.path.join(BASE_DATA_PATH, "folds_5cv")
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Benigno", "Maligno"]

# Configuración de Logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# --- Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# --- Data Preparation (Folds) ---
def prepare_physical_folds():
    """
    Genera directorios físicos para los 5 folds si no existen.
    Esto es necesario para YOLO y garantiza consistencia exacta entre modelos.
    Estructura: folds_5cv/fold_k/train y folds_5cv/fold_k/val
    """
    if os.path.exists(FOLDS_DIR):
        print(f"✅ Directorio de folds ya existe: {FOLDS_DIR}")
        return

    print(f"🔨 Generando folds físicos en {FOLDS_DIR}...")
    dataset = datasets.ImageFolder(TRAIN_DIR)
    targets = dataset.targets
    classes = dataset.classes
    filepaths = np.array(dataset.samples)[:, 0]
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, targets)):
        fold_name = f"fold_{fold_idx+1}"
        print(f"  Preparando {fold_name}...")
        
        # Crear directorios
        for split in ["train", "val"]:
            for cls in classes:
                os.makedirs(os.path.join(FOLDS_DIR, fold_name, split, cls), exist_ok=True)
                
        # Copiar imágenes (Symlinks podrían ahorrar espacio, pero copia es más segura para YOLO)
        # Usaremos symlinks si es posible para ahorrar espacio y tiempo
        
        for idx in train_idx:
             src = filepaths[idx]
             cls_name = classes[targets[idx]]
             dst = os.path.join(FOLDS_DIR, fold_name, "train", cls_name, os.path.basename(src))
             if not os.path.exists(dst):
                 shutil.copy2(src, dst)
                 
        for idx in val_idx:
             src = filepaths[idx]
             cls_name = classes[targets[idx]]
             dst = os.path.join(FOLDS_DIR, fold_name, "val", cls_name, os.path.basename(src))
             if not os.path.exists(dst):
                 shutil.copy2(src, dst)
                 
    print("✅ Generación de folds completada.")

# --- Data Loading (PyTorch Standard) ---
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
    erasing=0.3, # Cutout
)

val_pipeline = classify_transforms(
    size=IMG_SIZE,
    mean=MEAN,
    std=STD,
    interpolation="BILINEAR"
)

def get_dataloaders(fold_idx, batch_size):
    fold_path = os.path.join(FOLDS_DIR, f"fold_{fold_idx}")
    
    train_ds = datasets.ImageFolder(os.path.join(fold_path, "train"), transform=train_pipeline)
    val_ds = datasets.ImageFolder(os.path.join(fold_path, "val"), transform=val_pipeline)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Calcular pesos de clase (Weighted Loss)
    # Contamos 'Benigno' y 'Maligno' en este fold de train
    targets = train_ds.targets
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    # pesos = total / (num_classes * count)
    class_weights = torch.tensor(
        [total_samples / (2.0 * class_counts[0]), total_samples / (2.0 * class_counts[1])],
        dtype=torch.float
    ).to(DEVICE)
    
    return train_loader, val_loader, class_weights

# --- Metric: F1-Score (Maligno) ---
def calculate_f1_maligno(y_true, y_pred):
    # Asumiendo Maligno es clase 1 (verificar classes[1] == 'Maligno')
    # beta=1 pondera igual precision y recall
    return fbeta_score(y_true, y_pred, beta=1, pos_label=1, zero_division=0)

# --- Models: ResNet & DenseNet ---
def create_pytorch_model(model_name, trial, num_classes=2):
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        for param in model.layer4.parameters(): param.requires_grad = True # Unfreeze last block
        
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
        optimizer = optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': lr_backbone},
            {'params': model.fc.parameters(), 'lr': lr_head}
        ], weight_decay=weight_decay)
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = False
        for param in model.features.denseblock4.parameters(): param.requires_grad = True
        for param in model.features.norm5.parameters(): param.requires_grad = True
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier.in_features, num_classes)
        )
        optimizer = optim.AdamW([
            {'params': model.features.denseblock4.parameters(), 'lr': lr_backbone},
            {'params': model.features.norm5.parameters(), 'lr': lr_backbone},
            {'params': model.classifier.parameters(), 'lr': lr_head}
        ], weight_decay=weight_decay)
        
    return model.to(DEVICE), optimizer

def train_validate_pytorch(model, optimizer, train_loader, val_loader, class_weights, epochs=10):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_f1 = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Val
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = calculate_f1_maligno(all_labels, all_preds)
        if f1 > best_f1:
            best_f1 = f1
        
        print(f"  [Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f} | F1-Maligno: {f1:.4f}")
            
    return best_f1

def objective_pytorch(trial, model_name):
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    fold_scores = []
    
    for fold_idx in range(1, N_FOLDS + 1):
        train_loader, val_loader, class_weights = get_dataloaders(fold_idx, batch_size)
        model, optimizer = create_pytorch_model(model_name, trial)
        
        f1 = train_validate_pytorch(model, optimizer, train_loader, val_loader, class_weights, epochs=15)
        fold_scores.append(f1)
        
        trial.report(np.mean(fold_scores), step=fold_idx-1)
        
        # Cleanup memory after each fold
        del model
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return np.mean(fold_scores)

# --- Model: YOLO ---
def objective_yolo(trial, model_type="yolo26n-cls.pt"):
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 0.2)
    momentum = trial.suggest_float("momentum", 0.8, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    batch_size = 16 # Fijo para YOLO según multiclase
    
    fold_scores = []
    
    for fold_idx in range(1, N_FOLDS + 1):
        fold_path = os.path.join(FOLDS_DIR, f"fold_{fold_idx}")
        model = YOLO(model_type)
        
        # Train
        try:
            model.train(
                data=os.path.abspath(fold_path),
                epochs=15,
                imgsz=IMG_SIZE,
                batch=batch_size,
                lr0=lr0, lrf=lrf, momentum=momentum, weight_decay=weight_decay,
                dropout=dropout, optimizer=optimizer_name,
                project="hyperparameter_tuning_yolo_temp",
                name=f"trial_{trial.number}_fold_{fold_idx}",
                exist_ok=True, verbose=False, save=False, plots=False,
                freeze=10  # Congelar backbone (primeras 10 capas aprox)
            )
            
            # Validation (Custom for F2)
            # YOLO val() returns metrics but F2 might not be standard/easy to extract specifically for class 1
            # We will run inference manually on val set
            val_ds = datasets.ImageFolder(os.path.join(fold_path, "val"))
            y_true, y_pred = [], []
            
            results = model.predict(source=os.path.join(fold_path, "val"), stream=True, verbose=False)
            # El orden de stream puede no coincidir con ImageFolder, PERO
            # podemos iterar archivos. Mejor: usar el val loader de pytorch o iterar archivos y cargar con YOLO
            
            # Iteración segura
            class_to_idx = val_ds.class_to_idx # {'Benigno': 0, 'Maligno': 1}
            for cls_name in class_to_idx:
                 cls_dir = os.path.join(fold_path, "val", cls_name)
                 for img_file in os.listdir(cls_dir):
                     img_path = os.path.join(cls_dir, img_file)
                     res = model(img_path, verbose=False)[0]
                     pred = res.probs.top1
                     y_true.append(class_to_idx[cls_name])
                     y_pred.append(pred)
            
            f1 = calculate_f1_maligno(y_true, y_pred)
            fold_scores.append(f1)
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()
            shutil.rmtree(f"hyperparameter_tuning_yolo_temp/trial_{trial.number}_fold_{fold_idx}", ignore_errors=True)

        except Exception as e:
            print(f"Error YOLO Fold {fold_idx}: {e}")
            raise optuna.exceptions.TrialPruned()

        trial.report(np.mean(fold_scores), step=fold_idx-1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return np.mean(fold_scores)

# --- Main Execution ---
def run_all_tuning():
    print("🚀 INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS (BINARIO - MÉTRICA F1-SCORE)")
    prepare_physical_folds()
    
    # Base de datos en el mismo directorio del script (absolute path)
    db_path = os.path.join(SCRIPT_DIR, "binary_tuning_f1.db")
    db_name = f"sqlite:///{db_path}"
    print(f"📁 Base de datos de tuning: {db_name}")
    
    # 1. ResNet18
    print("\n\n--- Optimizando ResNet18 ---")
    study_resnet = optuna.create_study(
        direction="maximize", study_name="resnet18_binary_f1", storage=db_name, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study_resnet.optimize(lambda t: objective_pytorch(t, "resnet18"), n_trials=20)
    print("🏆 Mejores params ResNet18:", study_resnet.best_params)
    
    # Cleanup between studies
    del study_resnet
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. DenseNet121
    print("\n\n--- Optimizando DenseNet121 ---")
    study_densenet = optuna.create_study(
        direction="maximize", study_name="densenet121_binary_f1", storage=db_name, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study_densenet.optimize(lambda t: objective_pytorch(t, "densenet121"), n_trials=20)
    print("🏆 Mejores params DenseNet121:", study_densenet.best_params)

    # Cleanup between studies
    del study_densenet
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. YOLO26x
    print("\n\n--- Optimizando YOLO26x (puede tardar más) ---")
    
    # Ruta al pesos custom (DAVID/final/yolo26x-cls.pt)
    # PROJECT_ROOT/DAVID/final/yolo26x-cls.pt
    yolo_model_path = os.path.join(PROJECT_ROOT, "DAVID/final/yolo26x-cls.pt")
    
    if not os.path.exists(yolo_model_path):
        print(f"⚠️ No se encontró {yolo_model_path}, buscando en local...")
        yolo_model_path = "yolo26x-cls.pt"
        
    study_yolo = optuna.create_study(
        direction="maximize", study_name="yolo_binary_f1", storage=db_name, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study_yolo.optimize(lambda t: objective_yolo(t, yolo_model_path), n_trials=20)
    print("🏆 Mejores params YOLO:", study_yolo.best_params)

if __name__ == "__main__":
    run_all_tuning()
