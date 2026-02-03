import os
import sys
import random
import numpy as np
import optuna
from optuna.trial import TrialState
from ultralytics import YOLO
from pathlib import Path
import shutil
from contextlib import redirect_stdout, redirect_stderr
import logging

# --- Constants & Configuration ---
SEED = 42
N_FOLDS = 5
EPOCHS_PER_TRIAL = 25  # Default, se puede cambiar con args
DATA_PATH = "../../Datos/images/SPLIT_MULTICLASE"  # Relative path as per notebook

# Silenciar completamente logging de ultralytics
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
os.environ['YOLO_VERBOSE'] = 'False'

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # YOLO maneja seeds internamente, pero lo seteamos igual
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# --- Helper Functions ---

# NOTA: Para clasificación, YOLO NO usa archivos YAML
# En su lugar, espera directamente un directorio con estructura:
# dataset/
#   train/
#     clase1/
#     clase2/
#   val/
#     clase1/
#     clase2/

def get_f1_kappa_from_model(model, val_path):
    """
    Calcula F1 macro y Kappa validando el modelo en el conjunto de validación.
    
    Returns:
        float: 0.5 * F1_macro + 0.5 * Kappa
    """
    from sklearn.metrics import f1_score, cohen_kappa_score
    from pathlib import Path
    import logging
    
    # Silenciar logging de ultralytics
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    
    try:
        val_path_obj = Path(val_path) / "val"
        
        all_preds = []
        all_labels = []
        
        # Para cada clase
        for class_idx, class_dir in enumerate(sorted(val_path_obj.iterdir())):
            if not class_dir.is_dir():
                continue
            
            # Recoger todas las imágenes de esta clase
            img_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            
            # Predecir cada imagen
            for img_path in img_files:
                try:
                    # Hacer predicción (silenciosamente)
                    results = model(str(img_path), verbose=False)
                    
                    # Obtener clase predicha
                    if len(results) > 0 and hasattr(results[0], 'probs'):
                        pred_class = int(results[0].probs.top1)
                        all_preds.append(pred_class)
                        all_labels.append(class_idx)
                except Exception as e:
                    continue  # Skip imágenes problemáticas
        
        # Verificar que tenemos predicciones
        if len(all_preds) == 0 or len(all_labels) == 0:
            print(f"⚠️  No predictions collected (check image formats)")
            return 0.0
        
        # Calcular métricas
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        # Métrica combinada (igual que ResNet/DenseNet)
        combined_score = 0.5 * f1_macro + 0.5 * kappa
        
        return combined_score
        
    except Exception as e:
        print(f"⚠️  Error calculando F1/Kappa: {e}")
        return 0.0

# --- Objective Function ---

def objective(trial, model_name="yolo26x-cls.pt", folds_base_path="../../Datos/images/SPLIT_MULTICLASE/folds"):
    """
    Función objetivo para Optuna con 5-Fold Cross-Validation.
    
    Args:
        trial: Trial de Optuna
        model_name: Nombre del modelo YOLO (yolo26n-cls.pt, yolo26s-cls.pt, yolo26x-cls.pt, etc.)
        folds_base_path: Path a la carpeta con los folds (fold_1, fold_2, etc.)
    """
    
    # --- Hiperparámetros a optimizar ---
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)  # Learning rate inicial
    lrf = trial.suggest_float("lrf", 0.01, 0.2)  # Factor de reducción del LR (lr final = lr0 * lrf)
    momentum = trial.suggest_float("momentum", 0.8, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16])  # Igual que en entrenamiento original
    
    # Optimizer - YOLO soporta SGD, Adam, AdamW, etc.
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    
    print(f"\nTrial {trial.number}: Entrenando con hiperparámetros:")
    print(f"  lr0={lr0:.6f}, lrf={lrf:.3f}, momentum={momentum:.3f}")
    print(f"  weight_decay={weight_decay:.6f}, dropout={dropout:.1f}")
    print(f"  batch_size={batch_size}, optimizer={optimizer}")
    
    # --- 5-Fold Cross Validation ---
    fold_scores = []
    folds_path = Path(folds_base_path)
    
    for fold_idx in range(1, N_FOLDS + 1):
        fold_name = f"fold_{fold_idx}"
        fold_path = folds_path / fold_name
        
        print(f"  Fold {fold_idx}/{N_FOLDS}...", end=" ", flush=True)
        
        if not fold_path.exists():
            print(f"❌ Error: {fold_path} no existe!")
            raise optuna.exceptions.TrialPruned()
        
        # Cargar modelo YOLO (nuevo modelo para cada fold)
        model = YOLO(model_name)
        
       # Entrenar en este fold
        try:
            results = model.train(
                data=str(fold_path.absolute()),
                epochs=EPOCHS_PER_TRIAL,
                imgsz=224,
                batch=batch_size,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                dropout=dropout,
                optimizer=optimizer,
                seed=SEED,
                deterministic=True,
                verbose=False,
                plots=False,
                save=False,
                project="hyperparameter_tuning",
                name=f"trial_{trial.number}_fold_{fold_idx}",
                exist_ok=True,
            )
            
            # Calcular F1 macro + Kappa (igual que ResNet/DenseNet)
            val_metric = get_f1_kappa_from_model(model, str(fold_path.absolute()))
            fold_scores.append(val_metric)
            
            print(f"✓ Score (0.5*F1+0.5*Kappa): {val_metric:.4f}")
            
            # Cleanup inmediato para ahorrar espacio
            trial_fold_dir = Path("hyperparameter_tuning") / f"trial_{trial.number}_fold_{fold_idx}"
            if trial_fold_dir.exists():
                shutil.rmtree(trial_fold_dir)
            
            # CRÍTICO: Liberar memoria GPU explícitamente entre folds
            del model
            del results
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise optuna.exceptions.TrialPruned()
        
        # Reporte intermedio después de cada fold (para pruning)
        current_avg = np.mean(fold_scores)
        trial.report(current_avg, step=fold_idx - 1)
        
        if trial.should_prune():
            print(f"  🔪 Trial podado después del fold {fold_idx}")
            raise optuna.exceptions.TrialPruned()
    
    # Retornar score promedio de los 5 folds
    avg_score = np.mean(fold_scores)
    print(f"  📊 Score promedio 5-folds (0.5*F1+0.5*Kappa): {avg_score:.4f}")
    return avg_score

# --- Main Tuning Loop ---

def run_tuning(model_name="yolo26n-cls.pt", n_trials=20):
    """
    Ejecuta la búsqueda de hiperparámetros.
    
    Args:
        model_name: Modelo YOLO a usar (yolo26n-cls.pt, yolo26s-cls.pt, yolo26m-cls.pt, yolo26x-cls.pt)
        n_trials: Número de trials de Optuna
    """
    print(f"🚀 Iniciando hiperparametrización para {model_name}")
    print(f"📊 Trials: {n_trials} | Épocas por trial: {EPOCHS_PER_TRIAL}")
    
    # Verificar que el modelo existe
    if not Path(model_name).exists():
        print(f"⚠️  Modelo {model_name} no encontrado localmente.")
        print(f"   YOLO lo descargará automáticamente si es un modelo oficial.")
    
    # Base de datos SQLite para persistencia
    storage_name = f"sqlite:///hyperparameters/yolo_tuning.db"
    study_name = f"{Path(model_name).stem}_tuning"
    
    # Crear directorio para la DB
    Path("hyperparameters").mkdir(exist_ok=True)
    
    # Crear estudio
    study = optuna.create_study(
        direction="maximize",  # Maximizar accuracy
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=SEED),  # Random = exploración uniforme
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)  # Poda temprana
    )
    
    # Wrapper de objective con model_name
    def objective_wrapper(trial):
        return objective(trial, model_name=model_name)
    
    # Optimizar con reportes periódicos
    print(f"\n{'='*60}")
    for i in range(1, n_trials + 1):
        study.optimize(objective_wrapper, n_trials=1)
        
        # Verificar si hay trials exitosos antes de acceder a best_trial
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed_trials:
            trial = study.best_trial
            print(f"\n[Progress {i}/{n_trials}] 🏆 Mejor Score (0.5*F1+0.5*Kappa): {trial.value:.4f}")
            print("  Mejores Parámetros:")
            for key, value in trial.params.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")
        else:
            print(f"\n[Progress {i}/{n_trials}] ⚠️  Aún no hay trials exitosos.")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"✅ OPTIMIZACIÓN COMPLETADA - {model_name}")
    print(f"{'='*60}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if completed_trials:
        trial = study.best_trial
        print(f"🏆 Mejor Score (0.5*F1+0.5*Kappa): {trial.value:.4f}")
        print(f"📝 Mejores Hiperparámetros:")
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
    else:
        print(f"❌ No se completaron trials exitosos.")
        print(f"   Verifica los errores en los logs arriba.")
    
    print(f"{'='*60}\n")
    
    return study

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hiperparametrización de YOLO con Optuna")
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolo26n-cls.pt",
        help="Modelo YOLO a usar (yolo26n-cls.pt, yolo26s-cls.pt, yolo26m-cls.pt, yolo26x-cls.pt)"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=20,
        help="Número de trials de Optuna"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Épocas de entrenamiento por trial"
    )
    
    args = parser.parse_args()
    
    # Actualizar épocas si se especifica
    EPOCHS_PER_TRIAL = args.epochs
    
    # Verificar que el path de datos existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: No se encuentra el dataset en {DATA_PATH}")
        print(f"   Verifica la ruta y ajusta DATA_PATH en el script.")
        exit(1)
    
    # Ejecutar tuning
    run_tuning(model_name=args.model, n_trials=args.trials)
