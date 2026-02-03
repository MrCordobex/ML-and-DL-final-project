# Proyecto Final ML y DL - Clasificación de Lesiones Cutáneas

## 📁 Estructura del Proyecto

### `Datos/`
Contiene los datasets originales y procesados:
- `metadata.csv`: Dataset original con información clínica
- `Train_raw.csv`, `Test_raw.csv`: División inicial de datos
- `Train_clean.csv`, `Test_clean.csv`: Datos preprocesados
- `images/`: Carpeta con las imágenes dermatoscópicas

### `Modelos ML/`
Implementación de modelos de Machine Learning sobre datos tabulares:
- `Prepocesing.ipynb`: Limpieza de datos, imputación (KNNImputer), codificación y división Train/Test
- `BloqueA_ML.ipynb`: Entrenamiento de modelos multiclase (Random Forest, Regresión Logística, KNN, Gradient Boosting, etc.)
- `Biclase_ML.ipynb`: Entrenamiento de modelos de clasificación binaria

### `Modelos DL/`
Modelos de Deep Learning para clasificación de imágenes:
- `final/`: Notebooks y modelos para clasificación multiclase
  - `DL_multiclase.ipynb`: Entrenamiento de CNNs (ResNet18, DenseNet121, etc.)
  - Modelos guardados (`.pth`) y checkpoints
- `final_binario/`: Notebooks y modelos para clasificación binaria
  - `DL_binario.ipynb`: Entrenamiento para detección binaria (Melanoma vs No Melanoma)
- `latex/`: Documentación técnica del proyecto en LaTeX

### `Modelos Híbrido/`
Modelos que combinan diferentes aproximaciones:
- `Stacking.ipynb`: Técnicas de Stacking para combinar predicciones
- `Attention.ipynb`: Implementación de mecanismos de atención
- `Modelo_Gate2.ipynb`: Arquitecturas con gating

## 🛠 Requisitos

```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
torchvision
ultralytics
```

## 👥 Autores

- David Moreda Amezcua
- Javier Cerón Contreras
- Pedro Martínez Huertas

**Máster Universitario en Inteligencia Artificial - Universidad Loyola (2025/2026)**
