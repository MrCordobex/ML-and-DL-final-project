import json
import os

notebook_path = "/home/dmore/code/Máster IA/01.-MASTER COURSES/04.-DEEP LEARNING/ML-and-DL-final-project/DAVID/final_binario/DL_binario.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Code for the new cell
new_source = [
    "# === CELDA AÑADIDA: GENERACIÓN DE DATASET BINARIO ===\n",
    "import os\n",
    "import shutil\n",
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "\n",
    "# Configuración de Rutas\n",
    "CSV_PATH = \"../../Datos/Diccionario.csv\"\n",
    "SOURCE_ROOT = \"../../Datos/images/MULTICLASE/\"\n",
    "TARGET_ROOT = \"../../Datos/images/DATASET_SPLIT_BINARIO\"\n",
    "\n",
    "# 1. Leer Diccionario\n",
    "df = pd.read_csv(CSV_PATH)\n",
    "print(f\"Leídas {len(df)} entradas del diccionario.\")\n",
    "\n",
    "# 2. Mapeo de Biclase\n",
    "# 0 -> BENIGN\n",
    "# 1 -> MALIGNANT\n",
    "label_map = {0: 'BENIGN', 1: 'MALIGNANT'}\n",
    "\n",
    "# 3. Preparar Directorio Destino\n",
    "if os.path.exists(TARGET_ROOT):\n",
    "    print(f\"Limpiando directorio destino existente: {TARGET_ROOT}\")\n",
    "    shutil.rmtree(TARGET_ROOT)\n",
    "os.makedirs(TARGET_ROOT, exist_ok=True)\n",
    "\n",
    "splits = ['train', 'val', 'test']\n",
    "for s in splits:\n",
    "    for l in label_map.values():\n",
    "        os.makedirs(os.path.join(TARGET_ROOT, s, l), exist_ok=True)\n",
    "\n",
    "# 4. Copiar Imágenes\n",
    "print(\"Copiando imágenes a estructura binaria...\")\n",
    "missing_count = 0\n",
    "copy_count = 0\n",
    "\n",
    "for index, row in tqdm(df.iterrows(), total=len(df)):\n",
    "    img_id = row['img_id']\n",
    "    split_csv = row['conjunto_train_test_val']\n",
    "    diagnostic = row['diagnostic']\n",
    "    biclase = row['biclase']\n",
    "    \n",
    "    # Mapear split del csv a nombre de carpeta\n",
    "    # csv: train, validation, test\n",
    "    # folder: train, val, test\n",
    "    if split_csv == 'validation':\n",
    "        split_folder = 'val'\n",
    "    else:\n",
    "        split_folder = split_csv\n",
    "        \n",
    "    label_folder = label_map.get(biclase, 'UNKNOWN')\n",
    "    \n",
    "    # Origen: ../../Datos/images/MULTICLASE/{Diagnostic}/{ImgID}\n",
    "    # Nota: El LS anterior mostró que están en subcarpetas por diagnóstico (ACK, BCC...)\n",
    "    src_path = os.path.join(SOURCE_ROOT, diagnostic, img_id)\n",
    "    dst_path = os.path.join(TARGET_ROOT, split_folder, label_folder, img_id)\n",
    "    \n",
    "    if os.path.exists(src_path):\n",
    "        shutil.copy2(src_path, dst_path)\n",
    "        copy_count += 1\n",
    "    else:\n",
    "        # Intentar buscar sin carpeta intermedia por si acaso\n",
    "        src_path_direct = os.path.join(SOURCE_ROOT, img_id)\n",
    "        if os.path.exists(src_path_direct):\n",
    "             shutil.copy2(src_path_direct, dst_path)\n",
    "             copy_count += 1\n",
    "        else:\n",
    "             # print(f\"Missing: {src_path}\")\n",
    "             missing_count += 1\n",
    "\n",
    "print(f\"Proceso completado. Copiadas: {copy_count}, Faltantes: {missing_count}\")\n",
    "print(f\"Nuevo dataset en: {TARGET_ROOT}\")\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_source
}

# Insert after the first cell (Markdown Title)
nb['cells'].insert(1, new_cell)

# ====== MODIFICACIONES PARA ADAPTAR A BINARIO ======

# Recorrer celdas para reemplazar configuraciones
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Cambiar ruta del dataset en la celda de carga de datos
        if 'RUTA_BASE =' in source and 'DATASET_SPLIT' in source:
            new_source_lines = []
            for line in cell['source']:
                if 'RUTA_BASE =' in line:
                    new_source_lines.append('RUTA_BASE = "../../Datos/images/DATASET_SPLIT_BINARIO" # MODIFICADO PARA BINARIO\\n')
                else:
                    new_source_lines.append(line)
            cell['source'] = new_source_lines
            
        # 2. Cambiar número de clases en las cabezas de los modelos
        # Buscamos "nn.Linear(..., 6)" y cambiamos a "nn.Linear(..., 2)"
        # Buscamos "num_classes=6" y cambiamos a "num_classes=2"
        if 'num_classes=6' in source:
             cell['source'] = [line.replace('num_classes=6', 'num_classes=2') for line in cell['source']]
        
        if ', 6)' in source and 'nn.Linear' in source:
             cell['source'] = [line.replace(', 6)', ', 2)') for line in cell['source']]
             
        # 3. Adaptar EDA
        if 'CSV_PATH =' in source and 'Diccionario.csv' in source and 'sns.barplot' in source:
             # Modificar EDA para mostrar distribución binaria
             new_eda_source = []
             for line in cell['source']:
                 if "conteo = datos_split['diagnostic'].value_counts()" in line:
                     new_eda_source.append("    # MODIFICADO BINARIO: Usar columna 'biclase' (0=Benign, 1=Malignant)\\n")
                     new_eda_source.append("    conteo = datos_split['biclase'].value_counts().sort_index()\\n")
                     new_eda_source.append("    conteo.index = ['BENIGN', 'MALIGNANT'] if len(conteo)==2 else conteo.index\\n")
                 elif "axes[i].set_xlabel('Diagnóstico')" in line:
                     new_eda_source.append("    axes[i].set_xlabel('Clase Binaria')\\n")
                 else:
                     new_eda_source.append(line)
             cell['source'] = new_eda_source

        # 4. Adaptar Pesos de Loss (si existen)
        # Buscar "weights =" o "class_weights"
        if 'weights =' in source and '1.0 / class_counts' in source:
             # Asegurarse que class_counts se calcule sobre binario si se recalcula, 
             # o si está hardcoded, cambiarlo.
             # Asumimos que el código original calcula los pesos dinámicamente desde el loader.
             pass 

        # 5. Adaptar nombres de clases para Matriz de Confusión
        # Buscar "class_names" o similar, o donde se define ['ACK', 'BCC', ...]"
        # Reemplazar lista de 6 clases por ['BENIGN', 'MALIGNANT']
        if "'ACK'" in source and "'MEL'" in source:
             import re
             # Reemplazar listas de clases
             # Pattern: \['ACK'.*?\]
             regex = r"\['ACK'.*?\]"
             new_list = "['BENIGN', 'MALIGNANT']"
             
             new_source_lines = []
             for line in cell['source']:
                 new_line = re.sub(regex, new_list, line)
                 new_source_lines.append(new_line)
             cell['source'] = new_source_lines

        # 6. Ensemble Logic
        # Ajustar pesos del ensemble si están hardcoded 
        # "weights = [0.2, 0.2, 0.6]" -> Quizás mantener o simplificar.
        # El usuario dijo "adaptes las cabezas... y el ensemble... pero que la estructura sea casi la misma"
        # Si el ensemble usa soft voting, funcionará igual con 2 salidas.
        
        # 7. Cambiar num_features en cabezas si es necesario (generalmente es auto, pero nn.Linear(num_ftrs, 6) ya lo cambiamos)

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook modificado exitosamente.")
