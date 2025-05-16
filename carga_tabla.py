import os
import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Cargar el modelo solo una vez
model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths
INDEX_PATH = "productos.index"
TEXTOS_PATH = "textos.pkl"
DF_PATH = "productos_df.pkl"
CSV_PATH = "productos.csv"

# Si ya existen los archivos guardados
if os.path.exists(INDEX_PATH) and os.path.exists(TEXTOS_PATH) and os.path.exists(DF_PATH):
    print("Cargando embeddings e índice desde disco...")

    # Cargar FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Cargar textos (por si los necesitas más adelante)
    with open(TEXTOS_PATH, "rb") as f:
        textos = pickle.load(f)

    # Cargar DataFrame
    df = pd.read_pickle(DF_PATH)

else:
    print("Generando embeddings e índice FAISS...")

    # Cargar CSV
    df = pd.read_csv(CSV_PATH)

    # Unir columnas en un solo texto por producto
    textos = df.apply(lambda row: f"{row['Nombre de producto']} {row['Categoría']} {row['Modelo']} {row['Garantía']} {row['Características']} {row['Precio']}", axis=1).tolist()

    # Generar embeddings
    embeddings = model.encode(textos)

    # Crear índice FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Guardar los objetos
    faiss.write_index(index, INDEX_PATH)

    with open(TEXTOS_PATH, "wb") as f:
        pickle.dump(textos, f)

    df.to_pickle(DF_PATH)

    print("Embeddings e índice guardados.")


""" 
# Realizar una búsqueda de ejemplo (por ejemplo, con la palabra "pantalla")
query = "lenovo"
query_embedding = model.encode([query])

# Realizar la búsqueda en FAISS
distances, indices = index.search(np.array(query_embedding), k=3)  # k=3 para obtener los 3 más cercanos

# Mostrar los resultados
print(f"Distancias: {distances}")
print(f"Índices: {indices}")

# Mostrar los productos más cercanos
for idx in indices[0]:
    print(f"Producto: {df.iloc[idx]['Nombre de producto']}")
"""
