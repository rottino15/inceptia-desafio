"""Carga y genera embeddings para la búsqueda de productos."""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "productos.index"
TEXTOS_PATH = "textos.pkl"
DF_PATH = "productos_df.pkl"
CSV_PATH = "productos.csv"


@lru_cache(maxsize=1)
def load_resources() -> Tuple[SentenceTransformer, pd.DataFrame, faiss.Index]:
    """Return the model, dataframe and FAISS index.

    If the index and embeddings are already stored on disk they are loaded,
    otherwise they are generated and persisted for future runs.
    """

    model = SentenceTransformer(MODEL_NAME)

    if (
        os.path.exists(INDEX_PATH)
        and os.path.exists(TEXTOS_PATH)
        and os.path.exists(DF_PATH)
    ):
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTOS_PATH, "rb") as f:
            _ = pickle.load(f)
        df = pd.read_pickle(DF_PATH)
        return model, df, index

    df = pd.read_csv(CSV_PATH)
    textos = df.apply(
        lambda row: " ".join(
            [
                row["Nombre de producto"],
                row["Categoría"],
                row["Modelo"],
                row["Garantía"],
                row["Características"],
                str(row["Precio"]),
            ]
        ),
        axis=1,
    ).tolist()

    embeddings = model.encode(textos)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(TEXTOS_PATH, "wb") as f:
        pickle.dump(textos, f)
    df.to_pickle(DF_PATH)

    return model, df, index
