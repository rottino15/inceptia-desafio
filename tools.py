"""Herramientas personalizadas para el agente."""

from __future__ import annotations

import json

import numpy as np
from langchain.tools import tool

from carga_tabla import load_resources

model, df, index = load_resources()

# Cargar datos de compras una sola vez
with open("compras.json", "r", encoding="utf-8") as f:
    clientes_data = json.load(f)


@tool
def buscar_producto_por_embedding(query: str) -> str:
    """Busca productos semánticamente utilizando embeddings en FAISS."""

    # Generar el embedding de la consulta
    query_embedding = model.encode([query])[0]

    # Realizar la búsqueda en el índice FAISS
    _, indices = index.search(
        np.array([query_embedding]), k=3
    )  # Buscar los 3 productos más cercanos

    # Obtener los productos más cercanos
    result = df.iloc[indices[0]]

    # Formatear la respuesta
    if result.empty:
        return "No se encontraron productos relacionados."

    return result[["Nombre de producto", "Precio", "Características"]].to_string(
        index=False
    )


@tool
def consultar_compras(dni: str, codigo_compra: str | None = None) -> str:
    """Devuelve información de compras asociadas a un DNI.

    Si se especifica un ``codigo_compra`` se valida que pertenezca al cliente
    indicado. Si no, se listan las compras pendientes de entrega.
    """
    # Buscar al cliente por DNI
    cliente = next((c for c in clientes_data if c["dni"] == dni), None)
    if not cliente:
        return f"No se encontró ningún cliente con el DNI {dni}."

    # Si se proporciona un código de compra, validar que pertenezca a ese cliente
    if codigo_compra:

        compra = next(
            (
                compra
                for compra in cliente["compras"]
                if compra["codigo de compra"] == codigo_compra
            ),
            None,
        )

        if not compra:
            return (
                f"No se encontró ninguna compra con el código {codigo_compra} "
                f"asociada al DNI {dni}."
            )
        return (
            f"Compra encontrada:\n"
            f"Cliente: {cliente['nombre']}\n"
            f"Producto: {compra['producto']}\n"
            f"Fecha: {compra['fecha']}\n"
            f"Entregado: {'Sí' if compra['entregado'] else 'No'}"
        )

    # Si no hay código, devolver compras no entregadas
    compras_pendientes = [
        compra for compra in cliente["compras"] if not compra["entregado"]
    ]
    if not compras_pendientes:
        return f"No hay compras pendientes para el DNI {dni}."

    respuesta = f"Compras no entregadas de {cliente['nombre']}:\n"
    for compra in compras_pendientes:
        respuesta += (
            f"- {compra['producto']} (Código: {compra['codigo de compra']}, "
            f"Fecha: {compra['fecha']})\n"
        )
    return respuesta.strip()


def get_tools() -> list:
    """Devuelve la lista de herramientas disponibles para el agente."""

    return [buscar_producto_por_embedding, consultar_compras]
