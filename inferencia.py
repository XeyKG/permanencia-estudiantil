import json

import numpy as np
from pyarrow.parquet import ParquetFile


DERIVADAS = ("tasaaprobacion1", "tasaaprobacion2", "variacionnota")


def leer_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fila_parquet(stage):
    (part,) = stage.joinpath("data").glob("part-*.parquet")
    (row,) = ParquetFile(part).read().to_pylist()
    return row


def cargar_modelo(directory):
    manifest = leer_json(directory / "metadata" / "part-00000")
    stages = [
        directory / "stages" / f"{i}_{uid}"
        for i, uid in enumerate(manifest["paramMap"]["stageUids"])
    ]
    metadata = [leer_json(stage / "metadata" / "part-00000") for stage in stages]
    clases_esperadas = (
        "SQLTransformer", "StringIndexerModel", "OneHotEncoderModel",
        "VectorAssembler", "StandardScalerModel", "LogisticRegressionModel",
    )
    if tuple(stage["class"].split(".")[-1] for stage in metadata) != clases_esperadas:
        raise ValueError("El pipeline no tiene las etapas de la regresión logística del parcial.")
    _, indexer, encoder, assembler, scaler, _ = [
        stage["defaultParamMap"] | stage["paramMap"] for stage in metadata
    ]
    if indexer["handleInvalid"] != "keep" or encoder["dropLast"] or scaler["withMean"]:
        raise ValueError("La configuración de codificación o escalado no es compatible.")
    label_arrays = fila_parquet(stages[1])["labelsArray"]
    category_sizes = fila_parquet(stages[2])["categorySizes"]
    encoders = {}
    by_index = dict(zip(indexer["outputCols"], zip(indexer["inputCols"], label_arrays), strict=True))
    for indexed, output, size in zip(
        encoder["inputCols"], encoder["outputCols"], category_sizes, strict=True
    ):
        name, categories = by_index[indexed]
        encoders[output] = (name, dict(zip(categories, range(len(categories)))), size)

    std = np.array(fila_parquet(stages[4])["std"]["values"], dtype=float)
    scale = np.divide(1.0, std, out=np.zeros_like(std), where=std != 0)
    linear = fila_parquet(stages[5])
    matrix = linear["coefficientMatrix"]
    if not linear["isMultinomial"] or matrix["type"] != 1:
        raise ValueError("Se esperaba una matriz densa de regresión multinomial.")
    order = "C" if matrix["isTransposed"] else "F"
    weights = np.array(matrix["values"], dtype=float).reshape(
        (matrix["numRows"], matrix["numCols"]), order=order
    )
    intercept = np.array(linear["interceptVector"]["values"], dtype=float)
    labels = leer_json(directory / "labels.json")
    if weights.shape != (len(labels), len(scale)) or intercept.shape != (len(labels),):
        raise ValueError("Las dimensiones del modelo y las etiquetas difieren.")
    return assembler["inputCols"], encoders, scale, weights, intercept, labels


def predecir(values, model):
    columns, encoders, scale, weights, intercept, labels = model
    enrolled1 = values["curricularunits1stsemenrolled"]
    enrolled2 = values["curricularunits2ndsemenrolled"]
    grade1 = values["curricularunits1stsemgrade"]
    grade2 = values["curricularunits2ndsemgrade"]
    derived = (
        values["curricularunits1stsemapproved"] / enrolled1 if enrolled1 else 0.0,
        values["curricularunits2ndsemapproved"] / enrolled2 if enrolled2 else 0.0,
        grade2 - grade1 if grade1 > 0 and grade2 > 0 else 0.0,
    )
    fields = values | dict(zip(DERIVADAS, derived, strict=True))
    vector = []
    for column in columns:
        if column in encoders:
            name, indices, size = encoders[column]
            index = indices.get(str(fields[name]), len(indices))
            block = [0.0] * size
            block[index] = 1.0
            vector.extend(block)
        else:
            vector.append(float(fields[column]))

    logits = weights @ (np.array(vector) * scale) + intercept
    exps = np.exp(logits - logits.max())
    probabilities = exps / exps.sum()
    return {
        "label": labels[int(probabilities.argmax())],
        "probabilities": dict(zip(labels, map(float, probabilities), strict=True)),
        "derived": derived,
    }
