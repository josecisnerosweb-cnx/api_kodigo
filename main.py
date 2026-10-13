"""API de duración de viajes en taxi (NYC, enero 2024)."""
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

INFO = json.load(open("info_modelo.json", encoding="utf-8"))
NOMBRES = {int(k): v for k, v in INFO["nombres"].items()}
ZONAS = INFO["zonas"]

modelo = XGBRegressor()
modelo.load_model("modelo_duracion.ubj")

app = FastAPI(title="Duración de viajes en taxi", version="1.0")


class Viaje(BaseModel):
    distancia: float = Field(gt=0, le=100, description="Distancia estimada, en millas")
    hora: int = Field(ge=0, le=23, description="Hora de salida")
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes ... 6 = domingo")
    origen: int = Field(description="ID de la zona de origen")
    destino: int = Field(description="ID de la zona de destino")


def a_tabla(viajes):
    for v in viajes:
        if v["origen"] not in NOMBRES or v["destino"] not in NOMBRES:
            raise HTTPException(status_code=404, detail="Zona desconocida. Revisa /info.")
    tabla = pd.DataFrame(viajes).rename(columns={"distancia": "trip_distance"})
    tabla["origen"] = pd.Categorical(tabla["origen"], categories=ZONAS)
    tabla["destino"] = pd.Categorical(tabla["destino"], categories=ZONAS)
    return tabla[INFO["variables"]]


def resultado(fila):
    p10, p50, p90 = fila
    return {"p10": round(float(p10), 1), "p50": round(float(p50), 1), "p90": round(float(p90), 1),
            "mensaje": f"Tu viaje durará entre {p10:.0f} y {p90:.0f} minutos"}


@app.get("/")
def estado():
    return {"estado": "ok", "mensaje": "API de duración de viajes en taxi"}


@app.get("/info")
def info():
    return {"modelo": "XGBoost con cuantiles", "variables": INFO["variables"],
            "cuantiles": INFO["cuantiles"], "zonas": len(ZONAS)}


@app.get("/zonas/{zona_id}")
def zona(zona_id: int):
    if zona_id not in NOMBRES:
        raise HTTPException(status_code=404, detail=f"La zona {zona_id} no existe")
    return {"id": zona_id, "nombre": NOMBRES[zona_id]}


@app.post("/predecir")
def predecir(viaje: Viaje):
    tabla = a_tabla([viaje.model_dump()])
    return resultado(np.sort(modelo.predict(tabla), axis=1)[0])


@app.post("/predecir/lote")
def predecir_lote(viajes: list[Viaje]):
    tabla = a_tabla([v.model_dump() for v in viajes])
    return {"resultados": [resultado(f) for f in np.sort(modelo.predict(tabla), axis=1)]}
