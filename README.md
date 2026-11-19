# Permanencia estudiantil

Aplicación de Samuel Hernandez para el parcial de Big Data. Usa el pipeline de regresión logística multinomial entrenado con PySpark ML para estimar la trayectoria de un estudiante al cierre del segundo semestre. Muestra la clase predicha y la probabilidad de graduación, deserción y continuidad de matrícula.

## Ejecutar localmente

Usa Python 3.12. Abre una terminal en la carpeta que contiene `app.py` e instala las dependencias en un entorno virtual.

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

En Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre [la aplicación local](http://localhost:8501) en el navegador. Los valores de ejemplo están cargados para que puedas pulsar **Predecir** y comprobar el resultado.

## Cómo funciona

El entrenamiento y la exportación se hicieron con Spark ML. Para la inferencia, PyArrow lee los archivos Parquet del pipeline original en `modelo_final/`, y Streamlit conserva los parámetros en memoria con `st.cache_resource`. La app aplica las mismas fórmulas de aprobación y variación de nota, codificación categórica, orden del vector, escalado y regresión logística. Usa `modelo_final/labels.json` para relacionar las probabilidades con sus clases. Esta ruta de inferencia funciona sin una SparkSession ni una instalación de Java.

Los códigos de programa, jornada y formación previa se muestran con los nombres de la [ficha del conjunto de datos original](https://doi.org/10.24432/C5MC89).

El banner de `assets/banner.jpg` se generó con ImageGen. Se pidió una fotografía editorial de un cuaderno abierto, un bolígrafo verde azulado y un libro verde sobre el escritorio de una biblioteca universitaria, con luz natural y sin texto.

## Streamlit Community Cloud

Sube el contenido de esta carpeta a tu repositorio y conserva `modelo_final/`, `assets/` y `.streamlit/` junto a `app.py`, `inferencia.py` y `requirements.txt`. Selecciona `app.py` como archivo principal y Python 3.12 en los ajustes de despliegue. Consulta las [instrucciones de dependencias de Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

**URL pública pendiente de despliegue.** Agrega aquí la dirección `https://…streamlit.app` que asigne Streamlit.
