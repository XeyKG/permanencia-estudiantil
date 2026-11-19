import base64
import logging
from pathlib import Path

import streamlit as st

from inferencia import cargar_modelo, predecir

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modelo_final"
AUTHOR = "Samuel Hernandez"
LOGGER = logging.getLogger(__name__)

COURSES = {
    9003: "Agronomía",
    171: "Animación y diseño multimedia",
    9853: "Educación básica",
    9500: "Enfermería",
    9085: "Enfermería veterinaria",
    9130: "Equinocultura",
    9147: "Gestión",
    9991: "Gestión (jornada nocturna)",
    9556: "Higiene bucodental",
    9119: "Ingeniería informática",
    9773: "Periodismo y comunicación",
    9670: "Publicidad y gestión de marketing",
    9070: "Diseño de comunicación",
    9238: "Servicio social",
    8014: "Servicio social (jornada nocturna)",
    33: "Tecnologías de producción de biocombustibles",
    9254: "Turismo",
}
SCHEDULES = {1: "Diurna", 0: "Nocturna"}
QUALIFICATIONS = {
    1: "Educación secundaria",
    2: "Educación superior · bachillerato universitario",
    3: "Educación superior · licenciatura",
    4: "Educación superior · maestría",
    5: "Educación superior · doctorado",
    6: "Estudios superiores cursados",
    9: "12.º año sin completar",
    10: "11.º año sin completar",
    12: "Otro · 11.º año",
    14: "10.º año",
    15: "10.º año sin completar",
    19: "Educación básica · 3.er ciclo",
    38: "Educación básica · 2.º ciclo",
    39: "Especialización tecnológica",
    40: "Educación superior · título de 1.er ciclo",
    42: "Curso técnico superior profesional",
    43: "Educación superior · maestría de 2.º ciclo",
}
CLASS_NAMES = {
    "Graduate": "Graduación",
    "Dropout": "Deserción",
    "Enrolled": "Continúa matriculado",
}
@st.cache_resource(show_spinner=False)
def load_resources():
    return cargar_modelo(MODEL_DIR)


def category(label, options, default):
    codes = sorted(options, key=options.get)
    return st.selectbox(
        label, codes, index=codes.index(default),
        format_func=lambda code: options[code],
    )


def semester(title, suffix, defaults):
    st.markdown(f"**{title}**")
    enrolled, approved, grade, evaluations = defaults
    count_limit = 2_147_483_647
    prefix = f"curricularunits{suffix}sem"
    return {
        f"{prefix}enrolled": st.number_input(
            "Unidades matriculadas", 0, count_limit, enrolled, 1, key=f"{suffix}_enrolled",
        ),
        f"{prefix}approved": st.number_input(
            "Unidades aprobadas", 0, count_limit, approved, 1, key=f"{suffix}_approved",
        ),
        f"{prefix}grade": st.number_input(
            "Nota media (0 a 20)", 0.0, 20.0, grade, 0.1, format="%.2f",
            key=f"{suffix}_grade",
        ),
        f"{prefix}evaluations": st.number_input(
            "Evaluaciones realizadas", 0, count_limit, evaluations, 1,
            key=f"{suffix}_evaluations",
        ),
    }


def percent(value):
    return f"{value:.2%}".replace(".", ",")


def show_result(result):
    import pandas as pd

    st.header("Resultado de la predicción", anchor=False)
    predicted = result["label"]
    probabilities = result["probabilities"]
    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        with st.container(border=True):
            st.caption("Clase predicha")
            st.subheader(CLASS_NAMES[predicted], anchor=False)
            st.metric("Probabilidad estimada", percent(probabilities[predicted]))
    with right:
        with st.container(border=True):
            st.markdown("**Probabilidad por clase**")
            table = pd.DataFrame({
                "Clase": [CLASS_NAMES[label] for label in probabilities],
                "Estimación": [percent(value) for value in probabilities.values()],
                "Distribución": list(probabilities.values()),
            })
            st.dataframe(
                table, hide_index=True, width="stretch",
                column_config={
                    "Distribución": st.column_config.ProgressColumn(
                        "Distribución", min_value=0.0, max_value=1.0, format=" ",
                    ),
                },
            )
    with st.expander("Variables calculadas"):
        first, second, variation = result["derived"]
        st.write(
            f"Aprobación del primer semestre **{percent(first)}** · "
            f"Aprobación del segundo semestre **{percent(second)}** · "
            f"Variación de la nota **{variation:+.2f}** puntos"
        )


def main():
    st.set_page_config(page_title="Permanencia estudiantil", layout="wide")
    stylesheet = (ROOT / "assets" / "styles.css").read_text(encoding="utf-8")
    st.html(f"<style>{stylesheet}</style>")
    photo = base64.b64encode((ROOT / "assets" / "banner.jpg").read_bytes()).decode("ascii")
    st.html(f"""
        <section class="academic-hero" aria-labelledby="academic-title">
          <div class="academic-hero-copy">
            <h1 id="academic-title">Permanencia<br>estudiantil</h1>
            <p class="academic-intro">Estima la trayectoria de un estudiante a partir
            de sus dos primeros semestres.</p>
          </div>
          <img src="data:image/jpeg;base64,{photo}" alt="Cuaderno abierto en una biblioteca universitaria">
        </section>
    """)
    st.header("Datos del estudiante", anchor=False)
    st.write("Edita los valores de ejemplo y pulsa **Predecir** para ver la clase y sus probabilidades.")
    with st.form("student_record", clear_on_submit=False):
        st.subheader("Ingreso", anchor=False)
        program, schedule, qualification = st.columns([1.2, 0.8, 1.2], gap="medium")
        with program:
            course = category("Programa", COURSES, 171)
        with schedule:
            attendance = category("Jornada", SCHEDULES, 1)
        with qualification:
            previous = category("Formación previa", QUALIFICATIONS, 1)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            entry_grade = st.number_input("Nota de admisión (0 a 200)", 0.0, 200.0, 128.0, 0.1)
        with col2:
            prior_grade = st.number_input("Nota de formación previa (0 a 200)", 0.0, 200.0, 126.0, 0.1)

        st.subheader("Rendimiento académico", anchor=False)
        st.caption("Usa 0 como nota media si el estudiante no tiene una nota registrada ese semestre.")
        first_col, second_col = st.columns(2, gap="medium")
        with first_col:
            first = semester("Primer semestre", "1st", (6, 3, 11.0, 8))
        with second_col:
            second = semester("Segundo semestre", "2nd", (6, 1, 10.5, 7))
        submitted = st.form_submit_button("Predecir", type="primary")

    if submitted:
        st.session_state.pop("prediction", None)
        if any(
            st.session_state[f"{sem}_approved"] > st.session_state[f"{sem}_enrolled"]
            for sem in ("1st", "2nd")
        ):
            st.error("Las unidades aprobadas deben ser menores o iguales a las matriculadas en cada semestre.")
        else:
            values = {
                "course": course,
                "daytimeeveningattendance": attendance,
                "previousqualification": previous,
                "admissiongrade": entry_grade,
                "previousqualificationgrade": prior_grade,
                **first,
                **second,
            }
            try:
                with st.spinner("Calculando la predicción…"):
                    st.session_state["prediction"] = predecir(values, load_resources())
            except Exception:
                LOGGER.exception("Error al aplicar el modelo")
                st.error("No se pudo generar la predicción. Revisa los archivos de modelo_final.")

    if "prediction" in st.session_state:
        with st.container(key="resultado"):
            show_result(st.session_state["prediction"])
    st.divider()
    st.caption(f"Autor: {AUTHOR} · Parcial de Big Data")


if __name__ == "__main__":
    main()
