import streamlit as st
from joblib import load
import pandas as pd

CLASS_LABELS = {
    0: "No suscribirá un depósito a plazo",
    1: "Sí suscribirá un depósito a plazo"
}
MONTH_TRANSLATION = {
    "jan": "Enero", "feb": "Febrero", "mar": "Marzo",
    "apr": "Abril", "may": "Mayo", "jun": "Junio",
    "jul": "Julio", "aug": "Agosto", "sep": "Septiembre",
    "oct": "Octubre", "nov": "Noviembre", "dec": "Diciembre"
}

MONTH_ORDER = list(MONTH_TRANSLATION.keys())

st.set_page_config(page_title="Despliegue del Modelo", page_icon="🚀", layout="centered")
st.title("Predicción con Nuevos Datos")
st.write("Aplicación para predecir si un cliente suscribirá un depósito a plazo.")

# Cargar el pipeline

@st.cache_resource
def load_pack():
    return load("modelo_final.joblib")

try:
    pack = load_pack()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

pipeline = pack["pipeline"]
feature_metadata = pack["feature_metadata"]
classes_ = pack.get("classes_", [])
st.markdown("### Introduce los valores de las variables:")

# Usamos un formulario para evitar recalcular la predicción con cada pulsación
with st.form("prediction_form"):
    inputs = {}
    
    # Renderizamos las variables numéricas
    st.subheader("Variables Numéricas")
    num_cols = st.columns(2)
    num_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "numerical":
            with num_cols[num_idx % 2]:
                med = float(meta.get("median", 0.0))
                # Usamos number_input para entradas numéricas
                inputs[feat] = st.number_input(
                    label=feat,
                    min_value=float(meta.get("min", -1e9)),
                    max_value=float(meta.get("max", 1e9)),
                    value=med,
                    step=1.0 if med.is_integer() else 0.1
                )
            num_idx += 1
            
    # Renderizamos las variables categóricas
    st.subheader("Variables Categóricas")
    cat_cols = st.columns(2)
    cat_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "categorical":
            with cat_cols[cat_idx % 2]:
                opts = meta.get("options", [])
                inputs[feat] = st.selectbox(
                    label=feat,
                    options=opts,
                    index=0 if opts else None
                )
            cat_idx += 1

    # Renderizamos las variables ordinales
    st.subheader("Variables Ordinales")
    ord_cols = st.columns(2)
    ord_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "ordinal":
            with ord_cols[ord_idx % 2]:
                opts = meta.get("options", [])
                
                # Ordenar meses correctamente si es la variable "month"
                kwargs = {}
                if feat.lower() == "month":
                    opts = sorted(opts, key=lambda x: MONTH_ORDER.index(x.lower()) if x.lower() in MONTH_ORDER else 999)
                    kwargs["format_func"] = lambda x: MONTH_TRANSLATION.get(x.lower(), x)
                
                inputs[feat] = st.selectbox(
                    label=feat,
                    options=opts,
                    index=0 if opts else None,
                    **kwargs
                )
            ord_idx += 1
            
    st.markdown("---")
    submitted = st.form_submit_button("Predecir", use_container_width=True)

if submitted:
    # Convertimos las entradas en un DataFrame de una única fila
    X_new = pd.DataFrame([inputs])

    try:
        # Predecimos usando el Pipeline (que ya incorpora todo el preprocesamiento)
        proba = pipeline.predict_proba(X_new)[0]
        y_pred = pipeline.predict(X_new)[0]
        pred_text = CLASS_LABELS.get(y_pred, str(y_pred))
        
        st.success(f"### Predicción: **{pred_text}**")
        
        st.markdown("#### Probabilidades de la Predicción:")
        
        # Mostramos los resultados como métricas destacadas
        cols = st.columns(len(classes_))
        for i, (cls, p) in enumerate(zip(classes_, proba)):
            pred_text = CLASS_LABELS.get(cls, str(cls))
            cols[i].metric(label=pred_text, value=f"{p*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")