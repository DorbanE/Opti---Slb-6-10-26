import streamlit as st
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

st.set_page_config(page_title="Optimización Mochila Binaria", layout="centered")

st.title("🎒 Optimizador de Mochila Binaria Interactivo")
st.markdown("Modifica los parámetros en la barra lateral para ver cómo cambian los resultados del problema de optimización en tiempo real.")

# --- BARRA LATERAL: ENTRADA DE PARÁMETROS ---
st.sidebar.header("Coeficientes de la Función Objetivo")
st.sidebar.latex(r"\text{Maximizar } c_1 X_1 + c_2 X_2 + c_3 X_3 + c_4 X_4")
c1 = st.sidebar.slider("Coeficiente c1 (X1)", min_value=1, max_value=30, value=14)
c2 = st.sidebar.slider("Coeficiente c2 (X2)", min_value=1, max_value=30, value=5)
c3 = st.sidebar.slider("Coeficiente c3 (X3)", min_value=1, max_value=30, value=7)
c4 = st.sidebar.slider("Coeficiente c4 (X4)", min_value=1, max_value=30, value=3)

st.sidebar.header("Restricción de Capacidad")
st.sidebar.latex(r"a_1 X_1 + a_2 X_2 + a_3 X_3 + a_4 X_4 \le B")
a1 = st.sidebar.slider("Peso a1 (X1)", min_value=1, max_value=15, value=8)
a2 = st.sidebar.slider("Peso a2 (X2)", min_value=1, max_value=15, value=3)
a3 = st.sidebar.slider("Peso a3 (X3)", min_value=1, max_value=15, value=4)
a4 = st.sidebar.slider("Peso a4 (X4)", min_value=1, max_value=15, value=2)
capacidad = st.sidebar.slider("Capacidad Máxima (B)", min_value=1, max_value=50, value=9)

# --- RESOLUCIÓN DEL MODELO ---
c = -np.array([c1, c2, c3, c4])
A = np.array([[a1, a2, a3, a4]])
constraint = LinearConstraint(A, -np.inf, capacidad)
bounds = Bounds(0, 1)
integrality = np.ones_like(c)

res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)

# --- MOSTRAR RESULTADOS ---
st.subheader("Modelo de Optimización Actual")
st.latex(f"\\text{{Maximizar }} {c1}X_1 + {c2}X_2 + {c3}X_3 + {c4}X_4")
st.latex(f"{a1}X_1 + {a2}X_2 + {a3}X_3 + {a4}X_4 \\le {capacidad}")
st.latex(r"X_i \in \{0, 1\}")

st.divider()

if res.success:
    solucion = np.round(res.x).astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Valor Máximo de la Función Objetivo (Z)", value=int(-res.fun))
        peso_total = sum(solucion[i] * [a1, a2, a3, a4][i] for i in range(4))
        st.metric(label="Capacidad Utilizada", value=f"{peso_total} / {capacidad}")
        
    with col2:
        st.subheader("Variables de Decisión:")
        for i in range(4):
            st.write(f"**X{i+1}** = {solucion[i]}")
else:
    st.error(f"No se pudo encontrar una solución óptima: {res.message}")
