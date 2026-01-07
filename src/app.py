import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Crosspath AI",
    page_icon="🩺",
    layout="centered"
)

# --- RUTAS ---
# Buscamos el modelo ONNX en la carpeta assets/models
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(base_dir, 'assets', 'models', 'crosspath_ai.onnx')

# --- DICCIONARIO DE ETIQUETAS (DermaMNIST/HAM10000) ---
# El orden es CRÍTICO. Debe coincidir con el entrenamiento.
LABELS = {
    0: 'Queratosis Actínica (akiec)',
    1: 'Carcinoma Basocelular (bcc)',
    2: 'Lesión Benigna (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Nevus / Lunar (nv)',
    6: 'Lesión Vascular (vasc)'
}

# --- FUNCIONES ---
def load_model():
    """Carga el cerebro ONNX en memoria"""
    try:
        session = ort.InferenceSession(MODEL_PATH)
        return session
    except Exception as e:
        st.error(f"❌ Error crítico: No se encuentra el modelo en {MODEL_PATH}")
        st.stop()

def preprocess_image(image):
    """Transforma la foto HD del celular a lo que ve la IA (28x28 píxeles)"""
    # 1. Redimensionar a 28x28 (Lo que aprendió la IA)
    img = image.resize((28, 28))
    
    # 2. Convertir a Array NumPy
    img_array = np.array(img).astype('float32')
    
    # 3. Normalizar (Igual que en el entrenamiento: (x - 0.5) / 0.5)
    img_array = (img_array / 255.0 - 0.5) / 0.5
    
    # 4. Transponer canales: De (28, 28, 3) a (3, 28, 28)
    # PyTorch/ONNX piden: [Batch, Canales, Alto, Ancho]
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # 5. Agregar dimensión de Batch (1, 3, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- INTERFAZ GRÁFICA ---
st.title("🩺 Crosspath AI")
st.markdown("**Sistema de Soporte al Diagnóstico Dermatológico**")
st.caption("Powered by Raspberry Pi 5 & Edge AI")

# Cargar modelo una sola vez
session = load_model()

# Selector de entrada
opcion = st.radio("Seleccione método de entrada:", ("📸 Usar Cámara", "📂 Subir Foto"))

image_input = None

if opcion == "📸 Usar Cámara":
    img_file = st.camera_input("Capture la lesión")
    if img_file:
        image_input = Image.open(img_file)
else:
    img_file = st.file_uploader("Suba una imagen dermatoscópica", type=['jpg', 'png', 'jpeg'])
    if img_file:
        image_input = Image.open(img_file)

# --- LÓGICA DE PREDICCIÓN ---
if image_input is not None:
    # Mostrar imagen original
    st.image(image_input, caption="Imagen Capturada", use_column_width=True)
    
    with st.spinner("🔬 Analizando texturas celulares..."):
        # 1. Preprocesar
        processed_img = preprocess_image(image_input)
        
        # 2. Inferencia ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        result = session.run([output_name], {input_name: processed_img})
        logits = result[0][0] # Sacamos los números crudos
        
        # 3. Convertir a Probabilidades (Softmax)
        exp_vals = np.exp(logits - np.max(logits))
        probs = exp_vals / exp_vals.sum()
        
        # 4. Obtener la clase ganadora
        pred_idx = np.argmax(probs)
        pred_label = LABELS[pred_idx]
        confidence = probs[pred_idx] * 100

    # --- RESULTADOS ---
    st.divider()
    st.subheader("Diagnóstico Sugerido:")
    
    # Semáforo de Riesgo
    # Melanoma (4) o Carcinoma (1) son ALTO RIESGO
    if pred_idx in [1, 4]:
        st.error(f"⚠️ **{pred_label.upper()}**")
        st.markdown("🚨 **Recomendación:** Referencia Inmediata a Oncología.")
    elif pred_idx in [0, 6]:
        st.warning(f"⚠️ **{pred_label.upper()}**")
        st.markdown("👀 **Recomendación:** Vigilancia o Biopsia preventiva.")
    else:
        st.success(f"✅ **{pred_label}**")
        st.markdown("ℹ️ **Recomendación:** Seguimiento de rutina.")

    st.metric(label="Confianza del Modelo", value=f"{confidence:.1f}%")
    
    # Mostrar desglose completo (Debugging para médicos)
    with st.expander("Ver probabilidades detalladas"):
        for i, prob in enumerate(probs):
            st.progress(float(prob))
            st.text(f"{LABELS[i]}: {prob*100:.1f}%")