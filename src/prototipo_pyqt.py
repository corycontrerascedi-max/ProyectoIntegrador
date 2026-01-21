import sys
import os
# Importamos las herramientas graficas de Qt
# Agregamos QFileDialog para poder abrir el explorador de archivos
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt

# Importamos OpenCV para la camara y Numpy para matematicas
import cv2
import numpy as np
import onnxruntime as ort

class VentanaDermatoscopio(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- CONFIGURACION DE LA VENTANA ---
        self.setWindowTitle("Sistema Dermatoscopico - Tec de Monterrey")
        self.setGeometry(100, 100, 1000, 600)
        
        # Estilo clinico: Fondo blanco, texto negro
        self.setStyleSheet("background-color: #ffffff; color: black;")

        # --- CARGA DEL MODELO ONNX ---
        # Definimos las 7 clases del dataset DermaMNIST para poder dar el nombre correcto
        self.clases = {
            0: 'Queratosis Actinica (akiec)',
            1: 'Carcinoma Basocelular (bcc)',
            2: 'Lesion Benigna (bkl)',
            3: 'Dermatofibroma (df)',
            4: 'MELANOMA (mel)',
            5: 'Nevus / Lunar (nv)',
            6: 'Lesion Vascular (vasc)'
        }
        
        self.sesion_ia = None
        try:
            # Buscamos el modelo en la carpeta assets/models
            ruta_modelo = "./assets/models/crosspath_ai.onnx"
            self.sesion_ia = ort.InferenceSession(ruta_modelo)
            print(f"Modelo cargado desde: {ruta_modelo}")
        except Exception as e:
            print(f"Error critico cargando modelo: {e}")
            print("Asegurate de haber ejecutado export_onnx.py primero.")

        # --- CONFIGURACION DE LA CAMARA ---
        # Iniciamos la camara (indice 0 suele ser la webam integrada o USB)
        self.captura = cv2.VideoCapture(0)
        self.imagen_actual = None
        self.modo_archivo = False # Bandera para saber si estamos usando camara o archivo

        # --- DISEÑO DE LA INTERFAZ ---
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout()
        widget_central.setLayout(layout_principal)

        # --- COLUMNA IZQUIERDA: VISUALIZADOR ---
        self.etiqueta_video = QLabel("Iniciando Sistema...")
        self.etiqueta_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Fijamos tamaño estandar 640x480 para mantener consistencia
        self.etiqueta_video.setFixedSize(640, 480) 
        self.etiqueta_video.setStyleSheet("border: 2px solid #cccccc; background-color: #000;")
        
        layout_principal.addWidget(self.etiqueta_video)

        # --- COLUMNA DERECHA: LOS CONTROLES ---
        layout_botones = QVBoxLayout()
        
        # Titulo del Panel
        etiqueta_titulo = QLabel("Panel de Control")
        etiqueta_titulo.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        etiqueta_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        etiqueta_titulo.setStyleSheet("color: black;")
        layout_botones.addWidget(etiqueta_titulo)

        layout_botones.addStretch()

        # Boton 1: Capturar Video en Vivo
        self.boton_capturar = QPushButton("CONGELAR VIDEO")
        self.boton_capturar.setStyleSheet("background-color: #007acc; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_capturar.clicked.connect(self.congelar_imagen)
        layout_botones.addWidget(self.boton_capturar)

        # Boton 2: Cargar Archivo (NUEVO)
        self.boton_cargar = QPushButton("SUBIR ARCHIVO")
        self.boton_cargar.setStyleSheet("background-color: #17a2b8; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_cargar.clicked.connect(self.cargar_archivo)
        layout_botones.addWidget(self.boton_cargar)

        # Boton 3: Analizar (IA)
        self.boton_analizar = QPushButton("ANALIZAR LESION")
        self.boton_analizar.setStyleSheet("background-color: #28a745; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_analizar.clicked.connect(self.analizar_lesion)
        self.boton_analizar.setEnabled(False) # Desactivado al inicio
        layout_botones.addWidget(self.boton_analizar)

        # Boton 4: Reiniciar
        self.boton_reset = QPushButton("REINICIAR SISTEMA")
        self.boton_reset.setStyleSheet("background-color: #6c757d; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_reset.clicked.connect(self.reiniciar_sistema)
        layout_botones.addWidget(self.boton_reset)

        layout_botones.addStretch()

        # Etiqueta de RESULTADO
        self.etiqueta_resultado = QLabel("Estado: Esperando entrada...")
        self.etiqueta_resultado.setFont(QFont("Arial", 14))
        self.etiqueta_resultado.setStyleSheet("background-color: #f0f0f0; color: black; padding: 10px; border-radius: 5px; border: 1px solid #ddd;")
        self.etiqueta_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.etiqueta_resultado.setWordWrap(True)
        layout_botones.addWidget(self.etiqueta_resultado)

        layout_principal.addLayout(layout_botones)

        # --- TEMPORIZADOR DE VIDEO ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_cuadro)
        self.timer.start(30) # 30ms = ~30 FPS

    def actualizar_cuadro(self):
        # Solo actualizamos si estamos en modo camara (no archivo)
        if not self.modo_archivo:
            ret, cuadro = self.captura.read()
            if ret:
                # Guardamos en variable global para analisis posterior
                # Convertimos BGR (OpenCV) a RGB (Pantalla)
                self.imagen_actual = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
                self.mostrar_imagen_en_gui(self.imagen_actual)

    def mostrar_imagen_en_gui(self, imagen_rgb):
        # Funcion auxiliar para pintar la imagen en la etiqueta
        alto, ancho, canales = imagen_rgb.shape
        paso = canales * ancho
        imagen_qt = QImage(imagen_rgb.data, ancho, alto, paso, QImage.Format.Format_RGB888)
        
        mapa_bits = QPixmap.fromImage(imagen_qt)
        
        # Escalado suave manteniendo proporcion
        self.etiqueta_video.setPixmap(mapa_bits.scaled(
            640, 480,
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

    def congelar_imagen(self):
        # Detenemos el video en vivo
        self.timer.stop()
        self.modo_archivo = False # Confirmamos que viene del video
        
        # Habilitamos analisis
        self.boton_analizar.setEnabled(True)
        self.boton_capturar.setEnabled(False)
        self.etiqueta_resultado.setText("Imagen congelada. Lista para analisis.")

    def cargar_archivo(self):
        # Abrimos dialogo de sistema para seleccionar archivo
        archivo, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccionar Imagen Dermatoscopica", 
            "", 
            "Imagenes (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if archivo:
            # Detenemos la camara porque vamos a mostrar una foto estatica
            self.timer.stop()
            self.modo_archivo = True
            
            # Leemos la imagen con OpenCV
            img_leida = cv2.imread(archivo)
            
            if img_leida is not None:
                # Convertimos BGR a RGB
                self.imagen_actual = cv2.cvtColor(img_leida, cv2.COLOR_BGR2RGB)
                self.mostrar_imagen_en_gui(self.imagen_actual)
                
                # Activamos botones
                self.boton_analizar.setEnabled(True)
                self.boton_capturar.setEnabled(True) # Permitimos volver a capturar si quieren
                self.etiqueta_resultado.setText("Archivo cargado correctamente.")
            else:
                self.etiqueta_resultado.setText("Error: No se pudo leer el archivo.")

    def reiniciar_sistema(self):
        # Reactivamos la camara
        self.modo_archivo = False
        self.timer.start(30)
        
        # Reseteamos botones
        self.boton_analizar.setEnabled(False)
        self.boton_capturar.setEnabled(True)
        self.etiqueta_resultado.setText("Estado: Listo para escanear.")
        self.etiqueta_resultado.setStyleSheet("background-color: #f0f0f0; color: black; padding: 10px; border-radius: 5px;")

    def analizar_lesion(self):
        if self.imagen_actual is None or self.sesion_ia is None:
            self.etiqueta_resultado.setText("Error: Imagen o Modelo no disponibles.")
            return

        self.etiqueta_resultado.setText("Procesando con IA...")
        QApplication.processEvents() # Forzar actualizacion de interfaz

        # --- PREPROCESAMIENTO ---
        # 1. Redimensionar a 28x28 (Tamaño de MedMNIST)
        # OJO: Si entrenaste con 224x224 cambia esto. MedMNIST nativo es 28x28.
        # Basado en tu export_onnx.py pusimos dummy_input 28x28.
        img = cv2.resize(self.imagen_actual, (28, 28))
        
        # 2. Normalizar (0 a 1) y estandarizar (mean=0.5, std=0.5)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        # 3. Transponer (Alto, Ancho, Canales) -> (Canales, Alto, Ancho)
        img = img.transpose(2, 0, 1)
        
        # 4. Añadir dimension de Batch (1, 3, 28, 28)
        img = np.expand_dims(img, axis=0)

        # --- INFERENCIA ---
        try:
            nombre_input = self.sesion_ia.get_inputs()[0].name
            # Ejecutamos el modelo ONNX
            resultados = self.sesion_ia.run(None, {nombre_input: img})
            
            # resultados[0] contiene los "logits" (valores crudos)
            logits = resultados[0][0]
            
            # --- POST-PROCESAMIENTO ---
            # Aplicamos Softmax matematico para obtener porcentajes
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            probabilidades = softmax(logits)
            
            # Obtenemos el indice de la clase ganadora
            indice_ganador = np.argmax(probabilidades)
            confianza = probabilidades[indice_ganador] * 100
            
            nombre_clase = self.clases.get(indice_ganador, "Desconocido")
            
            # --- MOSTRAR RESULTADO ---
            mensaje = f"DIAGNOSTICO: {nombre_clase}\nConfianza: {confianza:.1f}%"
            
            # Logica de Colores: Rojo si es Melanoma (Clase 4) o Basocelular (Clase 1)
            if indice_ganador in [1, 4]:
                estilo = "background-color: #ffcccc; color: #cc0000; border: 2px solid red;"
                mensaje = "ALERTA: " + mensaje
            else:
                estilo = "background-color: #ccffcc; color: #006600; border: 2px solid green;"
            
            self.etiqueta_resultado.setText(mensaje)
            self.etiqueta_resultado.setStyleSheet(estilo + " padding: 10px; border-radius: 5px; font-weight: bold;")

        except Exception as e:
            self.etiqueta_resultado.setText(f"Error en inferencia: {e}")
            print(e)

    def closeEvent(self, event):
        # Liberar la camara al cerrar la ventana para no bloquearla
        if self.captura.isOpened():
            self.captura.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaDermatoscopio()
    ventana.show()
    sys.exit(app.exec())