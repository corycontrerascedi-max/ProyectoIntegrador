import sys
# Importamos las herramientas graficas de Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
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
        
        # 🔴 CAMBIO: Fondo BLANCO y letra NEGRA (Estilo Clinico)
        self.setStyleSheet("background-color: #ffffff; color: black;")

        # --- CARGA DEL MODELO ONNX ---
        self.sesion_ia = None
        try:
            self.sesion_ia = ort.InferenceSession("prototipo_pyqt.onnx")
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error cargando modelo: {e}")

        # --- CONFIGURACION DE LA CAMARA ---
        self.captura = cv2.VideoCapture(0)
        self.imagen_actual = None

        # --- DISEÑO DE LA INTERFAZ ---
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout()
        widget_central.setLayout(layout_principal)

        # --- COLUMNA IZQUIERDA: EL VIDEO ---
        self.etiqueta_video = QLabel("Iniciando Camara...")
        self.etiqueta_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 🔴 CAMBIO: Le ponemos borde gris suave y...
        # ¡IMPORTANTE! Fijamos el tamaño para que NO CREZCA sola
        self.etiqueta_video.setFixedSize(640, 480) 
        self.etiqueta_video.setStyleSheet("border: 2px solid #cccccc; background-color: #000;") # Fondo negro solo donde va el video
        
        layout_principal.addWidget(self.etiqueta_video)

        # --- COLUMNA DERECHA: LOS CONTROLES ---
        layout_botones = QVBoxLayout()
        
        # Titulo
        etiqueta_titulo = QLabel("Panel de Control")
        etiqueta_titulo.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        etiqueta_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 🔴 CAMBIO: Aseguramos que el titulo sea negro
        etiqueta_titulo.setStyleSheet("color: black;")
        layout_botones.addWidget(etiqueta_titulo)

        layout_botones.addStretch()

        # Boton 1: Capturar
        self.boton_capturar = QPushButton("CONGELAR IMAGEN")
        # 🔴 CAMBIO: Letra blanca sobre boton azul
        self.boton_capturar.setStyleSheet("background-color: #007acc; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_capturar.clicked.connect(self.congelar_imagen)
        layout_botones.addWidget(self.boton_capturar)

        # Boton 2: Analizar
        self.boton_analizar = QPushButton("ANALIZAR LESION")
        self.boton_analizar.setStyleSheet("background-color: #28a745; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_analizar.clicked.connect(self.analizar_lesion)
        self.boton_analizar.setEnabled(False)
        layout_botones.addWidget(self.boton_analizar)

        # Boton 3: Reiniciar
        self.boton_reset = QPushButton("NUEVA TOMA")
        self.boton_reset.setStyleSheet("background-color: #6c757d; color: white; padding: 15px; font-weight: bold; border-radius: 5px;")
        self.boton_reset.clicked.connect(self.reiniciar_camara)
        layout_botones.addWidget(self.boton_reset)

        layout_botones.addStretch()

        # Etiqueta de RESULTADO
        self.etiqueta_resultado = QLabel("Estado: Listo")
        self.etiqueta_resultado.setFont(QFont("Arial", 14))
        # 🔴 CAMBIO: Fondo gris clarito para el resultado
        self.etiqueta_resultado.setStyleSheet("background-color: #f0f0f0; color: black; padding: 10px; border-radius: 5px; border: 1px solid #ddd;")
        self.etiqueta_resultado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.etiqueta_resultado.setWordWrap(True)
        layout_botones.addWidget(self.etiqueta_resultado)

        layout_principal.addLayout(layout_botones)

        # --- TEMPORIZADOR ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_cuadro)
        self.timer.start(30)

    def actualizar_cuadro(self):
        ret, cuadro = self.captura.read()
        if ret:
            cuadro = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)
            self.imagen_actual = cuadro

            # Convertimos imagen para Qt
            alto, ancho, canales = cuadro.shape
            paso = canales * ancho
            imagen_qt = QImage(cuadro.data, ancho, alto, paso, QImage.Format.Format_RGB888)
            
            # Convertimos a mapa de bits
            mapa_bits = QPixmap.fromImage(imagen_qt)
            
            # 🔴 CAMBIO CRITICO: Escalamos DIRECTAMENTE al tamaño fijo (640x480)
            # Usamos IgnoreAspectRatio para llenar el cuadro o KeepAspectRatio para que no se deforme
            # Aquí usamos KeepAspectRatio para que el lunar no se vea "chato"
            self.etiqueta_video.setPixmap(mapa_bits.scaled(
                640, 480, # Tamaño fijo
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))

    def congelar_imagen(self):
        self.timer.stop()
        self.boton_analizar.setEnabled(True)
        self.boton_capturar.setEnabled(False)
        self.etiqueta_resultado.setText("Estado: Imagen congelada.")

    def reiniciar_camara(self):
        self.timer.start(30)
        self.boton_analizar.setEnabled(False)
        self.boton_capturar.setEnabled(True)
        self.etiqueta_resultado.setText("Estado: Listo para escanear.")
        self.etiqueta_resultado.setStyleSheet("background-color: #f0f0f0; color: black; padding: 10px; border-radius: 5px;")

    def analizar_lesion(self):
        if self.imagen_actual is None or self.sesion_ia is None:
            self.etiqueta_resultado.setText("Error: No hay imagen o modelo.")
            return

        self.etiqueta_resultado.setText("Procesando...")
        QApplication.processEvents()

        # Preprocesamiento
        img = cv2.resize(self.imagen_actual, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        try:
            nombre_input = self.sesion_ia.get_inputs()[0].name
            resultados = self.sesion_ia.run(None, {nombre_input: img})
            probabilidad = resultados[0][0] * 100
            
            if probabilidad > 50:
                mensaje = f"ALERTA: POSIBLE MELANOMA\nConfianza: {probabilidad:.1f}%"
                # Rojo suave
                self.etiqueta_resultado.setStyleSheet("background-color: #ffcccc; color: #cc0000; padding: 10px; border-radius: 5px; font-weight: bold; border: 1px solid red;")
            else:
                mensaje = f"DIAGNOSTICO: BENIGNO\nRiesgo: {probabilidad:.1f}%"
                # Verde suave
                self.etiqueta_resultado.setStyleSheet("background-color: #ccffcc; color: #006600; padding: 10px; border-radius: 5px; font-weight: bold; border: 1px solid green;")
            
            self.etiqueta_resultado.setText(mensaje)

        except Exception as e:
            self.etiqueta_resultado.setText(f"Error: {e}")

    def closeEvent(self, event):
        self.captura.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaDermatoscopio()
    ventana.show()
    sys.exit(app.exec())