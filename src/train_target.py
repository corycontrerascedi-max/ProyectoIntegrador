import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Importamos la arquitectura
from model import Net

# --- CONFIGURACIÓN ---
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 15 # Aumenté un poco las épocas porque ahora es más difícil aprender
# (Nota: Puedes bajarlo a 10 si tienes prisa, pero 15 dará mejor resultado)

# --- RUTAS DINÁMICAS ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(base_dir, 'assets', 'models', 'modelo_base_colon.pth')
TARGET_PATH = os.path.join(base_dir, 'assets', 'models', 'modelo_final_piel.pth')

def plot_training_results(loss_list, acc_list):
    """Genera la gráfica de curvas de aprendizaje"""
    plt.figure(figsize=(12, 5))
    
    # Gráfica de Loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Train Loss', color='red')
    plt.title('Curva de Pérdida (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label='Train Accuracy', color='blue')
    plt.title('Curva de Precisión (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.savefig('training_curves.png')
    print("📈 Gráficas de entrenamiento guardadas como 'training_curves.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Genera la Matriz de Confusión bonita"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicción de la IA')
    plt.ylabel('Realidad (Biopsia)')
    plt.title('Matriz de Confusión - Diagnóstico Dermatológico')
    plt.savefig('confusion_matrix.png')
    print("📊 Matriz de confusión guardada como 'confusion_matrix.png'")
    plt.close()

def train_target():
    print(f"--- INICIANDO FASE 2: TRANSFER LEARNING (CROSSPATH AI) ---")
    print(f"🔍 Buscando modelo base en: {SOURCE_PATH}")
    
    if not os.path.exists(SOURCE_PATH):
        print(f"❌ ERROR: No encuentro el archivo. Verifica la ruta.")
        return

    # Preparar Datos
    data_flag = 'dermamnist'
    info = INFO[data_flag]
    n_classes_piel = len(info['label'])
    
    print(f"📊 Dataset: {data_flag} | Clases destino: {n_classes_piel}")

    # ### --- CAMBIO 1: DATA AUGMENTATION ---
    # Hacemos el entrenamiento más difícil rotando y volteando imágenes
    # para que la IA no memorice posiciones.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # Voltear horizontalmente
        transforms.RandomVerticalFlip(),        # Voltear verticalmente
        transforms.RandomRotation(20),          # Rotar hasta 20 grados
        transforms.ColorJitter(brightness=0.1), # Jugar un poco con el brillo
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Para el test NO alteramos la imagen, solo la normalizamos
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = medmnist.DermaMNIST(split='train', transform=train_transform, download=True, root='./assets/datasets')
    test_dataset = medmnist.DermaMNIST(split='test', transform=test_transform, download=True, root='./assets/datasets')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- CARGAR EL CEREBRO ---
    print("🧠 Cargando modelo pre-entrenado...")
    model = Net(num_classes=9) 
    
    try:
        model.load_state_dict(torch.load(SOURCE_PATH))
        print("✅ Pesos de Colon cargados exitosamente.")
    except Exception as e:
        print(f"❌ Error al cargar pesos: {e}")
        return

    # Congelar capas (Freeze)
    for param in model.parameters():
        param.requires_grad = False

    # --- CIRUGÍA DE MODELO ---
    print(f"🔧 Reemplazando capa final 'fc2' (9 -> {n_classes_piel} neuronas)...")
    num_ftrs = model.fc2.in_features
    model.fc2 = nn.Linear(num_ftrs, n_classes_piel)

    # Entrenamos solo lo que no está congelado
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # ### --- CAMBIO 2: CLASS WEIGHTS (CASTIGO PONDERADO) ---
    print("⚖️ Calculando pesos para balancear el castigo...")
    
    # Extraemos todas las etiquetas del entrenamiento para contarlas
    # Nota: DermaMNIST guarda las etiquetas en .labels (array numpy)
    # Si falla, usamos el método de iteración, pero este debería ser directo.
    try:
        targets_np = train_dataset.labels.squeeze()
    except:
        # Método alternativo seguro si .labels falla
        temp_list = []
        for _, t in train_dataset:
            temp_list.append(t[0])
        targets_np = np.array(temp_list)

    class_counts = np.bincount(targets_np)
    
    # Fórmula: Peso = Total / (Num_Clases * Cantidad_de_esa_Clase)
    # Resultado: Las clases raras tienen números más grandes
    total_samples = sum(class_counts)
    class_weights = total_samples / (n_classes_piel * class_counts)
    
    # Convertimos a Tensor
    weights_tensor = torch.FloatTensor(class_weights)
    print(f"   Pesos asignados: {np.round(class_weights, 2)}")
    print("   (Las clases con números altos son las que la IA priorizará)")

    # Aplicamos los pesos al criterio
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    # ### ----------------------------------------------------

    loss_history = []
    acc_history = []

    print("\n🚀 --- ENTRENANDO (VERSIÓN BALANCEADA) ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze().long()
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss/len(train_loader)
        epoch_acc = 100. * correct / total
        
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        print(f"   Epoch [{epoch+1}/{EPOCHS}] -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # Guardar
    torch.save(model.state_dict(), TARGET_PATH)
    print(f"\n💾 Modelo guardado en: {TARGET_PATH}")
    
    # Gráficas
    print("\n🎨 Generando gráficas científicas...")
    plot_training_results(loss_history, acc_history)

    print("🔍 Generando Matriz de Confusión...")
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(targets.squeeze().numpy())
            
    # Etiquetas reales de DermaMNIST
    labels_humanos = ['Actinic', 'BCC', 'Benign', 'Dermato', 'Melanoma', 'Nevus', 'Vasc']
    plot_confusion_matrix(all_targets, all_preds, labels_humanos)

    print("\n✨ ¡FASE 2 COMPLETADA! Revisa 'confusion_matrix.png'.")

if __name__ == '__main__':
    train_target()