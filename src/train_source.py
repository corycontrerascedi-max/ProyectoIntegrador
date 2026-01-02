import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms  # <--- CORRECCIÓN AQUÍ
import medmnist
from medmnist import INFO
from model import Net
import os

def train_source():
    print("--- Entrenando Modelo Base (Colon) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Preparar datos
    info = INFO['pathmnist']
    
    # CORRECCIÓN: Usamos transforms de torchvision
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    root = './assets/datasets'
    
    if not os.path.exists(root):
        os.makedirs(root)
        
    print("Cargando dataset...")
    # Descargamos si no existe
    train_dataset = medmnist.PathMNIST(split='train', transform=data_transform, download=True, root=root)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # Configurar modelo
    print("Iniciando modelo...")
    model = Net(num_classes=len(info['label'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento rápido (3 épocas)
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze().long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoca {epoch+1}/{EPOCHS} -> Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    # Guardar
    if not os.path.exists('./assets/models'):
        os.makedirs('./assets/models')
    torch.save(model.state_dict(), './assets/models/modelo_base_colon.pth')
    print("Modelo guardado.")

if __name__ == '__main__':
    train_source()