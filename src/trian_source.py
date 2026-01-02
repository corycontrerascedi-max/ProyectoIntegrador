import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import medmnist
from medmnist import INFO
from model import Net
import os

def train_source():
    print("--- Entrenando Modelo Base (Colon) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preparar datos
    info = INFO['pathmnist']
    transform = medmnist.transforms.ToTensor()
    root = './assets/datasets'
    
    # Si no existen los datos, los descarga
    if not os.path.exists(root):
        os.makedirs(root)
        
    train_dataset = medmnist.PathMNIST(split='train', transform=transform, download=True, root=root)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    # Configurar modelo
    model = Net(num_classes=len(info['label'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento rápido (3 épocas)
    for epoch in range(3):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.squeeze().long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Época {epoch+1} completada.")

    # Guardar
    if not os.path.exists('./assets/models'):
        os.makedirs('./assets/models')
    torch.save(model.state_dict(), './assets/models/modelo_base_colon.pth')
    print("✅ Modelo guardado.")

if __name__ == '__main__':
    train_source()