import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from app.services.model import load_model


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train_model():
    start_time = time.time()
    train_dataset = datasets.ImageFolder(root="datasets/train", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root="datasets/valid", transform=predict_transforms)
    test_dataset  = datasets.ImageFolder(root="datasets/test",  transform=predict_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

    
    model = load_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    _device_env = os.getenv("TORCH_DEVICE", "").strip().lower()
    if _device_env:
        device = torch.device(_device_env)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")      # NVIDIA GPU
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)
    

    num_epochs = 10
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                valid_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_valid_loss = valid_loss / len(valid_loader)
        accuracy = correct / total * 100

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    # Test evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total * 100
    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), "banana_clock_model.pth")
    print("Model saved as banana_clock_model.pth")

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"Training completed in {minutes}m {seconds}s")
