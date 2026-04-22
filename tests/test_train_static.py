"""
Static data smoke-test for the training pipeline.
Uses synthetic random tensors — no real images required.
Run: python -m pytest tests/test_train_static.py -v
  or: python tests/test_train_static.py
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.services.model import load_model, NUM_CLASSES


def make_fake_loader(num_samples=32, batch_size=8):
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def test_train_static():
    train_loader = make_fake_loader()
    valid_loader = make_fake_loader(num_samples=16)
    test_loader  = make_fake_loader(num_samples=16)

    model = load_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    device = torch.device("cpu")
    model.to(device)

    training_losses   = []
    validation_losses = []
    test_losses       = []

    num_epochs = 2  # keep it fast
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            training_losses.append(loss.item())
        avg_train = train_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                valid_loss += loss.item()
                validation_losses.append(loss.item())
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_valid = valid_loss / len(valid_loader)
        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train:.4f} | Valid Loss: {avg_valid:.4f} | Acc: {acc:.1f}%")

    # Test phase
    model.eval()
    test_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            test_loss += loss.item()
            test_losses.append(loss.item())
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_test = test_loss / len(test_loader)
    test_acc = correct / total * 100
    print(f"Test Loss: {avg_test:.4f} | Test Accuracy: {test_acc:.1f}%")

    # Plot
    os.makedirs("graphs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses,   label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.plot(test_losses,       label='Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Static Data Smoke-Test — Loss Over Time')
    plt.legend()
    plt.grid()
    plt.savefig("graphs/static_test_loss.png")
    plt.close()
    print("Plot saved → graphs/static_test_loss.png")

    assert avg_test > 0, "Test loss should be positive"
    print("All assertions passed ✓")


if __name__ == "__main__":
    test_train_static()
