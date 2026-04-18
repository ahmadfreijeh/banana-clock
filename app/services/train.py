import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from app.services.model import load_model


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def train_model():
    dataset = datasets.ImageFolder(
    root="datasets/v1",
    transform=train_transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    print("Loading model...")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Total images: {len(dataset)}")
    print(f"Classes found: {dataset.classes}")


    model = load_model()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    # Save the trained model
    torch.save(model.state_dict(), "banana_clock_model.pth")
    print("Model trained and saved as banana_clock_model.pth")


if __name__ == "__main__":
    train_model()