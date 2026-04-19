import torch
from torchvision import transforms
from PIL import Image
from app.services.model import load_model, CLASS_NAMES

predict_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_days_estimate(class_name):
    estimates = {
        'unripe':   '5-7 days until ripe, 12-14 days until inedible',
        'ripe':     'Perfect now! 4-6 days until overripe',
        'overripe': '1-2 days left, eat soon!',
        'rotten':   'Too late! Time to throw it away'
    }
    return estimates[class_name]

def predict(image: Image.Image):
    # Load the trained model
    model = load_model()
    model.load_state_dict(torch.load("banana_clock_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Preprocess the input image
    input_tensor = predict_transforms(image).unsqueeze(0)

   # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted = torch.argmax(outputs.logits, dim=1).item()

    print(f"Predicted class index: {predicted}, class name: {CLASS_NAMES[predicted]}")
    
    # Return result
    class_name = CLASS_NAMES[predicted]
    days = get_days_estimate(class_name)

    # Loss > 1.0   → bad
    # Loss 0.5-1.0 → okay
    # Loss 0.1-0.5 → good
    # Loss < 0.1   → very good
    
    return {
        'ripeness': class_name,
        'days_until_inedible': days
    }