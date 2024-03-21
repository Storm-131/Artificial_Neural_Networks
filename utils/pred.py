#---------------------------------------------------------*\
# Title: Testing the model
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def predict_digit(image_path, model):
    # Preprocess the image
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image)

    # plt.imshow(image.numpy().squeeze(), cmap='gray')

    model.eval()
    output = model(image.unsqueeze(0))

    print(f"Probabilities for all digits: {image_path}\n")

    with torch.no_grad():
        prediction = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)
        for i, prob in enumerate(probs.squeeze()):            
            print(f"Digit {i}: {prob:.2%}")
        print(f"\nPredicted digit: {prediction}")

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\