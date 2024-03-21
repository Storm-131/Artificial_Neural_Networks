# ---------------------------------------------------------*\
# Title: ANN (PyTorch) 🔥
# ---------------------------------------------------------*/

# PyTorch 🔥
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize

# Utils 🛠
from utils.pred import predict_digit


# ---------------------------------------------------------*/
# Load Data (Mnist) 🌊
# ---------------------------------------------------------*/
def create_data_loaders():
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, persistent_workers=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, persistent_workers=True, num_workers=8)

    return train_loader, test_loader

# ---------------------------------------------------------*/
# Define Network 🧠
# ---------------------------------------------------------*/


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------*/
# Train und Test model
# ---------------------------------------------------------*/


def train_model(train_loader, model, criterion, optimizer):
    for epoch in range(5):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  # Print every 500 batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 500:.3f}')
                running_loss = 0.0

        print(f"Epoch {epoch+1} ended successfully ⭐️")

    print('Finished Training')


def test_model(test_loader, model):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# ---------------------------------------------------------*/
# Main 🚀
# ---------------------------------------------------------*/


def main():

    # Load the Data
    train_loader, test_loader = create_data_loaders()

    # Create a model
    model = Network()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    train_model(train_loader, model, criterion, optimizer)
    test_model(test_loader, model)

    # Predict a digit
    predict_digit("data/samples/2.png", model)

if __name__ == "__main__":
    main()

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
