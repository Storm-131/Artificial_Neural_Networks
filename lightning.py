# ---------------------------------------------------------*\
# Title: ANN (PyTorch Lightning) ⚡️
# ---------------------------------------------------------*/

# PyTorch 🔥
import torch
from torch.utils.data import DataLoader
from torch.nn import Sequential, Flatten, Linear, ReLU, CrossEntropyLoss
from torchvision import datasets, transforms
from torchmetrics.functional import accuracy

# Lightning ⚡️
import pytorch_lightning as pl

# Utils 🛠
from utils.pred import predict_digit

# ---------------------------------------------------------*/
# Load Data (Mnist) 🌊
# ---------------------------------------------------------*/


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)

    def setup(self, stage=None):
        self.train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True,
                                       persistent_workers=True, num_workers=8)
        self.test_loader = DataLoader(self.test_data, batch_size=32, shuffle=False,
                                      persistent_workers=True, num_workers=8)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

# ---------------------------------------------------------*/
# Define Network 🧠
# ---------------------------------------------------------*/


class NetWork(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Sequential(
            Flatten(),
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = CrossEntropyLoss()(logits, labels)
        acc = accuracy(logits, labels, task="multiclass", num_classes=10)
        self.log('train/loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log('train/acc', acc, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = CrossEntropyLoss()(logits, labels)
        acc = accuracy(logits, labels, task="multiclass", num_classes=10)
        self.log("test/loss_epoch", loss, on_epoch=True, prog_bar=True)
        self.log("test/acc_epoch", acc, on_epoch=True, prog_bar=True)


# ---------------------------------------------------------*/
# Main 🚀
# ---------------------------------------------------------*/

def main():

    # Load the Data
    data = DataModule()
    data.prepare_data()
    data.setup()

    # Create a model and trainer
    model = NetWork()
    trainer = pl.Trainer(max_epochs=5, logger=False, enable_checkpointing=False)

    # Train the model
    trainer.fit(model, data)

    # Test the model
    trainer.test(model, data)

    # Predict a digit
    predict_digit("data/samples/2.png", model)


if __name__ == '__main__':
    main()

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
