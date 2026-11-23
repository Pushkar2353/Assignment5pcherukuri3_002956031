import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torch.utils.data import (
    DataLoader,
    Subset,
)

from torchvision import (
    datasets,
    transforms,
)


DATA_DIR = (
    "/mnt/shared/mnist_data"
)

METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "problem2_pytorch_metrics.txt"
)


# ============================================================
# CNN
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
        )


        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
        )


        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128,
        )


        self.fc2 = nn.Linear(
            128,
            10,
        )


    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = F.max_pool2d(
            x,
            2,
        )


        x = self.conv2(x)

        x = F.relu(x)

        x = F.max_pool2d(
            x,
            2,
        )


        x = torch.flatten(
            x,
            1,
        )


        x = self.fc1(x)

        x = F.relu(x)


        x = self.fc2(x)


        return x


# ============================================================
# Dataset
# ============================================================

transform = (
    transforms.ToTensor()
)


full_train_dataset = (
    datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )
)


test_dataset = (
    datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )
)


# ============================================================
# Use 10% of MNIST
# ============================================================

percent = 0.1


subset_size = math.ceil(
    len(full_train_dataset)
    * percent
)


torch.manual_seed(42)


subset_indices = torch.randperm(
    len(full_train_dataset)
)[:subset_size]


small_train_dataset = Subset(
    full_train_dataset,
    subset_indices,
)


print(
    "Original training images:",
    len(full_train_dataset),
)

print(
    "Training images used:",
    len(small_train_dataset),
)

print(
    "Testing images:",
    len(test_dataset),
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    small_train_dataset,
    batch_size=64,
    shuffle=True,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)


# ============================================================
# CPU model
# ============================================================

device = "cpu"


model = SimpleCNN().to(
    device
)


print(
    "\nPyTorch MNIST CNN"
)

print(
    model
)


loss_fn = (
    nn.CrossEntropyLoss()
)


optimizer = Adam(
    model.parameters(),
    lr=0.001,
)


# ============================================================
# Training
# ============================================================

epochs = 10


for epoch in range(
    epochs
):

    model.train()


    for data, target in train_loader:

        data = data.to(
            device
        )

        target = target.to(
            device
        )


        output = model(
            data
        )


        loss = loss_fn(
            output,
            target,
        )


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()


    print(
        f"Epoch {epoch + 1:2d} - "
        f"Loss: {loss.item():.6f}"
    )


# ============================================================
# Evaluation
# ============================================================

model.eval()


correct = 0
total = 0


with torch.no_grad():

    for data, target in test_loader:

        data = data.to(
            device
        )

        target = target.to(
            device
        )


        output = model(
            data
        )


        prediction = (
            output.argmax(
                dim=1
            )
        )


        correct += (
            prediction
            == target
        ).sum().item()


        total += (
            target.size(0)
        )


test_accuracy = (
    correct
    / total
)


print(
    "\n================================"
)

print(
    "PyTorch MNIST Results"
)

print(
    "================================"
)

print(
    "Training images:",
    len(small_train_dataset),
)

print(
    "Testing images:",
    len(test_dataset),
)

print(
    "Test Accuracy:",
    f"{test_accuracy:.4f}",
)


# ============================================================
# Save metrics
# ============================================================

with open(
    METRICS_FILE,
    "w",
) as file:

    file.write(
        "PyTorch MNIST CNN\n"
    )

    file.write(
        f"Training images: "
        f"{len(small_train_dataset)}\n"
    )

    file.write(
        f"Testing images: "
        f"{len(test_dataset)}\n"
    )

    file.write(
        f"Epochs: {epochs}\n"
    )

    file.write(
        "Batch size: 64\n"
    )

    file.write(
        "Conv1: 16 filters, "
        "3x3, stride 1, padding 1\n"
    )

    file.write(
        "Conv2: 32 filters, "
        "3x3, stride 1, padding 1\n"
    )

    file.write(
        f"Final training loss: "
        f"{loss.item():.6f}\n"
    )

    file.write(
        f"Test accuracy: "
        f"{test_accuracy:.6f}\n"
    )


print(
    "\nCreated:",
    METRICS_FILE,
)
