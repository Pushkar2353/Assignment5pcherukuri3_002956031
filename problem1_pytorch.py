import csv
import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


INPUT_FILE = "/mnt/shared/Assignment5/iris.csv"
OUTPUT_FILE = "/mnt/shared/Assignment5/pytorch_iris_predictions.csv"
METRICS_FILE = "/mnt/shared/Assignment5/pytorch_iris_metrics.txt"


SPECIES_TO_LABEL = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}

LABEL_TO_SPECIES = [
    "setosa",
    "versicolor",
    "virginica",
]


def load_iris():

    features = []
    labels = []

    with open(INPUT_FILE, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            features.append([
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
            ])

            labels.append(
                SPECIES_TO_LABEL[row["species"]]
            )

    features = np.array(
        features,
        dtype=np.float32,
    )

    labels = np.array(
        labels,
        dtype=np.int64,
    )

    np.random.seed(42)

    indices = np.arange(len(features))

    np.random.shuffle(indices)

    split = int(0.8 * len(indices))

    train_indices = indices[:split]
    test_indices = indices[split:]

    return (
        features[train_indices],
        labels[train_indices],
        features[test_indices],
        labels[test_indices],
    )


class IrisNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),

            nn.Linear(5, 6),
            nn.ReLU(),

            nn.Linear(6, 3),
        )

    def forward(self, x):

        return self.network(x)


x_train, y_train, x_test, y_test = load_iris()


print("Training records:", len(x_train))
print("Testing records:", len(x_test))


train_dataset = TensorDataset(
    torch.tensor(
        x_train,
        dtype=torch.float32,
    ),
    torch.tensor(
        y_train,
        dtype=torch.long,
    ),
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)


model = IrisNetwork()


print("\nPyTorch Iris Model")
print(model)


loss_fn = nn.CrossEntropyLoss()


optimizer = Adam(
    model.parameters(),
    lr=0.01,
)


for epoch in range(100):

    model.train()

    for X, y in train_loader:

        output = model(X)

        loss = loss_fn(
            output,
            y,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch == 0 or (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:3d} "
            f"Loss: {loss.item():.6f}"
        )


model.eval()


x_test_tensor = torch.tensor(
    x_test,
    dtype=torch.float32,
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long,
)


with torch.no_grad():

    output = model(x_test_tensor)

    predictions = output.argmax(
        dim=1
    )

    correct = (
        predictions
        == y_test_tensor
    ).sum().item()

    test_accuracy = (
        correct
        / len(y_test_tensor)
    )


print(
    "\nPyTorch Iris Test Accuracy:",
    f"{test_accuracy:.4f}",
)


with open(
    OUTPUT_FILE,
    "w",
    newline="",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "actual_species",
        "predicted_species",
        "petal_length",
        "petal_width",
    ])

    for i in range(len(y_test)):

        writer.writerow([
            LABEL_TO_SPECIES[int(y_test[i])],
            LABEL_TO_SPECIES[int(predictions[i])],
            float(x_test[i][2]),
            float(x_test[i][3]),
        ])


with open(METRICS_FILE, "w") as file:

    file.write(
        f"Training records: {len(x_train)}\n"
    )

    file.write(
        f"Testing records: {len(x_test)}\n"
    )

    file.write(
        f"Test accuracy: {test_accuracy:.6f}\n"
    )


print("Created:", OUTPUT_FILE)
print("Created:", METRICS_FILE)
