import csv
import numpy as np

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


INPUT_FILE = "/mnt/shared/Assignment5/iris.csv"

OUTPUT_FILE = (
    "/mnt/shared/Assignment5/"
    "pytorch_ray_iris_predictions.csv"
)

METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "pytorch_ray_iris_metrics.txt"
)


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


def train_func(config):

    x_train, y_train, x_test, y_test = (
        load_iris()
    )


    model = IrisNetwork()


    # Ray lecture method.
    model = ray.train.torch.prepare_model(
        model
    )


    loss_fn = nn.CrossEntropyLoss()


    optimizer = Adam(
        model.parameters(),
        lr=0.01,
    )


    dataset = TensorDataset(
        torch.tensor(
            x_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.long,
        ),
    )


    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )


    # Ray lecture method.
    loader = (
        ray.train.torch
        .prepare_data_loader(loader)
    )


    for epoch in range(
        config["epochs"]
    ):

        if (
            ray.train.get_context()
            .get_world_size()
            > 1
        ):

            loader.sampler.set_epoch(
                epoch
            )


        model.train()


        for X, y in loader:

            output = model(X)

            loss = loss_fn(
                output,
                y,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


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

        output = model(
            x_test_tensor
        )

        predictions = output.argmax(
            dim=1
        )

        correct = (
            predictions
            == y_test_tensor
        ).sum().item()

        accuracy = (
            correct
            / len(y_test_tensor)
        )


    train.report({
        "test_accuracy": float(accuracy),
        "actual_labels": y_test.tolist(),
        "predicted_labels": predictions.tolist(),
        "petal_length": x_test[:, 2].tolist(),
        "petal_width": x_test[:, 3].tolist(),
    })


if __name__ == "__main__":

    ray.init(
        address="auto"
    )


    trainer = TorchTrainer(
        train_func,

        train_loop_config={
            "epochs": 100,
        },

        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False,
        ),

        run_config=RunConfig(
            storage_path="/mnt/shared"
        ),
    )


    result = trainer.fit()


    metrics = result.metrics


    accuracy = float(
        metrics["test_accuracy"]
    )


    print(
        "\nPyTorch + Ray Iris "
        "Test Accuracy:",
        f"{accuracy:.4f}",
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

        for i in range(
            len(metrics["actual_labels"])
        ):

            writer.writerow([
                LABEL_TO_SPECIES[
                    int(
                        metrics[
                            "actual_labels"
                        ][i]
                    )
                ],

                LABEL_TO_SPECIES[
                    int(
                        metrics[
                            "predicted_labels"
                        ][i]
                    )
                ],

                metrics[
                    "petal_length"
                ][i],

                metrics[
                    "petal_width"
                ][i],
            ])


    with open(
        METRICS_FILE,
        "w",
    ) as file:

        file.write(
            f"Test accuracy: "
            f"{accuracy:.6f}\n"
        )


    ray.shutdown()
