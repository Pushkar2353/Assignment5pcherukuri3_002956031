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


# ============================================================
# File paths
# ============================================================

INPUT_FILE = "/mnt/shared/Assignment5/iris.csv"

OUTPUT_FILE = (
    "/mnt/shared/Assignment5/"
    "problem1_pytorch_ray_predictions.csv"
)

METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "problem1_pytorch_ray_metrics.txt"
)


# ============================================================
# Iris labels
# ============================================================

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


# ============================================================
# Load Iris dataset
#
# Assignment requirement:
# Random 80% training / 20% testing split
# ============================================================

def load_iris():

    features = []
    labels = []

    with open(INPUT_FILE, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            features.append(
                [
                    float(row["sepal_length"]),
                    float(row["sepal_width"]),
                    float(row["petal_length"]),
                    float(row["petal_width"]),
                ]
            )

            labels.append(
                SPECIES_TO_LABEL[
                    row["species"]
                ]
            )


    features = np.array(
        features,
        dtype=np.float32,
    )

    labels = np.array(
        labels,
        dtype=np.int64,
    )


    # Random 80/20 split.
    np.random.seed(42)

    indices = np.arange(
        len(features)
    )

    np.random.shuffle(
        indices
    )


    split = int(
        0.8 * len(indices)
    )


    train_indices = indices[:split]
    test_indices = indices[split:]


    x_train = features[
        train_indices
    ]

    y_train = labels[
        train_indices
    ]

    x_test = features[
        test_indices
    ]

    y_test = labels[
        test_indices
    ]


    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


# ============================================================
# Feedforward Neural Network
#
# Required architecture:
#
# Input:
#   4 Iris features
#
# Hidden Layer 1:
#   5 neurons
#
# Hidden Layer 2:
#   6 neurons
#
# Output:
#   3 Iris classes
# ============================================================

class IrisNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                4,
                5,
            ),

            nn.ReLU(),

            nn.Linear(
                5,
                6,
            ),

            nn.ReLU(),

            nn.Linear(
                6,
                3,
            ),
        )


    def forward(self, x):

        return self.network(x)


# ============================================================
# Ray Train worker function
# ============================================================

def train_func(config):

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_iris()


    # --------------------------------------------------------
    # Ray worker information
    # --------------------------------------------------------

    context = train.get_context()

    world_size = (
        context.get_world_size()
    )

    world_rank = (
        context.get_world_rank()
    )


    print(
        "Ray Train world size:",
        world_size,
    )

    print(
        "Ray Train world rank:",
        world_rank,
    )

    print(
        "Training records:",
        len(x_train),
    )

    print(
        "Testing records:",
        len(x_test),
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = IrisNetwork()


    # --------------------------------------------------------
    # Prepare model for distributed Ray execution.
    #
    # Ray wraps the model in DistributedDataParallel when
    # more than one training worker is used.
    # --------------------------------------------------------

    model = (
        ray.train.torch
        .prepare_model(
            model
        )
    )


    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    loss_fn = (
        nn.CrossEntropyLoss()
    )


    optimizer = Adam(
        model.parameters(),
        lr=0.01,
    )


    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )


    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long,
    )


    train_dataset = TensorDataset(
        x_train_tensor,
        y_train_tensor,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )


    # --------------------------------------------------------
    # Prepare DataLoader for distributed training.
    #
    # Ray adds a DistributedSampler when world_size > 1.
    # --------------------------------------------------------

    train_loader = (
        ray.train.torch
        .prepare_data_loader(
            train_loader
        )
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    epochs = config.get(
        "epochs",
        100,
    )


    for epoch in range(
        epochs
    ):

        # Required for distributed shuffling.
        if world_size > 1:

            train_loader.sampler.set_epoch(
                epoch
            )


        model.train()


        for X, y in train_loader:

            outputs = model(X)


            loss = loss_fn(
                outputs,
                y,
            )


            optimizer.zero_grad()


            loss.backward()


            optimizer.step()


        if (
            epoch == 0
            or
            (epoch + 1) % 10 == 0
        ):

            print(
                f"Rank {world_rank} - "
                f"Epoch {epoch + 1:3d} - "
                f"Loss: {loss.item():.6f}"
            )


    # --------------------------------------------------------
    # Test model
    # --------------------------------------------------------

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

        outputs = model(
            x_test_tensor
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        correct = (
            predictions
            == y_test_tensor
        ).sum().item()


        test_accuracy = (
            correct
            / len(
                y_test_tensor
            )
        )


    print(
        "\nPyTorch + Ray Iris "
        "Test Accuracy:",
        f"{test_accuracy:.4f}",
    )


    # --------------------------------------------------------
    # Only Ray worker rank 0 writes the final files.
    #
    # This prevents two workers from writing to the same
    # shared file simultaneously.
    # --------------------------------------------------------

    if world_rank == 0:

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow(
                [
                    "actual_species",
                    "predicted_species",
                    "petal_length",
                    "petal_width",
                ]
            )


            for i in range(
                len(y_test)
            ):

                actual_species = (
                    LABEL_TO_SPECIES[
                        int(
                            y_test[i]
                        )
                    ]
                )


                predicted_species = (
                    LABEL_TO_SPECIES[
                        int(
                            predictions[i]
                            .item()
                        )
                    ]
                )


                writer.writerow(
                    [
                        actual_species,
                        predicted_species,

                        float(
                            x_test[i][2]
                        ),

                        float(
                            x_test[i][3]
                        ),
                    ]
                )


        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        with open(
            METRICS_FILE,
            "w",
        ) as file:

            file.write(
                "PyTorch + Ray "
                "Iris Classification\n"
            )


            file.write(
                "Architecture: "
                "4 -> 5 -> 6 -> 3\n"
            )


            file.write(
                "Hidden Layer 1: "
                "5 neurons\n"
            )


            file.write(
                "Hidden Layer 2: "
                "6 neurons\n"
            )


            file.write(
                "Training split: 80%\n"
            )


            file.write(
                "Testing split: 20%\n"
            )


            file.write(
                f"Training records: "
                f"{len(x_train)}\n"
            )


            file.write(
                f"Testing records: "
                f"{len(x_test)}\n"
            )


            file.write(
                f"Ray Train workers: "
                f"{world_size}\n"
            )


            file.write(
                "GPU used: False\n"
            )


            file.write(
                f"Epochs: "
                f"{epochs}\n"
            )


            file.write(
                "Batch size: 16\n"
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
            OUTPUT_FILE,
        )


        print(
            "Created:",
            METRICS_FILE,
        )


    # --------------------------------------------------------
    # All workers must call train.report() the same number
    # of times.
    #
    # Checkpoint is optional according to the Ray lecture.
    #
    # We report only simple scalar metrics here.
    # The driver does NOT try to read result.metrics later.
    # --------------------------------------------------------

    train.report(
        {
            "test_accuracy":
            float(
                test_accuracy
            ),

            "loss":
            float(
                loss.item()
            ),

            "training_records":
            int(
                len(x_train)
            ),

            "testing_records":
            int(
                len(x_test)
            ),
        }
    )


# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Connect to the existing Ray cluster.
    # --------------------------------------------------------

    ray.init(
        address="auto"
    )


    print(
        "Connected to Ray cluster."
    )


    # --------------------------------------------------------
    # Ray Train scaling
    #
    # Assignment requirement:
    # 2 CPU training workers
    # --------------------------------------------------------

    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=False,
    )


    # --------------------------------------------------------
    # Create TorchTrainer
    # --------------------------------------------------------

    trainer = TorchTrainer(

        train_loop_per_worker=(
            train_func
        ),

        train_loop_config={
            "epochs": 100,
        },

        scaling_config=(
            scaling_config
        ),

        run_config=RunConfig(
            storage_path="/mnt/shared"
        ),
    )


    # --------------------------------------------------------
    # Run distributed training.
    #
    # Worker rank 0 already writes the final CSV and metrics
    # to /mnt/shared/Assignment5.
    # --------------------------------------------------------

    result = trainer.fit()


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT use:
    #
    # metrics = result.metrics
    # metrics["test_accuracy"]
    #
    # because your installed Ray version is returning None
    # for these free-floating reported metrics.
    #
    # The final output is already written by rank 0.
    # --------------------------------------------------------

    print(
        "\n========================================"
    )


    print(
        "PyTorch + Ray Iris training "
        "completed successfully."
    )


    print(
        "========================================"
    )


    print(
        "Prediction file:",
        OUTPUT_FILE,
    )


    print(
        "Metrics file:",
        METRICS_FILE,
    )


    ray.shutdown()
