import os

# Ray TensorFlow lecture uses legacy tf.keras for distributed training.
# tf_keras must be installed on both Ray nodes.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import numpy as np
import ray
import tensorflow as tf

from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.tensorflow import TensorflowTrainer


# ============================================================
# File paths
# ============================================================

INPUT_FILE = "/mnt/shared/Assignment5/iris.csv"

OUTPUT_FILE = (
    "/mnt/shared/Assignment5/"
    "tensorflow_ray_iris_predictions.csv"
)

METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "tensorflow_ray_iris_metrics.txt"
)


# ============================================================
# Iris class labels
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
# Randomly split dataset into:
# 80% training
# 20% testing
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
# Build Feedforward Neural Network
#
# Assignment-required architecture:
#
# Input layer:
#   4 Iris features
#
# Hidden layer 1:
#   5 neurons
#
# Hidden layer 2:
#   6 neurons
#
# Output layer:
#   3 Iris classes
# ============================================================

def build_model():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(4,)
            ),

            tf.keras.layers.Dense(
                5,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                6,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                3,
                activation="softmax",
            ),
        ]
    )

    return model


# ============================================================
# Ray Train worker function
#
# This function is executed by each Ray Train worker.
# ============================================================

def train_func(config):

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_iris()


    # --------------------------------------------------------
    # Obtain Ray distributed-worker information.
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
    # TensorFlow distributed strategy.
    #
    # This follows the TensorFlow + Ray Train lecture pattern.
    # --------------------------------------------------------

    strategy = (
        tf.distribute
        .MultiWorkerMirroredStrategy()
    )


    # --------------------------------------------------------
    # Build and compile model inside strategy.scope().
    # --------------------------------------------------------

    with strategy.scope():

        model = build_model()


        model.compile(
            optimizer="adam",

            loss=(
                "sparse_categorical_"
                "crossentropy"
            ),

            metrics=[
                "accuracy"
            ],
        )


    # --------------------------------------------------------
    # Batch size
    #
    # Lecture:
    #
    # global_batch_size =
    # per_worker_batch_size * number_of_workers
    # --------------------------------------------------------

    per_worker_batch_size = (
        config.get(
            "batch_size",
            16,
        )
    )


    global_batch_size = (
        per_worker_batch_size
        * world_size
    )


    print(
        "Per-worker batch size:",
        per_worker_batch_size,
    )

    print(
        "Global batch size:",
        global_batch_size,
    )


    # --------------------------------------------------------
    # TensorFlow Dataset
    # --------------------------------------------------------

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_train,
                y_train,
            )
        )
        .shuffle(
            len(x_train)
        )
        .batch(
            global_batch_size
        )
    )


    # --------------------------------------------------------
    # Train network
    # --------------------------------------------------------

    epochs = config.get(
        "epochs",
        100,
    )


    print(
        "\nStarting TensorFlow + Ray "
        "Iris training..."
    )


    model.fit(
        train_dataset,
        epochs=epochs,
        verbose=2,
    )


    # --------------------------------------------------------
    # Evaluate trained network
    # --------------------------------------------------------

    test_loss, test_accuracy = (
        model.evaluate(
            x_test,
            y_test,
            verbose=0,
        )
    )


    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    probabilities = model.predict(
        x_test,
        verbose=0,
    )


    predictions = np.argmax(
        probabilities,
        axis=1,
    )


    print(
        "\nTensorFlow + Ray Iris "
        "Test Accuracy:",
        f"{test_accuracy:.4f}",
    )

    print(
        "TensorFlow + Ray Iris "
        "Test Loss:",
        f"{test_loss:.6f}",
    )


    # --------------------------------------------------------
    # Only rank 0 writes the final output files.
    #
    # This prevents both Ray workers from trying to write
    # the same files at the same time.
    #
    # /mnt/shared is accessible from both VMs.
    # --------------------------------------------------------

    if world_rank == 0:

        # ----------------------------------------------------
        # Save predictions.
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
        # Save metrics.
        # ----------------------------------------------------

        with open(
            METRICS_FILE,
            "w",
        ) as file:

            file.write(
                "TensorFlow + Ray "
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
                f"Per-worker batch size: "
                f"{per_worker_batch_size}\n"
            )

            file.write(
                f"Global batch size: "
                f"{global_batch_size}\n"
            )

            file.write(
                f"Test loss: "
                f"{test_loss:.6f}\n"
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
    # Only simple scalar metrics are reported.
    # The program does NOT rely on result.metrics afterward.
    # --------------------------------------------------------

    train.report(
        {
            "test_accuracy":
            float(test_accuracy),

            "test_loss":
            float(test_loss),

            "training_records":
            int(len(x_train)),

            "testing_records":
            int(len(x_test)),
        }
    )


# ============================================================
# Main Program
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
    # Ray Train scaling configuration
    #
    # Assignment:
    # 2 workers
    # CPU execution
    # --------------------------------------------------------

    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=False,
    )


    # --------------------------------------------------------
    # Create TensorFlow Trainer
    # --------------------------------------------------------

    trainer = TensorflowTrainer(

        train_loop_per_worker=(
            train_func
        ),

        train_loop_config={
            "epochs": 100,
            "batch_size": 16,
        },

        scaling_config=(
            scaling_config
        ),

        run_config=RunConfig(
            storage_path="/mnt/shared"
        ),
    )


    # --------------------------------------------------------
    # Start distributed training.
    #
    # Rank 0 writes the final prediction and metric files.
    # --------------------------------------------------------

    result = trainer.fit()


    # --------------------------------------------------------
    # Important:
    #
    # We intentionally do NOT use:
    #
    #     result.metrics["test_accuracy"]
    #
    # because with the current Ray Train behavior the returned
    # Result may not contain free-floating metrics when no
    # checkpoint is attached.
    #
    # The final results are already saved by Ray worker rank 0.
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "TensorFlow + Ray Iris training "
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
