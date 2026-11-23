import csv
import numpy as np
import ray
import tensorflow as tf

from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.tensorflow import TensorflowTrainer


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

INPUT_FILE = "/mnt/shared/Assignment5/iris.csv"

OUTPUT_FILE = (
    "/mnt/shared/Assignment5/"
    "tensorflow_ray_iris_predictions.csv"
)

METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "tensorflow_ray_iris_metrics.txt"
)


# ---------------------------------------------------------
# Iris class labels
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Load Iris dataset and perform random 80/20 split
# ---------------------------------------------------------

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


    # Random split
    np.random.seed(42)

    indices = np.arange(
        len(features)
    )

    np.random.shuffle(
        indices
    )


    # 80% training, 20% testing
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


# ---------------------------------------------------------
# Build required Feedforward Neural Network
#
# Input layer   : 4 Iris features
# Hidden layer 1: 5 neurons
# Hidden layer 2: 6 neurons
# Output layer  : 3 Iris classes
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Ray Train worker function
# ---------------------------------------------------------

def train_func(config):

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_iris()


    # Number of Ray Train workers
    num_workers = (
        train.get_context()
        .get_world_size()
    )


    print(
        "Ray Train world size:",
        num_workers,
    )

    print(
        "Training records:",
        len(x_train),
    )

    print(
        "Testing records:",
        len(x_test),
    )


    # -----------------------------------------------------
    # TensorFlow distributed training strategy
    # -----------------------------------------------------

    strategy = (
        tf.distribute
        .MultiWorkerMirroredStrategy()
    )


    # Model must be created and compiled inside
    # strategy.scope() for multi-worker training.
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


    # -----------------------------------------------------
    # Global batch size
    #
    # Lecture:
    # global_batch_size =
    # per_worker_batch_size * num_workers
    # -----------------------------------------------------

    per_worker_batch_size = (
        config.get(
            "batch_size",
            16,
        )
    )


    global_batch_size = (
        per_worker_batch_size
        * num_workers
    )


    print(
        "Per-worker batch size:",
        per_worker_batch_size,
    )

    print(
        "Global batch size:",
        global_batch_size,
    )


    # -----------------------------------------------------
    # TensorFlow training dataset
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Evaluate on test dataset
    # -----------------------------------------------------

    test_loss, test_accuracy = (
        model.evaluate(
            x_test,
            y_test,
            verbose=0,
        )
    )


    # -----------------------------------------------------
    # Generate predictions for visualization
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Report results to Ray Train
    #
    # Every Ray Train worker calls train.report().
    # -----------------------------------------------------

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

            "actual_labels":
            y_test.tolist(),

            "predicted_labels":
            predictions.tolist(),

            "petal_length":
            x_test[:, 2].tolist(),

            "petal_width":
            x_test[:, 3].tolist(),
        }
    )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if __name__ == "__main__":

    # Connect to existing Ray cluster.
    ray.init(
        address="auto"
    )


    print(
        "Connected to Ray cluster."
    )


    # -----------------------------------------------------
    # Ray Train configuration
    #
    # Assignment:
    # Two training workers
    # CPU only
    #
    # /mnt/shared is available from both VMs.
    # -----------------------------------------------------

    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=False,
    )


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


    # -----------------------------------------------------
    # Start distributed training
    # -----------------------------------------------------

    result = trainer.fit()


    # -----------------------------------------------------
    # Get final metrics from Ray
    # -----------------------------------------------------

    metrics = result.metrics


    test_accuracy = float(
        metrics[
            "test_accuracy"
        ]
    )

    test_loss = float(
        metrics[
            "test_loss"
        ]
    )


    print(
        "\n================================"
    )

    print(
        "TensorFlow + Ray Iris Results"
    )

    print(
        "================================"
    )

    print(
        "Training records:",
        metrics[
            "training_records"
        ],
    )

    print(
        "Testing records:",
        metrics[
            "testing_records"
        ],
    )

    print(
        "Test Loss:",
        f"{test_loss:.6f}",
    )

    print(
        "Test Accuracy:",
        f"{test_accuracy:.4f}",
    )


    # -----------------------------------------------------
    # Save predictions for visualization
    # -----------------------------------------------------

    actual_labels = metrics[
        "actual_labels"
    ]

    predicted_labels = metrics[
        "predicted_labels"
    ]

    petal_length = metrics[
        "petal_length"
    ]

    petal_width = metrics[
        "petal_width"
    ]


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
            len(actual_labels)
        ):

            actual_species = (
                LABEL_TO_SPECIES[
                    int(
                        actual_labels[i]
                    )
                ]
            )

            predicted_species = (
                LABEL_TO_SPECIES[
                    int(
                        predicted_labels[i]
                    )
                ]
            )


            writer.writerow(
                [
                    actual_species,
                    predicted_species,
                    petal_length[i],
                    petal_width[i],
                ]
            )


    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

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
            "Training split: 80%\n"
        )

        file.write(
            "Testing split: 20%\n"
        )

        file.write(
            "Ray Train workers: 2\n"
        )

        file.write(
            "GPU used: False\n"
        )

        file.write(
            "Training records: "
            f"{metrics['training_records']}\n"
        )

        file.write(
            "Testing records: "
            f"{metrics['testing_records']}\n"
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


    ray.shutdown()
