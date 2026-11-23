import csv
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models


INPUT_FILE = "/mnt/shared/Assignment5/iris.csv"
OUTPUT_FILE = "/mnt/shared/Assignment5/tensorflow_iris_predictions.csv"
METRICS_FILE = "/mnt/shared/Assignment5/tensorflow_iris_metrics.txt"


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

    # Random 80/20 split.
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


x_train, y_train, x_test, y_test = load_iris()


print("Training records:", len(x_train))
print("Testing records:", len(x_test))


# Required architecture:
# 4 inputs -> 5 neurons -> 6 neurons -> 3 outputs

model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(5, activation="relu"),
    layers.Dense(6, activation="relu"),
    layers.Dense(3, activation="softmax"),
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


print("\nTensorFlow Iris Model")
model.summary()


model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=16,
    validation_data=(x_test, y_test),
    shuffle=True,
    verbose=1,
)


test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0,
)


print(
    "\nTensorFlow Iris Test Accuracy:",
    f"{test_accuracy:.4f}",
)


probabilities = model.predict(
    x_test,
    verbose=0,
)

predictions = np.argmax(
    probabilities,
    axis=1,
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
