import csv

import matplotlib

# Ubuntu server does not have a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Prediction files created by the four Problem 1 programs
# ============================================================

FILES = {
    "TensorFlow":
        "/mnt/shared/Assignment5/"
        "tensorflow_iris_predictions.csv",

    "TensorFlow + Ray":
        "/mnt/shared/Assignment5/"
        "tensorflow_ray_iris_predictions.csv",

    "PyTorch":
        "/mnt/shared/Assignment5/"
        "pytorch_iris_predictions.csv",

    "PyTorch + Ray":
        "/mnt/shared/Assignment5/"
        "problem1_pytorch_ray_predictions.csv",
}


# ============================================================
# Output visualization files
# ============================================================

OUTPUTS = {
    "TensorFlow":
        "/mnt/shared/Assignment5/"
        "tensorflow_iris_results.png",

    "TensorFlow + Ray":
        "/mnt/shared/Assignment5/"
        "tensorflow_ray_iris_results.png",

    "PyTorch":
        "/mnt/shared/Assignment5/"
        "pytorch_iris_results.png",

    "PyTorch + Ray":
        "/mnt/shared/Assignment5/"
        "pytorch_ray_iris_results.png",
}


# Different markers for the three Iris classes.
MARKERS = {
    "setosa": "o",
    "versicolor": "s",
    "virginica": "^",
}


def load_predictions(filename):

    rows = []

    with open(filename, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(
                {
                    "actual":
                        row["actual_species"],

                    "predicted":
                        row["predicted_species"],

                    "petal_length":
                        float(
                            row["petal_length"]
                        ),

                    "petal_width":
                        float(
                            row["petal_width"]
                        ),
                }
            )

    return rows


def plot_results(
    model_name,
    input_file,
    output_file,
):

    rows = load_predictions(
        input_file
    )


    plt.figure(
        figsize=(8, 6)
    )


    # --------------------------------------------------------
    # Plot correctly classified samples
    # --------------------------------------------------------

    for species in [
        "setosa",
        "versicolor",
        "virginica",
    ]:

        correct_rows = [
            row
            for row in rows
            if (
                row["actual"] == species
                and
                row["actual"]
                == row["predicted"]
            )
        ]


        plt.scatter(
            [
                row["petal_length"]
                for row in correct_rows
            ],

            [
                row["petal_width"]
                for row in correct_rows
            ],

            marker=MARKERS[
                species
            ],

            label=(
                species
                + " - correct"
            ),
        )


    # --------------------------------------------------------
    # Misclassified samples
    # --------------------------------------------------------

    error_rows = [
        row
        for row in rows
        if (
            row["actual"]
            != row["predicted"]
        )
    ]


    if len(error_rows) > 0:

        plt.scatter(
            [
                row["petal_length"]
                for row in error_rows
            ],

            [
                row["petal_width"]
                for row in error_rows
            ],

            c="red",
            marker="x",
            s=100,
            linewidths=2,
            label="Misclassified",
        )


    plt.xlabel(
        "Petal Length"
    )

    plt.ylabel(
        "Petal Width"
    )

    plt.title(
        model_name
        + " Iris Classification"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    print(
        "Created:",
        output_file,
    )


# ============================================================
# Generate all four figures
# ============================================================

for model_name in FILES:

    plot_results(
        model_name,
        FILES[
            model_name
        ],
        OUTPUTS[
            model_name
        ],
    )


print(
    "\nProblem 1 visualization "
    "completed successfully."
)
