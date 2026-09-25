"""Export portfolio diagrams and model results as PNG, SVG, and PDF files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "media"
BG = "#F4F6F8"
INK = "#14293D"
MUTED = "#526577"
TEAL = "#087F83"
BLUE = "#315DC5"
LINE = "#D8E1E8"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    }
)


def canvas():
    fig = plt.figure(figsize=(18, 13.5), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1800), ylim=(1350, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, value, size=28, color=INK, weight="normal", **kwargs):
    return ax.text(
        x,
        y,
        value,
        fontsize=size * 0.72,
        color=color,
        weight=weight,
        va="top",
        linespacing=1.5,
        **kwargs,
    )


def box(ax, x, y, width, height, fill=WHITE, edge=LINE, radius=20):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1,
        )
    )


def line(ax, points, color=LINE, width=1.3, **kwargs):
    ax.plot(*zip(*points, strict=True), color=color, linewidth=width, **kwargs)


def arrow(ax, start, end, color=TEAL):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def header(ax, category, title, subtitle, number):
    box(ax, 80, 62, 9, 28, TEAL, TEAL, 3)
    text(ax, 108, 64, f"ML INFERENCE SERVICE  /  {category}", 22, TEAL, "bold")
    text(ax, 80, 124, title, 61, weight="bold")
    text(ax, 80, 209, subtitle, 28, MUTED)
    text(ax, 1720, 64, number, 22, MUTED, ha="right")


def footer(ax, source):
    line(ax, [(80, 1255), (1720, 1255)])
    text(ax, 80, 1280, source, 19, MUTED)
    text(ax, 1720, 1280, "KEANU SWART", 19, INK, "bold", ha="right")


def export(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Catch text accidentally placed outside the canvas before exporting.
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for label in ax.texts:
            bounds = label.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(
                bounds.x1, bounds.y1
            ):
                raise ValueError(f"Text outside canvas: {label.get_text()}")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", dpi=160, facecolor=BG)
    plt.close(fig)


def architecture():
    fig, ax = canvas()
    header(
        ax,
        "ARCHITECTURE",
        "From request to prediction",
        "Independent HTTP requests share a custom scheduler and a single inference worker.",
        "01 / 02",
    )

    box(ax, 80, 290, 1080, 710, "#EAF0F5", "#EAF0F5", 24)
    text(ax, 112, 321, "EVENT LOOP", 22, BLUE, "bold")
    text(ax, 112, 359, "Owns queues, deadlines and request completion", 25, MUTED)

    box(ax, 110, 423, 475, 296)
    text(ax, 142, 453, "01  /  RECEIVE", 20, BLUE, "bold")
    text(ax, 142, 493, "FastAPI endpoint", 35, weight="bold")
    text(
        ax,
        142,
        550,
        "Validate body and text limits\nLook up model + version\nAssign request ID and deadline",
        25,
    )

    box(ax, 655, 423, 475, 296)
    text(ax, 687, 453, "02  /  SCHEDULE", 20, BLUE, "bold")
    text(ax, 687, 493, "Bounded queue", 35, weight="bold")
    text(
        ax,
        687,
        550,
        "Single • Immediate • Timed\nBatch size and wait limits\nQueue full → HTTP 429",
        25,
    )
    arrow(ax, (585, 571), (655, 571), BLUE)

    box(ax, 1230, 290, 490, 710, INK, INK, 24)
    text(ax, 1267, 321, "DEDICATED WORKER THREAD", 21, "#8EDBDC", "bold")
    text(ax, 1267, 382, "03  /  EXECUTE", 20, "#8EDBDC", "bold")
    text(ax, 1267, 430, "One active batch", 36, WHITE, "bold")
    text(ax, 1267, 490, "Tokenize → forward pass → scores", 24, WHITE)
    arrow(ax, (1130, 571), (1230, 571))
    text(ax, 1180, 531, "batch", 18, MUTED, ha="center")

    line(ax, [(1267, 580), (1683, 580)], "#385064")
    text(ax, 1267, 613, "ONE MODEL PER SERVER SESSION", 19, "#A9C0D0", "bold")
    text(ax, 1267, 657, "Custom PyTorch classifier", 26, WHITE, "bold")
    text(ax, 1267, 701, "or pinned DistilBERT", 26, WHITE, "bold")
    text(ax, 1267, 767, "Shared adapter interface\nOrdered results for every batch", 24, "#D4E2EA")
    text(ax, 1267, 900, "Model work runs off the event loop.", 23, "#8EDBDC")

    box(ax, 110, 774, 1020, 189)
    text(ax, 142, 800, "04  /  RESPOND", 20, BLUE, "bold")
    text(ax, 142, 840, "Each result returns to its original caller", 32, weight="bold")
    text(ax, 142, 898, "Check result count • Resolve request futures • Include request ID", 24)
    arrow(ax, (1230, 865), (1130, 865))
    text(ax, 1180, 890, "results", 18, MUTED, ha="center")

    text(ax, 80, 1055, "CONTROLLED REQUEST LIFECYCLE", 21, TEAL, "bold")
    text(ax, 80, 1098, "Deadlines • Cancellation cleanup • Graceful shutdown • Worker watchdog", 28)
    text(ax, 80, 1158, "OBSERVABILITY", 21, TEAL, "bold")
    text(
        ax,
        80,
        1200,
        "Prometheus metrics • Structured JSON logs • Health and readiness endpoints",
        26,
    )
    footer(ax, "Source: docs/ARCHITECTURE.md • One server process; one configured model")
    export(fig, "inference-architecture")


def results():
    data = json.loads((ROOT / "docs/results/custom-small.json").read_text())
    accuracy = 100 * data["test"]["accuracy"]
    baseline = 100 * data["training_majority_test_baseline"]["accuracy"]
    matrix = data["test"]["confusion_matrix_true_by_predicted"]
    total = data["test"]["count"]
    assert sum(map(sum, matrix)) == total
    assert abs((matrix[0][0] + matrix[1][1]) / total * 100 - accuracy) < 1e-8

    fig, ax = canvas()
    header(
        ax,
        "MODEL EVALUATION",
        "A sentiment model trained from scratch",
        "PyTorch classifier • UCI Sentiment Labelled Sentences • Evaluation on unseen test data",
        "02 / 02",
    )

    for x, value, label, detail in (
        (80, f"{accuracy:.2f}%", "TEST ACCURACY", "309 correct predictions out of 449"),
        (
            640,
            f"+{accuracy - baseline:.2f}",
            "PERCENTAGE POINTS",
            "Above the majority-class baseline",
        ),
        (1200, f"{data['test']['macro_f1']:.4f}", "MACRO-F1", "F1 averaged across both classes"),
    ):
        box(ax, x, 290, 520, 207)
        text(ax, x + 30, 315, label, 20, MUTED, "bold")
        text(ax, x + 30, 357, value, 62, TEAL, "bold")
        text(ax, x + 30, 444, detail, 22, MUTED)

    box(ax, 80, 527, 920, 544)
    text(ax, 115, 560, "Accuracy on 449 test sentences", 32, weight="bold")
    text(ax, 115, 608, "Both evaluated on the same held-out split", 23, MUTED)
    for value in (0, 25, 50, 75, 100):
        x = 120 + value * 8
        line(ax, [(x, 702), (x, 954)], "#E4EAF0", 1)
        text(ax, x, 973, f"{value}%", 21, MUTED, ha="center")
    for y, label, value, fill in (
        (658, "Custom PyTorch model", accuracy, TEAL),
        (805, "Majority-class baseline", baseline, "#8193A4"),
    ):
        text(ax, 120, y, label, 25, INK, "bold")
        box(ax, 120, y + 51, 8 * value, 53, fill, fill, 7)
        text(ax, 120 + 8 * value + 16, y + 60, f"{value:.2f}%", 28, fill, "bold")
    text(ax, 115, 1020, "Baseline predicts negative for every test sentence.", 22, MUTED)

    box(ax, 1030, 527, 690, 544)
    text(ax, 1065, 560, "Prediction breakdown", 32, weight="bold")
    text(ax, 1065, 608, "Confusion matrix · Number of sentences", 23, MUTED)
    text(ax, 1450, 659, "PREDICTED", 19, MUTED, "bold", ha="center")
    text(ax, 1342, 698, "Negative", 23, INK, ha="center")
    text(ax, 1562, 698, "Positive", 23, INK, ha="center")
    text(ax, 1065, 737, "ACTUAL", 19, MUTED, "bold")
    for row, label in enumerate(("Negative", "Positive")):
        y = 746 + row * 112
        text(ax, 1065, y + 43, label, 23)
        for col in range(2):
            x = 1240 + col * 220
            correct = row == col
            box(ax, x, y, 204, 98, TEAL if correct else "#EAF0F5", WHITE, 10)
            text(
                ax,
                x + 102,
                y + 24,
                str(matrix[row][col]),
                43,
                WHITE if correct else INK,
                "bold",
                ha="center",
            )
    text(ax, 1065, 985, "Teal cells = correct predictions", 22, TEAL, "bold")
    text(ax, 1065, 1020, "103 positive sentences were missed.", 22, MUTED)

    text(ax, 80, 1112, "HOW THE MODEL WAS BUILT", 21, TEAL, "bold")
    text(
        ax,
        80,
        1154,
        "1,200 training sentences • Vocabulary fit on training only • "
        "Checkpoint chosen on validation",
        25,
    )
    text(
        ax,
        80,
        1200,
        "Token embeddings → masked mean pooling → hidden layer → two sentiment scores",
        25,
        MUTED,
    )
    footer(ax, "Source: docs/results/custom-small.json • UCI / Kotzias (2015), CC BY 4.0")
    export(fig, "custom-model-results")


if __name__ == "__main__":
    architecture()
    results()
    print(f"Wrote PNG, SVG and PDF graphics to {OUTPUT}")
