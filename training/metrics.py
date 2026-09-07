from collections.abc import Sequence


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
    if not labels or len(labels) != len(predictions):
        raise ValueError("Expected nonempty, aligned labels and predictions")
    matrix = [[0, 0], [0, 0]]
    for label, prediction in zip(labels, predictions, strict=True):
        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("Expected binary labels")
        matrix[label][prediction] += 1
    f1 = []
    for label in (0, 1):
        tp = matrix[label][label]
        denominator = 2 * tp + matrix[1 - label][label] + matrix[label][1 - label]
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "count": len(labels),
        "accuracy": (matrix[0][0] + matrix[1][1]) / len(labels),
        "macro_f1": sum(f1) / 2,
        "confusion_matrix_true_by_predicted": matrix,
        "class_distribution": {"negative": labels.count(0), "positive": labels.count(1)},
    }
