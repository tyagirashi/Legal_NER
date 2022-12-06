from typing import List

import torch


def precision(
    predicted_labels: List[torch.Tensor],
    true_labels: List[torch.Tensor],
    outside_tag_idx: int
):
    """
    Precision is True Positives / All Positives Predictions
    """
    TP = torch.tensor([0])
    denom = torch.tensor([0])
    for pred, true in zip(predicted_labels, true_labels):
        TP += sum((pred == true)[pred != outside_tag_idx])
        denom += sum(pred != outside_tag_idx)

    # Avoid division by 0
    denom = torch.tensor(1) if denom == 0 else denom
    return TP / denom


def recall(
    predicted_labels: List[torch.Tensor],
    true_labels: List[torch.Tensor],
    outside_tag_idx: int
):
    """
    Recall is True Positives / All Positive Labels
    """
    TP = torch.tensor([0])
    denom = torch.tensor([0])
    for pred, true in zip(predicted_labels, true_labels):
        TP += sum((pred == true)[true != outside_tag_idx])
        denom += sum(true != outside_tag_idx)

    # Avoid division by 0
    denom = torch.tensor(1) if denom == 0 else denom
    return TP / denom


def f1_score(predicted_labels, true_labels, outside_tag_idx):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels, outside_tag_idx)
    R = recall(predicted_labels, true_labels, outside_tag_idx)
    return 2*P*R/(P+R)