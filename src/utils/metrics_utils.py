from collections.abc import Collection

import statistics


def compute_std(data: list[float | int]) -> float:
  """Computes the standard deviation of a list of values."""
  return statistics.pstdev(data)


class AverageMeter:

  def __init__(self):
    self.reset()

  def reset(self):
    """Resets all the meter variables."""
    self.val = 0  # most recent value
    self.avg = 0  # average of all values
    self.sum = 0  # sum of all values
    self.count = 0  # number of updates
    self.std = None
    self.vals = []  # list of all values

  def update(self, value):
    """
        Updates the meter with a new value.

        Args:
            value (float): The new value to add.
        """
    self.val = value
    self.vals.append(value)
    self.sum += value
    self.count += 1
    self.avg = self.sum / self.count if self.count != 0 else 0
    self.std = compute_std(self.vals)


class AverageStrListMeter:
  """
    Tracks and averages precision and recall over multiple updates.
    
    Each update expects a ground truth list and a prediction list (both lists of strings),
    and computes the precision and recall for that update:
      - precision = (# of correct predictions) / (# of predictions)
      - recall    = (# of correct predictions) / (# of ground truth items)
      
    The averages are computed over all updates.
    """

  def __init__(self):
    self.reset()

  def reset(self):
    """Resets the meter."""
    self.total_precision = 0.0
    self.total_recall = 0.0
    self.count = 0
    self.vals = []
    self.std = {
        "precision": None,
        "recall": None,
    }

  def update(
      self,
      gt: Collection[str] | None = None,
      pred: Collection[str] | None = None,
      precision: float | None = None,
      recall: float | None = None,
  ) -> dict[str, float]:
    """Update the meter with a new batch of ground truth and predictions."""

    if precision is None or recall is None:
      assert gt is not None, "Ground truth must be provided if precision and recall are not given."
      assert pred is not None, "Predictions must be provided if precision and recall are not given."
      # Compute number of correct predictions (intersection of sets)
      correct = len(set(gt) & set(pred))

      # Compute precision and recall for this update.
      # If pred is empty, define precision as 0.
      precision = correct / len(pred) if pred else 0.0
      # If gt is empty, define recall as 0.
      recall = correct / len(gt) if gt else 0.0

    self.total_precision += precision
    self.total_recall += recall
    self.count += 1
    self.vals.append({
        "precision": precision,
        "recall": recall,
    })
    self.std["precision"] = compute_std([v["precision"] for v in self.vals])
    self.std["recall"] = compute_std([v["recall"] for v in self.vals])
    return {
        "precision": precision,
        "recall": recall,
    }

  @property
  def precision(self) -> float:
    """Returns the average precision over all updates."""
    return self.total_precision / self.count if self.count > 0 else 0.0

  @property
  def recall(self) -> float:
    """Returns the average recall over all updates."""
    return self.total_recall / self.count if self.count > 0 else 0.0

  def __str__(self):
    return f"Avg Precision: {self.precision:.4f}, Avg Recall: {self.recall:.4f}"
