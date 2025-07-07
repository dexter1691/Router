import json

def recall_at_k(predictions: list[tuple[str, float]], expected: list[str], k: int, score_threshold: float) -> float:
    """ Compute the recall at k for a list of predictions and expected values.
    Args:
        predictions: A list of predictions and their scores.
        expected: A list of expected values.
        k: The number of top predictions to consider.
        score_threshold: The score threshold for a prediction to be considered.
    """
    assert k > 0, "k must be greater than 0"
    tp = set()
    for prediction in predictions[:k]:
        if prediction[1] >= score_threshold and prediction[0] in expected:
            tp.add(prediction[0])
    # print(f"tp: {tp}, expected: {expected}, predictions: {predictions[:k]}")
    return len(tp) / len(expected)


class Evals:
    def __init__(self, evals_path: str):
        with open(evals_path, "r") as f:
            self.evals = json.load(f)

    def get_queries(self):
        return [eval["query"] for eval in self.evals]
    
    def get_data(self):
        return self.evals

evals = Evals("data/evals.json")