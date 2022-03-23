import numpy as np
import numpy.typing as npt
import pytest
from pytrec_eval import RelevanceEvaluator

from irmetrics.metrics import precision_at_k

SEEDS = [0, 13, 23, 42, 110, 666, 911, 1337, 2718, 31415, 8675309]
K = [1, 2, 3, 4, 5, 10, 15, 20, 30, 100, 200, 500, 1000]
N = [5, 10, 100, 1000]


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"Seed: {seed}")
@pytest.mark.parametrize("k", K, ids=lambda k: f"k: {k}")
@pytest.mark.parametrize("n", N, ids=lambda n: f"n: {n}")
def test_precision_at_k(seed: int, k: int, n: int):
    rng = np.random.default_rng(seed)

    relevancies = rng.integers(low=0, high=1, size=n, endpoint=True)
    scores = rng.uniform(low=0, high=10, size=n)

    expected_value = compute_score_via_trec_eval(f"P_{k}", relevancies, scores)

    actual_value = precision_at_k(relevancies, scores, k)

    assert actual_value == pytest.approx(expected_value)


def compute_score_via_trec_eval(measure_name: str, relevancies: npt.NDArray[int], scores: npt.NDArray[float]) -> float:
    assert len(relevancies) == len(scores)
    n = len(relevancies)

    qrel = {f"d{i+1}": int(relevancies[i]) for i in range(n)}
    run = {f"d{i + 1}": float(scores[i]) for i in range(n)}

    evaluator = RelevanceEvaluator({"q1": qrel}, {measure_name})
    result = evaluator.evaluate({"q1": run})
    score = result["q1"][measure_name]
    return score
