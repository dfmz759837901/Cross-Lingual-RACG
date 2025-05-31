import itertools
import numpy as np

from typing import *


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def count_pass_at_k(result_data: List[Dict], pass_k: int = 1) -> Tuple[float, int]:
    id_map = {}
    num_correct = []
    num_samples = []
    for r in result_data:
        problem_id: str = r["task_id"]
        if problem_id not in id_map:
            id_map[problem_id] = len(id_map)
            num_correct.append(0)
            num_samples.append(0)
        idx = id_map[problem_id]
        if num_samples[idx] >= pass_k:
            print(f"{problem_id} num_samples > pass_k = {pass_k}")
            continue
        num_samples[idx] += 1
        assert "passed" in r
        if r["passed"] == True:
            assert r["result"] == "passed", r["result"]
            num_correct[idx] += 1
    assert len(num_correct) == len(num_samples)
    assert len(id_map) == len(num_correct)
    for n in num_samples:
        assert n == pass_k, f"{n} != {pass_k}"
    return estimate_pass_at_k(
        num_samples=num_samples, num_correct=num_correct, k=pass_k
    ).mean(), len(num_samples)


def get_unique_ordered(items: List[str]) -> List[str]:
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]


def calc_precision(golden: List[str], pred: List[str], top_k: int) -> float:
    assert top_k > 0
    golden = set(get_unique_ordered(golden))
    pred = get_unique_ordered(pred)
    assert len(pred) >= top_k, f"{len(pred)} < {top_k}, {pred}"
    pred = set(pred[:top_k])
    return len(golden & pred) / len(pred)


def calc_recall(golden: List[str], pred: List[str], top_k: int) -> float:
    assert top_k > 0
    golden = set(get_unique_ordered(golden))
    pred = get_unique_ordered(pred)
    assert len(pred) >= top_k, f"{len(pred)} < {top_k}, {pred}"
    pred = set(pred[:top_k])
    return len(golden & pred) / len(golden)
