"""Select a reproducible, proportionate utility-evaluation subset."""

import random


def allocate_samples(sizes: dict[str, int], budget: int) -> dict[str, int]:
    """Allocate an exact budget proportionally, using largest remainders."""
    total = sum(sizes.values())
    if budget < 1 or not sizes or min(sizes.values()) < 1:
        raise ValueError("Sample budget and task sizes must be positive")
    budget = min(budget, total)
    counts = {name: budget * size // total for name, size in sizes.items()}
    order = sorted(sizes, key=lambda name: -(budget * sizes[name] % total))
    for name in order[: budget - sum(counts.values())]:
        counts[name] += 1
    if min(counts.values()) == 0:
        raise ValueError("Utility sample budget is too small to include every subtask")
    return counts


def sample_utility_indices(
    task_sizes: dict[str, dict[str, int]], budget: int, seed: int
) -> dict[str, list[int]]:
    """Preserve task and subtask proportions, sampling without replacement."""
    counts = allocate_samples(
        {name: sum(sizes.values()) for name, sizes in task_sizes.items()}, budget
    )
    rng = random.Random(seed)
    indices = {}
    for name, sizes in task_sizes.items():
        for subtask, count in allocate_samples(sizes, counts[name]).items():
            indices[subtask] = sorted(rng.sample(range(sizes[subtask]), count))
    return indices
