import unittest

from representation_noising.utility_sampling import (
    allocate_samples,
    sample_utility_indices,
)


class UtilitySamplingTest(unittest.TestCase):
    def test_task_proportions_and_exact_budget(self):
        sizes = {"mmlu": 14042, "truthfulqa": 817, "arc_easy": 2376, "hellaswag": 10042}
        self.assertEqual(
            allocate_samples(sizes, 1000),
            {"mmlu": 515, "truthfulqa": 30, "arc_easy": 87, "hellaswag": 368},
        )

    def test_random_subset_and_subject_proportions(self):
        sizes = {"group": {"a": 600, "b": 400}, "other": {"c": 1000}}
        indices = sample_utility_indices(sizes, 100, 42)
        self.assertEqual(
            {k: len(v) for k, v in indices.items()}, {"a": 30, "b": 20, "c": 50}
        )
        self.assertEqual(indices, sample_utility_indices(sizes, 100, 42))
        self.assertNotEqual(indices, sample_utility_indices(sizes, 100, 43))
        for name, size in {"a": 600, "b": 400, "c": 1000}.items():
            self.assertEqual(len(indices[name]), len(set(indices[name])))
            self.assertTrue(all(0 <= index < size for index in indices[name]))
            self.assertNotEqual(indices[name], list(range(len(indices[name]))))

    def test_budget_above_dataset_size_uses_all_questions(self):
        self.assertEqual(
            sample_utility_indices({"task": {"a": 3}}, 1000, 42), {"a": [0, 1, 2]}
        )

    def test_invalid_or_unrepresentative_budgets(self):
        for budget in [0, -1, 1]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                sample_utility_indices({"group": {"a": 100, "b": 100}}, budget, 42)
