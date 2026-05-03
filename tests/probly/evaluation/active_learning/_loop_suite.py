"""Shared loop test suite for all backends."""

from __future__ import annotations

from probly.evaluation.active_learning.loop import active_learning_steps
from probly.evaluation.active_learning.pool import from_dataset
from probly.evaluation.active_learning.strategies import RandomQuery


class LoopSuite:
    """Backend-agnostic loop tests.

    Requires fixtures: classification_data, make_estimator.
    """

    def test_yields_initial_state(self, classification_data, make_estimator):
        """n_iterations=0 must yield exactly one state with iteration=0."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        est = make_estimator()
        states = list(active_learning_steps(pool, est, RandomQuery(seed=1), n_iterations=0))
        assert len(states) == 1
        assert states[0].iteration == 0

    def test_yields_correct_number(self, classification_data, make_estimator):
        """n_iterations=3 must yield 4 states: initial + 3 query-retrain cycles."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        est = make_estimator()
        states = list(active_learning_steps(pool, est, RandomQuery(seed=1), query_size=10, n_iterations=3))
        assert len(states) == 4

    def test_sequential_iteration_numbers(self, classification_data, make_estimator):
        """Iteration numbers must be 0, 1, 2, 3 in order."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        est = make_estimator()
        states = list(active_learning_steps(pool, est, RandomQuery(seed=1), query_size=10, n_iterations=3))
        assert [s.iteration for s in states] == [0, 1, 2, 3]

    def test_pool_grows_each_iteration(self, classification_data, make_estimator):
        """Labeled set must grow by query_size after each query-retrain step."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        est = make_estimator()
        query_size = 10
        labeled_sizes = [
            state.pool.n_labeled
            for state in active_learning_steps(pool, est, RandomQuery(seed=1), query_size=query_size, n_iterations=3)
        ]

        initial_size = labeled_sizes[0]
        for i, size in enumerate(labeled_sizes):
            assert size == initial_size + i * query_size

    def test_estimator_can_predict_after_training(self, classification_data, make_estimator):
        """Each yielded state's estimator must be able to predict on test data."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        est = make_estimator()
        for state in active_learning_steps(pool, est, RandomQuery(seed=1), query_size=10, n_iterations=2):
            preds = state.estimator.predict(state.pool.x_test)
            assert len(preds) == len(x_test)

    def test_stops_when_pool_exhausted(self, classification_data, make_estimator):
        """Iterator must stop early when the unlabeled pool is emptied."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=130, seed=0)
        est = make_estimator()
        states = list(active_learning_steps(pool, est, RandomQuery(seed=1), query_size=10, n_iterations=5))
        # 20 unlabeled, query_size=10 -> initial + 2 iterations = 3 states
        assert len(states) == 3
