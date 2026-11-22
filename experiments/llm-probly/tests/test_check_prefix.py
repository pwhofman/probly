from __future__ import annotations

from scripts.check_prefix import compare_prefix, format_report


def row(i: int, answer: str = "a", correct: bool = True) -> dict:
    return {
        "id": i,
        "question": f"q{i}",
        "aliases": [f"x{i}"],
        "greedy_answer": answer,
        "samples": [answer] * 10,
        "correct": correct,
    }


def test_identical_prefix():
    small = [row(i) for i in range(3)]
    large = [row(i) for i in range(5)]
    report = compare_prefix(small, large)
    assert report.same_questions
    assert report.n_compared == 3
    assert (report.same_greedy, report.same_samples, report.same_correct) == (3, 3, 3)


def test_different_question_is_found():
    small = [row(i) for i in range(3)]
    large = [row(0), row(7), row(2), row(3)]
    report = compare_prefix(small, large)
    assert not report.same_questions
    assert report.first_mismatch == 1
    assert "DIFFERENT" in format_report(report, "s", "l")


def test_changed_answers_are_counted_but_questions_still_match():
    small = [row(0), row(1, answer="b", correct=False)]
    large = [row(0), row(1, answer="c", correct=True), row(2)]
    report = compare_prefix(small, large)
    assert report.same_questions
    assert report.same_greedy == 1
    assert report.same_correct == 1


def test_partial_large_cache_compares_what_exists():
    small = [row(i) for i in range(5)]
    report = compare_prefix(small, [row(0), row(1)])
    assert report.n_compared == 2
    assert report.same_questions


def test_unlabelled_rows_are_not_compared_for_correctness():
    small = [row(0)]
    unlabelled = row(0)
    del unlabelled["correct"]
    report = compare_prefix(small, [unlabelled])
    assert report.n_labelled == 0
    assert "correctness" not in format_report(report, "s", "l")
