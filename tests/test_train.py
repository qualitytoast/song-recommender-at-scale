from recsys.train import EarlyStopping, EvalSchedule


def run(scores, min_checks, patience):
    """Feed scores check by check; return (check stopped at, checks that improved)."""
    stopper, improved_at = EarlyStopping(min_checks, patience), []
    for check, score in enumerate(scores):
        improved, stop = stopper.update(check, score)
        if improved:
            improved_at.append(check)
        if stop:
            return check, improved_at
    return None, improved_at


def test_v1_run_shape():
    # v1 stopped at epoch 32 with min_checks 20, patience 10: best at epoch 22,
    # then epochs 23..32 are ten counted epochs without improvement.
    scores = [e / 100 for e in range(23)] + [0.0] * 20
    stopped, improved_at = run(scores, min_checks=20, patience=10)
    assert stopped == 32 and improved_at[-1] == 22


def test_flat_checks_before_min_checks_do_not_count():
    # Best at epoch 0, nothing better after. Counting starts at epoch 3,
    # so with patience 2 it stops at epoch 4, not epoch 2.
    stopped, _ = run([1.0] + [0.5] * 10, min_checks=3, patience=2)
    assert stopped == 4


def test_new_best_resets_counter():
    # min 0, patience 2: 1.0, 0.5 (1), 2.0 (best, reset), 0.5 (1), 0.5 (2) -> stop at 4
    stopped, improved_at = run([1.0, 0.5, 2.0, 0.5, 0.5], min_checks=0, patience=2)
    assert stopped == 4 and improved_at == [0, 2]


def test_equal_score_is_not_an_improvement():
    stopped, improved_at = run([1.0, 1.0, 1.0], min_checks=0, patience=2)
    assert stopped == 2 and improved_at == [0]


def test_interval_of_one_epoch_checks_exactly_at_each_epoch_end():
    # 10 targets per epoch in batches of 4, 4, 2: checks after the 3rd and 6th batch
    schedule = EvalSchedule(every=10)
    due = [schedule.add(n) for n in [4, 4, 2] * 2]
    assert due == [False, False, True, False, False, True]


def test_shorter_interval_checks_mid_epoch():
    schedule = EvalSchedule(every=5)
    assert [schedule.add(4) for _ in range(5)] == [False, True, True, True, True]  # at 8, 12, 16, 20 targets


def test_a_batch_passing_several_multiples_gives_one_check():
    schedule = EvalSchedule(every=3)
    assert schedule.add(10) is True      # passes 3, 6 and 9: one check
    assert schedule.next == 12
    assert schedule.add(1) is False and schedule.add(1) is True  # 11, then 12
