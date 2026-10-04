from recsys.train import EarlyStopping


def run(scores, min_epochs, patience):
    """Feed scores epoch by epoch; return (epoch stopped at, epochs that improved)."""
    stopper, improved_at = EarlyStopping(min_epochs, patience), []
    for epoch, score in enumerate(scores):
        improved, stop = stopper.update(epoch, score)
        if improved:
            improved_at.append(epoch)
        if stop:
            return epoch, improved_at
    return None, improved_at


def test_v1_run_shape():
    # v1 stopped at epoch 32 with min_epochs 20, patience 10: best at epoch 22,
    # then epochs 23..32 are ten counted epochs without improvement.
    scores = [e / 100 for e in range(23)] + [0.0] * 20
    stopped, improved_at = run(scores, min_epochs=20, patience=10)
    assert stopped == 32 and improved_at[-1] == 22


def test_flat_epochs_before_min_epochs_do_not_count():
    # Best at epoch 0, nothing better after. Counting starts at epoch 3,
    # so with patience 2 it stops at epoch 4, not epoch 2.
    stopped, _ = run([1.0] + [0.5] * 10, min_epochs=3, patience=2)
    assert stopped == 4


def test_new_best_resets_counter():
    # min 0, patience 2: 1.0, 0.5 (1), 2.0 (best, reset), 0.5 (1), 0.5 (2) -> stop at 4
    stopped, improved_at = run([1.0, 0.5, 2.0, 0.5, 0.5], min_epochs=0, patience=2)
    assert stopped == 4 and improved_at == [0, 2]


def test_equal_score_is_not_an_improvement():
    stopped, improved_at = run([1.0, 1.0, 1.0], min_epochs=0, patience=2)
    assert stopped == 2 and improved_at == [0]
