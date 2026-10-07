import torch

from recsys.augment import augment_plan


def plan(seed=0, batch=500, length=10, mask=0.0, crop=0, reorder=0.0):
    return augment_plan(batch, length, torch.Generator().manual_seed(seed), mask, crop, reorder)


def test_all_off_changes_nothing():
    order, hidden = plan()
    assert torch.equal(order, torch.arange(10).expand(500, 10))
    assert not hidden.any()


def test_reorder_shuffles_one_run_of_3_to_5_songs_and_keeps_every_song():
    order, _ = plan(reorder=1.0)
    for row in order:
        assert sorted(row.tolist()) == list(range(10))           # a permutation: no song lost or repeated
        moved = (row != torch.arange(10)).nonzero().flatten()
        if len(moved):                                             # (a shuffle can leave a run unchanged)
            assert moved.max() - moved.min() + 1 <= 5              # all moves inside one run of <= 5
    assert (order != torch.arange(10)).any(dim=1).float().mean() > 0.6


def test_reorder_probability_controls_how_many_rows_change():
    order, _ = plan(reorder=0.5, batch=4000)
    changed = (order != torch.arange(10)).any(dim=1).float().mean().item()
    assert 0.35 < changed < 0.5  # 50% chosen, minus shuffles that happen to keep the order


def test_crop_hides_a_prefix_of_at_most_crop_songs():
    _, hidden = plan(crop=5)
    counts = hidden.sum(dim=1)
    assert counts.max() == 5 and counts.min() == 0
    for row, n in zip(hidden, counts):
        assert row[:n].all() and not row[n:].any()  # hidden songs are exactly the first n


def test_mask_hides_about_mask_prob_of_songs():
    _, hidden = plan(mask=0.2, batch=4000)
    assert 0.18 < hidden.float().mean().item() < 0.22


def test_same_generator_seed_same_plan():
    a, b = plan(seed=3, mask=0.2, crop=5, reorder=0.5), plan(seed=3, mask=0.2, crop=5, reorder=0.5)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
