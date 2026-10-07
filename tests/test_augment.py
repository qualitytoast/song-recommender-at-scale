import torch

from recsys.augment import augment_plan, ignored_targets


def plan(seed=0, batch=500, length=10, mask=0.0, crop=0, reorder=0.0, lengths=None):
    return augment_plan(batch, length, torch.Generator().manual_seed(seed), mask, crop, reorder, lengths)[:2]


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



# --- chunks (every-position objective) ---

def test_full_lengths_give_exactly_the_window_plan():
    # Passing lengths for full rows must not change any random draw (p2_augment reproduces).
    a = augment_plan(300, 10, torch.Generator().manual_seed(5), 0.2, 5, 0.5)
    b = augment_plan(300, 10, torch.Generator().manual_seed(5), 0.2, 5, 0.5, torch.full((300,), 10))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_shuffled_runs_stay_inside_the_real_songs_of_short_chunks():
    lengths = torch.randint(1, 11, (2000,), generator=torch.Generator().manual_seed(1))
    order, _ = plan(batch=2000, reorder=1.0, lengths=lengths)
    for row, n in zip(order, lengths):
        assert torch.equal(row[n:], torch.arange(n, 10))   # padding never moves
        if n < 3:
            assert torch.equal(row, torch.arange(10))       # too short for a 3-song run


def test_no_kept_target_can_see_a_later_song_or_its_answer():
    # The property that matters: wherever a target is kept, the context so far holds
    # exactly the songs that really came before it (in any order), nothing from later.
    order, hidden, mid = augment_plan(3000, 10, torch.Generator().manual_seed(2), 0.0, 0, 1.0)
    ignored = ignored_targets(hidden, mid)
    for row, ign in zip(order, ignored):
        for p in range(10):
            if not ign[p]:
                assert set(row[:p + 1].tolist()) == set(range(p + 1))
    assert ignored.any(dim=1).float().mean() > 0.5  # most rows had a run shuffled, so some targets are ignored


def test_targets_with_nothing_visible_are_ignored():
    hidden = torch.tensor([[True, True, False, True], [False, True, True, True]])
    no_shuffle = torch.zeros(2, 4, dtype=torch.bool)
    assert ignored_targets(hidden, no_shuffle).tolist() == [[True, True, False, False], [False, False, False, False]]
