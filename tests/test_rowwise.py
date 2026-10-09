import numpy as np
import torch

from recsys.lazy_adam import LazyAdamW, is_table
from recsys.rowwise import RowwiseAdamW
from tests.test_lazy_adam import batch, batch_loss, model, rows_for


def test_rowwise_updates_equal_lazy_adamw_without_full_size_gradients():
    lazy_model, row_model = model(), model()
    lazy = LazyAdamW(lazy_model, lr=0.01, weight_decay=0.05)
    rowwise = RowwiseAdamW(row_model, lr=0.01, weight_decay=0.05)
    row_model.row_grads = rowwise.grads
    rng = np.random.RandomState(0)
    for _ in range(10):
        X, Y, N, random_songs = batch(rng)
        rows = rows_for(X, Y, N, random_songs)
        for m, opt in ((lazy_model, lazy), (row_model, rowwise)):
            opt.zero_grad()
            batch_loss(m, X, Y, N, random_songs).backward()
            opt.step(rows)
        for name, p in row_model.named_parameters():
            if is_table(name):
                assert p.grad is None, name  # no full-size gradient was ever built
    for (name, a), (_, b) in zip(lazy_model.named_parameters(), row_model.named_parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)


def test_lookups_give_the_same_values_with_or_without_row_gradients():
    m = model()
    X, Y, N, random_songs = batch(np.random.RandomState(1))
    before = batch_loss(m, X, Y, N, random_songs)
    m.row_grads = {}
    torch.testing.assert_close(batch_loss(m, X, Y, N, random_songs), before)
