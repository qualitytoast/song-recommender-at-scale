"""Row-only table updates: lazy AdamW (recsys/lazy_adam.py) without full-size gradients.

A batch uses a few thousand rows of each per-ID table (about 2% of the songs at 200k), yet
PyTorch's backward pass for a table lookup builds a gradient the size of the whole table, mostly
zeros, which lazy AdamW then reads the used rows out of. That work was nearly 40% of a training
step at 200k. Here a table lookup (RowLookup) hands just the looked-up rows' gradients to a
collector in its backward pass; RowwiseAdamW adds them up per used row and applies lazy AdamW's
update to those rows. The same update, up to the order numbers are added in, without full-size
gradients. Large recommender systems train their item tables the same way (e.g. TorchRec's
fused embedding optimizer).

    optimizer = RowwiseAdamW(model, lr, weight_decay)
    model.row_grads = optimizer.grads   # the model's table lookups now feed the optimizer
    loss.backward(); optimizer.step(used_rows(...))
"""
import torch

from recsys.lazy_adam import LazyAdamW


class RowLookup(torch.autograd.Function):
    """weight[ids] whose backward passes (ids, row gradients) to grads[name] instead of returning
    a full-size gradient for weight. padded: ID 0 is padding and gets no gradient (padding_idx)."""

    @staticmethod
    def forward(ctx, weight, ids, name, grads, padded):
        ctx.save_for_backward(ids)
        ctx.name, ctx.grads, ctx.padded, ctx.row_shape = name, grads, padded, weight.shape[1:]
        return weight[ids]

    @staticmethod
    def backward(ctx, grad):
        (ids,) = ctx.saved_tensors
        ids = ids.reshape(-1)
        grad = grad.reshape(len(ids), *ctx.row_shape)
        if ctx.padded:  # multiply, not drop: dropping would make the GPU report back how many are left
            grad = grad * (ids != 0).reshape(-1, *[1] * len(ctx.row_shape))
        ctx.grads.setdefault(ctx.name, []).append((ids, grad))
        return None, None, None, None, None


class RowwiseAdamW(LazyAdamW):
    """LazyAdamW whose table gradients come from RowLookup's collector (self.grads), not .grad."""

    def __init__(self, model, lr, weight_decay, **kwargs):
        super().__init__(model, lr, weight_decay, **kwargs)
        self.grads = {}  # {table name: [(ids, row gradients), ...]}, filled by the backward pass

    def zero_grad(self):
        super().zero_grad()
        self.grads.clear()

    def table_gradient(self, name, p, rows):
        """The summed gradient of each row in rows (sorted unique IDs), from the collector."""
        parts = self.grads.pop(name, None)
        if not parts:
            return None
        ids = torch.cat([i for i, _ in parts])
        values = torch.cat([g for _, g in parts])
        local = torch.searchsorted(rows, ids)  # each contribution's place among the used rows
        return torch.zeros(len(rows), *p.shape[1:], dtype=values.dtype, device=values.device).index_add_(
            0, local, values)
