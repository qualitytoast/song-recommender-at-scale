import numpy as np
import torch

from recsys.lazy_adam import LazyAdamW, is_table, used_rows
from recsys.model import SongRecommender
from recsys.sampled import candidate_set, log_q, random_probs, sampled_softmax_loss

ARTIST = np.array([0, 0, 1, 2, 2, 2, 1, 3, 3, 1])
GENRES = np.array([[1, 2], [1, 0], [2, 0], [0, 0], [1, 2], [2, 0], [1, 0], [3, 0], [3, 1], [0, 0]])
VOCAB = 10


def model(seed=0):
    torch.manual_seed(seed)
    return SongRecommender(vocab_size=VOCAB, embed_dim=4, context_length=3, num_layers=1, dropout=0.0,
                           scale_attention=True, init="pytorch", song_features={"artist": ARTIST},
                           name_word_count=4, song_genres=GENRES, causal=True)


def batch_loss(m, X, Y, N, random_songs):
    """The p3 training loss for one batch: every position, sampled softmax."""
    Xt, Yt, Nt = (torch.from_numpy(a) for a in (X, Y, N))
    cand, real = candidate_set(Yt, len(random_songs), VOCAB, None, random_songs=torch.from_numpy(random_songs))
    freq = torch.full((VOCAB,), 1 / VOCAB)
    logits = m(Xt, Nt, all_positions=True, candidates=cand)
    return sampled_softmax_loss(logits, Yt, cand, real,
                                log_q(cand, (Yt != -100).sum(), freq, len(random_songs), random_probs(freq, 0)))


def batch(rng):
    X = rng.randint(0, VOCAB, (2, 3))
    Y = rng.randint(0, VOCAB, (2, 3))
    Y[1, 2] = -100  # padding
    return X, Y, rng.randint(0, 5, (2, 2)), rng.randint(0, VOCAB, 3)


def rows_for(X, Y, N, random_songs):
    candidates = np.concatenate([np.clip(Y.ravel(), 0, None), random_songs])
    return used_rows(X, candidates, N, {"artist": ARTIST}, GENRES, "cpu")


def test_no_gradient_is_ever_dropped():
    # Every table row that gets a nonzero gradient must be among the rows the CPU says were used.
    m, rng = model(), np.random.RandomState(0)
    for _ in range(20):
        X, Y, N, random_songs = batch(rng)
        m.zero_grad()
        batch_loss(m, X, Y, N, random_songs).backward()
        rows = rows_for(X, Y, N, random_songs)
        for name, p in m.named_parameters():
            if is_table(name):
                touched = set(torch.nonzero(p.grad.reshape(len(p.grad), -1).abs().sum(1)).flatten().tolist())
                assert touched <= set(rows[name].tolist()), name


def test_with_every_row_used_it_is_exactly_adamw():
    a, b = model(), model()
    ref = torch.optim.AdamW(a.parameters(), lr=0.01, weight_decay=0.05)
    lazy = LazyAdamW(b, lr=0.01, weight_decay=0.05)
    every = {n: torch.arange(len(p)) for n, p in b.named_parameters() if is_table(n)}
    rng = np.random.RandomState(1)
    for _ in range(5):
        X, Y, N, random_songs = batch(rng)
        ref.zero_grad(); lazy.zero_grad()
        batch_loss(a, X, Y, N, random_songs).backward()
        batch_loss(b, X, Y, N, random_songs).backward()
        ref.step(); lazy.step(every)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pa, pb, msg=n)


def test_unused_rows_never_change_and_used_rows_do():
    m = model()
    lazy = LazyAdamW(m, lr=0.01, weight_decay=0.05)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    X, Y, N = np.array([[1, 2, 3]]), np.array([[2, 3, 4]]), np.array([[1, 0]])
    random_songs = np.array([5])
    lazy.zero_grad()
    batch_loss(m, X, Y, N, random_songs).backward()
    rows = rows_for(X, Y, N, random_songs)
    lazy.step(rows)
    song_rows = set(rows["song_embedding.weight"].tolist())          # {1, 2, 3}
    for i in range(VOCAB):
        same = torch.equal(m.song_embedding.weight[i], before["song_embedding.weight"][i])
        assert same == (i not in song_rows)  # used rows move (decay alone moves them), others don't
    assert torch.equal(m.output.weight[9], before["output.weight"][9])  # song 9 wasn't a candidate
    assert not torch.equal(m.position_embedding.weight, before["position_embedding.weight"])  # shared: AdamW
