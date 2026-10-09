"""Extra inputs for the second-stage ranker: numbers describing each candidate and how it
relates to the playlist. The ranker's correction layer reads them next to the transformer's
output for the candidate (recsys/ranker.py). Computed in torch on the device, a batch at a time.

Groups (RankerConfig.features), their columns in this order:
  retriever_score  score_gap: the candidate's retriever score minus the best score in its
                   shortlist; log_rank: log of its place in the shortlist (K + 1 if not in it)
  popularity       log_popularity: log(1 + times the song is a next song in part-A playlists)
  overlap          against the 10 context songs: same_artist and same_album (share of context
                   songs with the candidate's artist / album), genre_match (for each of the
                   candidate's genres, the share of context songs that have it, averaged),
                   last_same_artist (1 if the last context song has the candidate's artist)
  cooccurrence     from part-A playlists, how often the candidate came 1 to PAIR_WINDOW songs
                   after a song: log_after_last (after the last context song), log_after_context
                   (after the 10 context songs, summed), context_with_pair (share of context
                   songs it ever came after)
  history          overlap and cooccurrence over the playlist so far (its last HISTORY songs,
                   the 10 context songs included) instead of the 10 context songs, plus
                   log_hist_len: log(1 + how many songs that is)
  neighbours       of the NEIGHBOURS part-A playlists most like this one (neighbour_lists),
                   the share that contain the candidate: nbr_share_50 (all of them) and
                   nbr_share_10 (the 10 most alike)
  song_length      length_gap: how far the candidate's length is from the context songs'
                   typical length, |log(its length) - mean log(their lengths)|
  playlist_length  log_playlist_len: log(1 + songs in the playlist so far). The same for every
                   candidate of a window, so only a correction that combines inputs can use it

Counts and neighbours come from part-A playlists only (the retriever's training playlists).
The ranker trains on part B; counting part B would put the very transitions it is asked to
predict into its inputs.
"""
import numpy as np
import torch

GROUPS = {
    "retriever_score": ["score_gap", "log_rank"],
    "popularity": ["log_popularity"],
    "overlap": ["same_artist", "same_album", "genre_match", "last_same_artist"],
    "cooccurrence": ["log_after_last", "log_after_context", "context_with_pair"],
    "history": ["hist_same_artist", "hist_same_album", "hist_genre_match", "hist_log_after", "hist_with_pair",
                "log_hist_len"],
    "neighbours": ["nbr_share_50", "nbr_share_10"],
    "song_length": ["length_gap"],
    "playlist_length": ["log_playlist_len"],
}
PAIR_WINDOW = 5   # a pair (a, b): song b comes 1 to 5 songs after song a in a playlist
HISTORY = 100     # history inputs look at up to this many of the playlist's most recent songs
NEIGHBOURS = 50   # neighbour inputs look at this many of the most alike part-A playlists


def pair_counts(songs, offsets, vocab_size, window=PAIR_WINDOW):
    """(keys, counts), keys sorted: key a * vocab_size + b; counts[i] is how often song b comes
    1 to `window` songs after song a in the same playlist. songs: playlists concatenated (vocab
    IDs, -1 = outside the vocab, never counted); offsets: where each starts, plus the end."""
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    keys = []
    for d in range(1, window + 1):
        a, b = songs[:-d], songs[d:]
        ok = (a >= 0) & (b >= 0) & (playlist[:-d] == playlist[d:])
        keys.append(a[ok] * vocab_size + b[ok])
    return np.unique(np.concatenate(keys), return_counts=True)


def playlist_vectors(songs, offsets, vectors):
    """(playlists, d) float32: each playlist's average song vector (recsys.search.song_vectors)
    over its songs in the vocab (songs: playlists concatenated, -1 = outside the vocab). A
    window's query vector . this = the retriever's average score for the playlist's songs."""
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    known = songs >= 0
    sums = torch.zeros(len(offsets) - 1, vectors.shape[1]).index_add_(
        0, torch.from_numpy(playlist[known]), torch.from_numpy(vectors[songs[known]]))
    counts = np.bincount(playlist[known], minlength=len(offsets) - 1).clip(min=1)
    return (sums / torch.from_numpy(counts)[:, None].float()).numpy()


def membership_keys(songs, offsets, vocab_size):
    """Sorted keys playlist * vocab_size + song, one per (playlist, song in it) pair."""
    playlist = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    known = songs >= 0
    return np.unique(playlist[known] * vocab_size + songs[known])


def neighbour_lists(queries, playlists, device, k=NEIGHBOURS):
    """(windows, k) int32: for each query vector (recsys.search.query_vectors), the k playlists
    (rows of playlist_vectors) it scores highest, most alike first."""
    from recsys.search import ExactSearch  # here, so this module's other parts don't need search
    return ExactSearch(playlists, device).search(queries, k)[0].astype(np.int32)


def history_matrix(histories, rows, length=HISTORY):
    """(len(rows), length) song IDs: the last `length` songs of each window's playlist so far,
    right-aligned (the last column is the window's last context song), -1 before the playlist
    starts and for songs outside the vocab. histories: (songs, begin, end) (data.window_histories)."""
    songs, begin, end = histories
    idx = end[rows, None] - length + np.arange(length)
    return np.where(idx >= begin[rows, None], songs[np.maximum(idx, 0)], -1)


class CandidateFeatures:
    """Computes the chosen groups' columns for batches of windows and their candidates."""

    def __init__(self, groups, retriever, popularity, pairs, device, durations_ms=None, members=None):
        """popularity (vocab,) counts; pairs (keys, counts) from pair_counts; durations_ms (vocab,)
        song lengths, for song_length; members from membership_keys, for neighbours."""
        unknown = sorted(set(groups) - set(GROUPS))
        if unknown:
            raise ValueError(f"unknown ranker feature groups {unknown}; known: {list(GROUPS)}")
        self.groups = [g for g in GROUPS if g in groups]  # always in GROUPS order
        self.names = [name for g in self.groups for name in GROUPS[g]]
        needs_songs = {"overlap", "history"} & set(self.groups)
        if needs_songs and not ({"artist", "album"} <= set(retriever.feature_names)
                                and retriever.song_genres is not None):
            raise ValueError(f"{sorted(needs_songs)} need a retriever with artist, album and genre features")
        if needs_songs:
            self.artist = retriever.song_feature_ids("artist").to(device)
            self.album = retriever.song_feature_ids("album").to(device)
            self.genres = retriever.song_genres.to(device)
            self.genre_count = int(self.genres.max()) + 1
        self.vocab_size = retriever.output.out_features
        self.log_popularity = torch.log1p(torch.as_tensor(popularity, dtype=torch.float32)).to(device)
        self.keys = torch.as_tensor(pairs[0], dtype=torch.int64).to(device)
        self.counts = torch.as_tensor(pairs[1], dtype=torch.float32).to(device)
        if "song_length" in self.groups:  # at least 1 s, so a missing length can't be log(0)
            self.log_length = torch.log(torch.as_tensor(durations_ms, dtype=torch.float32).clamp(min=1000)).to(device)
        if "neighbours" in self.groups:
            self.members = torch.as_tensor(members, dtype=torch.int64).to(device)
        self.playlists = None  # part-A playlist vectors, set by the ranker when it first finds neighbours

    def __call__(self, context, cands, base, ranks, top, history=None, neighbours=None, playlist_len=None):
        """context (b, L) song IDs; cands (b, C) song IDs; base (b, C) their retriever scores;
        ranks (b, C) their places in the shortlist (K + 1 if not in it); top (b,) the shortlist's
        best score; history (b, HISTORY) song IDs (history_matrix) if "history" is used;
        neighbours (b, NEIGHBOURS) playlist IDs (neighbour_lists) if "neighbours" is used;
        playlist_len (b,) songs in the playlist so far if "playlist_length" is used.
        Returns (b, C, len(self.names)) float32."""
        cols = []
        if "retriever_score" in self.groups:
            cols += [base - top[:, None], torch.log(ranks.float())]
        if "popularity" in self.groups:
            cols.append(self.log_popularity[cands])
        if "overlap" in self.groups:
            artist, album, genre = self.overlap(context, cands)
            cols += [artist, album, genre, (self.artist[cands] == self.artist[context[:, -1:]]).float()]
        if "cooccurrence" in self.groups:
            after = self.after_counts(context, cands)  # (b, C, L)
            cols += [torch.log1p(after[..., -1]), torch.log1p(after.sum(-1)), (after > 0).float().mean(-1)]
        if "history" in self.groups:
            n = (history >= 0).sum(1, keepdim=True).clamp(min=1).float()  # (b, 1) real songs
            artist, album, genre = self.overlap(history, cands)
            after = self.after_counts(history, cands)
            cols += [artist, album, genre, torch.log1p(after.sum(-1)), (after > 0).float().sum(-1) / n,
                     torch.log1p(n).expand_as(base)]
        if "neighbours" in self.groups:
            query = (neighbours.long()[:, None, :] * self.vocab_size + cands[:, :, None]).contiguous()
            idx = torch.searchsorted(self.members, query).clamp(max=len(self.members) - 1)
            contains = (self.members[idx] == query).float()  # (b, C, NEIGHBOURS), most alike first
            cols += [contains.mean(-1), contains[..., :10].mean(-1)]
        if "song_length" in self.groups:
            typical = self.log_length[context].mean(-1, keepdim=True)  # (b, 1)
            cols.append((self.log_length[cands] - typical).abs())
        if "playlist_length" in self.groups:
            cols.append(torch.log1p(playlist_len.float())[:, None].expand_as(base))
        return torch.stack(cols, dim=-1).float()

    def overlap(self, songs, cands):
        """Against songs (b, M) (-1 = no song): each candidate's (same_artist, same_album,
        genre_match), each (b, C)."""
        real = songs >= 0
        n = real.sum(1, keepdim=True).clamp(min=1).float()  # (b, 1)
        s = songs.clamp(min=0)

        def same(table):  # share of real songs whose value equals the candidate's
            return ((table[s][:, None, :] == table[cands][:, :, None]) & real[:, None, :]).sum(-1) / n

        # Share of the songs having each genre (b, genres); genre 0 is padding and stays 0.
        genres = (self.genres[s] * real[..., None]).flatten(1)
        share = torch.zeros(len(songs), self.genre_count, device=songs.device)
        share.scatter_add_(1, genres, torch.ones_like(genres, dtype=torch.float32))
        share[:, 0] = 0
        share /= n
        cand_genres = self.genres[cands]  # (b, C, k)
        has = (cand_genres > 0).float()
        matched = share.gather(1, cand_genres.flatten(1)).view(cand_genres.shape)
        genre_match = (matched * has).sum(-1) / has.sum(-1).clamp(min=1)
        return same(self.artist), same(self.album), genre_match

    def after_counts(self, songs, cands):
        """(b, C, M): how often each candidate came 1 to PAIR_WINDOW songs after each of songs
        (b, M) in part-A playlists; 0 for -1 (no song)."""
        if len(self.keys) == 0:
            return torch.zeros(*cands.shape, songs.shape[1], device=cands.device)
        query = (songs[:, None, :] * self.vocab_size + cands[:, :, None]).contiguous()
        idx = torch.searchsorted(self.keys, query).clamp(max=len(self.keys) - 1)
        found = (self.keys[idx] == query) & (songs[:, None, :] >= 0)
        return torch.where(found, self.counts[idx], torch.zeros((), device=cands.device))
