from recsys.summarize import mean_range, table


def result(ndcg, epoch=5, params=1000):
    return {"ndcg": ndcg, "hits10": 0.07, "hits1": 0.03, "best_epoch": epoch,
            "minutes": 3.0, "params": params}


def test_mean_range():
    assert mean_range([0.01, 0.02, 0.06], lambda x: f"{x:.2f}") == "0.03 (0.01–0.06)"


def test_change_is_difference_of_means_from_row_above():
    lines = table([("base", [result(0.050), result(0.052), result(0.054)]),   # mean 0.052
                   ("artist", [result(0.055), result(0.057), result(0.059)])])  # mean 0.057
    base, artist = lines[2], lines[3]
    assert "| — |" in base
    assert "0.0520 (0.0500–0.0540)" in base
    assert "| +0.0050 |" in artist


def test_lists_each_seeds_best_epoch_and_handles_missing_params():
    line = table([("old", [result(0.05, epoch=4, params=None), result(0.05, epoch=7, params=None)])])[2]
    assert "| 4, 7 |" in line and "| — |" in line
