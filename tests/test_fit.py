from MOBSTERm.create_synthetic_dataset import generate_synthetic_data
from MOBSTERm.model_mobster import fit


def test_fit_smoke():
    NV, DP, *_ = generate_synthetic_data(
        N=500, K=5, D=2, purity=[1, 1], coverage=50, seed=11
    )

    result = fit(
        NV=NV, DP=DP,
        num_iter=30, lr=0.01,
        mut_id=[f"M{i}" for i in range(1, NV.shape[0] + 1)],
        seed_list=[11,22], K=[4, 5],
        num_of_threads=1, quiet=True,
    )

    assert "best_fit" in result
    assert "runs" in result
    assert len(result["runs"]) == 2

    best_fit = result["best_fit"]

    assert len(best_fit["cluster_id"]) == NV.shape[0]
    assert len(best_fit["mutation_id"]) == NV.shape[0]
    assert best_fit["n_components"] in (4, 5)
    assert best_fit["seed"] in (11, 22)
