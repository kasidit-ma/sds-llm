def test_estimated_speedup():
    from estimation.speedup_estimator import estimated_speedup

    assert estimated_speedup(0.5, K=10) > 1.0


def test_decision_policy():
    from decision.policy import decide

    out = decide({"pbe_p90_at_1": 1.0}, estimated_speedup=1.8)
    assert out["use_speculative"] is True
    assert out["mode"] == "depth"


def test_evaluate_perfect_prediction():
    from estimation.evaluator import evaluate

    out = evaluate([1.0, 2.0], [1.0, 2.0])
    assert out["mae"] == 0.0
    assert out["r2"] == 1.0


def test_acceptance_model_fit_predict():
    from estimation.acceptance_model import AcceptanceModel

    # acceptance rises with repetition_rate (first feature)
    features = [[0.1, 0.5, 1.0], [0.3, 0.5, 1.0], [0.5, 0.5, 1.0],
                [0.7, 0.5, 1.0], [0.9, 0.5, 1.0]]
    acceptance = [0.1, 0.3, 0.5, 0.7, 0.9]
    model = AcceptanceModel().fit(features, acceptance)
    lo, hi = model.predict([0.1, 0.5, 1.0]), model.predict([0.9, 0.5, 1.0])
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi > lo


def _bench_csv(tmp_path):
    cols = ("heaps_beta,repetition_rate,pbe_p90_at_1,pbe_p90_at_2,pbe_p90_at_4,"
            "pbe_mean_at_1,pbe_mean_at_2,pbe_mean_at_4")
    header = ("dataset,mode,n,K,acceptance_rate,actual_speedup,"
              + ",".join(f"{s}_{c}" for s in ("ctx", "gen") for c in cols.split(",")) + "\n")
    feats_a = "0.5,0.3,1.0,1.0,1.0,1.2,1.4,1.6"
    feats_b = "0.7,0.2,2.0,2.0,2.0,1.8,2.0,2.2"
    path = tmp_path / "bench.csv"
    path.write_text(
        header
        + f"a,baseline,0,4,0.0,1.0,{feats_a},{feats_a}\n"
        + f"a,depth,2,4,0.6,2.5,{feats_a},{feats_a}\n"
        + f"a,width,2,4,0.4,1.8,{feats_a},{feats_a}\n"
        + f"b,depth,2,4,0.3,1.4,{feats_b},{feats_b}\n"
        + f"b,width,2,4,0.2,1.3,{feats_b},{feats_b}\n"
    )
    return path


def test_fit_from_benchmark(tmp_path):
    from estimation.curve_fitting import fit_from_benchmark

    path = _bench_csv(tmp_path)
    for scope in ("ctx", "gen"):
        out = fit_from_benchmark(str(path), scope=scope)
        assert "theta" in out and "mae" in out
        assert out["scope"] == scope
        assert out["n_rows"] == 4  # baseline row excluded


def test_fit_linear_speedup(tmp_path):
    from estimation.curve_fitting import fit_linear_speedup

    out = fit_linear_speedup(str(_bench_csv(tmp_path)), mode="depth", n=2)
    assert "beta" in out["formula"] and len(out["table"]) == 2
    # 2 points, 3 params -> lstsq reproduces the data exactly
    for row in out["table"]:
        assert abs(row["predict"] - row["speedup"]) < 1e-6
