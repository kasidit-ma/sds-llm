def test_make_plots_writes_four_pngs(tmp_path):
    from spec_bench.plots import make_plots

    scope_cols = ("heaps_beta,repetition_rate,pbe_p90_at_1,pbe_p90_at_2,pbe_p90_at_4,"
                  "pbe_mean_at_1,pbe_mean_at_2,pbe_mean_at_4")
    header = (
        "dataset,mode,n,K,acceptance_rate,actual_speedup,"
        + ",".join(f"{s}_{c}" for s in ("ctx", "gen") for c in scope_cols.split(","))
        + "\n"
    )
    feats = "0.5,0.3,1.0,1.0,1.0,1.2,1.4,1.6"
    lines = [header, f"a,baseline,0,4,0.0,1.0,{feats},{feats}\n"]
    for ds, bump in (("a", 0.0), ("b", 0.2)):
        for mode, boost in (("depth", 0.2), ("width", 0.0)):
            for n in (1, 2):
                acc = 0.2 + bump + boost + 0.1 * n
                lines.append(f"{ds},{mode},{n},4,{acc},{1 + 3 * acc},{feats},{feats}\n")
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text("".join(lines))

    written = make_plots(str(csv_path), out_dir=str(tmp_path / "plots"))
    assert len(written) == 7
    for path in written:
        assert path.exists() and path.stat().st_size > 0
