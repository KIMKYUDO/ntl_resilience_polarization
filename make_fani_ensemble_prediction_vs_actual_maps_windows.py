# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import pandas as pd

EVENT_KEYWORD = "fani"
PRED_PATH = Path("outputs/predictions/ensemble_predictions.parquet")


def read_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Missing prediction file: " + str(path))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported prediction file type: " + str(path))


def main():
    root = Path.cwd()
    grid_path = root / "data" / "interim" / "grid" / "india_grid_5km.shp"
    out_dir = root / "outputs" / "figures" / "event_model_compare" / "fani_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fani ensemble predicted-vs-actual map generator ===")
    print("Project root:", root)

    try:
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        print("[ERROR] Missing packages.")
        print("Install one of these:")
        print("  conda install -c conda-forge geopandas pyogrio shapely fiona matplotlib -y")
        print("  pip install geopandas pyogrio shapely fiona matplotlib")
        raise

    if not grid_path.exists():
        raise FileNotFoundError("Missing grid shapefile: " + str(grid_path))

    print("Reading grid:", grid_path)
    grid = gpd.read_file(grid_path)
    if "grid_id" not in grid.columns:
        raise KeyError("Grid file does not contain grid_id. Columns: " + str(list(grid.columns)))
    grid["grid_id"] = grid["grid_id"].astype(str)

    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")
    grid = grid.to_crs("EPSG:4326")

    path = root / PRED_PATH
    print("Reading ensemble prediction:", path)
    df = read_prediction(path)

    required = {"event_id", "grid_id", "pred_delayed_prob", "y_delayed_slowest_20pct"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Ensemble prediction missing columns: {missing}. Columns: {list(df.columns)}")

    sub = df[df["event_id"].astype(str).str.contains(EVENT_KEYWORD, case=False, na=False)].copy()
    if sub.empty:
        print("Available event_id examples:", df["event_id"].drop_duplicates().head(20).tolist())
        raise RuntimeError(f"No event_id containing '{EVENT_KEYWORD}' found in {path}")

    sub["grid_id"] = sub["grid_id"].astype(str)
    sub = sub.sort_values("pred_delayed_prob", ascending=False).drop_duplicates("grid_id")

    print("Fani rows:", len(sub))
    print("Predicted risk mean:", float(sub["pred_delayed_prob"].mean()))
    print("Actual delayed rate:", float(sub["y_delayed_slowest_20pct"].mean()))

    gdf = grid.merge(
        sub[["grid_id", "pred_delayed_prob", "y_delayed_slowest_20pct"]],
        on="grid_id",
        how="left",
    )
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=grid.crs)

    matched = int(gdf["pred_delayed_prob"].notna().sum())
    print("Matched grids:", matched)

    pred_path = out_dir / "ensemble_fani_predicted_risk.png"
    actual_path = out_dir / "ensemble_fani_actual_delayed.png"
    panel_path = out_dir / "ensemble_fani_predicted_vs_actual.png"
    summary_path = out_dir / "ensemble_fani_map_summary.csv"
    geojson_path = out_dir / "ensemble_fani_predicted_vs_actual.geojson"

    fig, ax = plt.subplots(figsize=(9, 11))
    gdf.plot(
        column="pred_delayed_prob",
        ax=ax,
        legend=True,
        linewidth=0,
        cmap="YlOrRd",
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Fani 2019 predicted delayed-recovery risk: ensemble")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(pred_path, dpi=250)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 11))
    gdf.plot(ax=ax, color="lightgrey", linewidth=0)
    gdf[gdf["y_delayed_slowest_20pct"] == 1].plot(ax=ax, color="red", linewidth=0)
    ax.set_title("Fani 2019 actual delayed recovery label: ensemble")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(actual_path, dpi=250)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    gdf.plot(
        column="pred_delayed_prob",
        ax=axes[0],
        legend=True,
        linewidth=0,
        cmap="YlOrRd",
        missing_kwds={"color": "lightgrey"},
    )
    axes[0].set_title("Predicted risk: ensemble")
    axes[0].set_axis_off()

    gdf.plot(ax=axes[1], color="lightgrey", linewidth=0)
    gdf[gdf["y_delayed_slowest_20pct"] == 1].plot(ax=axes[1], color="red", linewidth=0)
    axes[1].set_title("Actual delayed label")
    axes[1].set_axis_off()

    fig.suptitle("Fani 2019 predicted vs actual delayed recovery: ensemble", fontsize=14)
    plt.tight_layout()
    plt.savefig(panel_path, dpi=250)
    plt.close()

    gdf.to_file(geojson_path, driver="GeoJSON")

    summary = pd.DataFrame([
        {
            "model": "ensemble",
            "event_keyword": EVENT_KEYWORD,
            "mapped_grids": matched,
            "actual_delayed_rate": float(sub["y_delayed_slowest_20pct"].mean()),
            "pred_delayed_prob_mean": float(sub["pred_delayed_prob"].mean()),
            "pred_delayed_prob_p90": float(sub["pred_delayed_prob"].quantile(0.90)),
            "pred_delayed_prob_min": float(sub["pred_delayed_prob"].min()),
            "pred_delayed_prob_max": float(sub["pred_delayed_prob"].max()),
        }
    ])
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Predicted map:", pred_path)
    print("Actual map:", actual_path)
    print("Panel map:", panel_path)
    print("GeoJSON:", geojson_path)
    print("Summary:", summary_path)
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FAILED]", type(e).__name__, str(e))
        sys.exit(1)
