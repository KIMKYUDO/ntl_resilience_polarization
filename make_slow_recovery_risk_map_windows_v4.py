# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import pandas as pd

def try_read_table(path: Path):
    if not path.exists():
        return None, "missing"
    if path.stat().st_size <= 1:
        return None, "empty-file"
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            return None, "unsupported"
        if df.shape[0] == 0 or df.shape[1] == 0:
            return None, "empty-table shape=" + str(df.shape)
        if "grid_id" not in df.columns:
            return None, "no-grid_id columns=" + str(list(df.columns)[:20])
        if "pred_delayed_prob" not in df.columns:
            return None, "no-pred_delayed_prob columns=" + str(list(df.columns)[:20])
        return df, "ok shape=" + str(df.shape)
    except Exception as e:
        return None, "read-error " + type(e).__name__ + ": " + str(e)

def read_best_risk_table(root: Path):
    candidates = [
        root / "outputs" / "maps" / "slow_recovery_risk_map_ready.parquet",
        root / "outputs" / "maps" / "slow_recovery_risk_map_ready.csv",
        root / "outputs" / "final_bundle" / "slow_recovery_risk_map_ready.parquet",
        root / "outputs" / "final_bundle" / "slow_recovery_risk_map_ready.csv",
        root / "outputs" / "predictions" / "ensemble_predictions.parquet",
        root / "outputs" / "predictions" / "ensemble_predictions.csv",
        root / "outputs" / "final_bundle" / "ensemble_predictions.parquet",
        root / "outputs" / "final_bundle" / "ensemble_predictions.csv",
    ]

    print("=== candidate risk/prediction files ===")
    for p in candidates:
        size = p.stat().st_size if p.exists() else "NA"
        print(str(p), "exists=", p.exists(), "size=", size)

    print("\n=== trying candidates ===")
    for p in candidates:
        df, status = try_read_table(p)
        print(p.name, "=>", status)
        if df is not None:
            print("Using:", p)
            return df, p

    raise RuntimeError("No usable file found. Need a non-empty table with grid_id and pred_delayed_prob.")

def main():
    root = Path.cwd()
    grid_shp = root / "data" / "interim" / "grid" / "india_grid_5km.shp"
    out_map_dir = root / "outputs" / "maps"
    out_fig_dir = root / "outputs" / "figures"
    out_map_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Slow recovery risk map generation v4 ===")
    print("Project root:", root)

    if not grid_shp.exists():
        raise FileNotFoundError("Grid shapefile not found: " + str(grid_shp))

    risk, source_path = read_best_risk_table(root)
    print("\nRisk/prediction source:", source_path)
    print("Risk shape:", risk.shape)
    print("Risk columns:", list(risk.columns))

    try:
        import geopandas as gpd
    except Exception:
        print("[ERROR] Missing geopandas.")
        print("Run:")
        print("  conda install -c conda-forge geopandas pyogrio shapely fiona matplotlib -y")
        print("or:")
        print("  pip install geopandas pyogrio shapely fiona matplotlib")
        raise

    print("\nReading grid shapefile:", grid_shp)
    grid = gpd.read_file(grid_shp)
    print("Grid shape:", grid.shape)
    print("Grid columns:", list(grid.columns))
    print("Grid CRS:", grid.crs)

    if "grid_id" not in grid.columns:
        candidates = ["GRID_ID", "cell_id", "CELL_ID", "tile_id", "TILE_ID", "id", "ID"]
        found = None
        for c in candidates:
            if c in grid.columns:
                found = c
                break
        if found is None:
            raise KeyError("grid shapefile does not contain grid_id. Columns: " + str(list(grid.columns)))
        grid = grid.rename(columns={found: "grid_id"})
        print("Renamed grid column", found, "to grid_id")

    risk["grid_id"] = risk["grid_id"].astype(str)
    grid["grid_id"] = grid["grid_id"].astype(str)

    risk_ids = set(risk["grid_id"])
    grid_ids = set(grid["grid_id"])
    matched = len(risk_ids & grid_ids)

    print("\n=== grid_id match check ===")
    print("Unique risk grid_id:", len(risk_ids))
    print("Unique grid geometry grid_id:", len(grid_ids))
    print("Matched unique grid_id:", matched)

    if matched == 0:
        print("Risk grid_id examples:", list(risk["grid_id"].head(10)))
        print("Grid grid_id examples:", list(grid["grid_id"].head(10)))
        raise RuntimeError("No matched grid_id. Need ID format inspection.")

    gdf = risk.merge(grid[["grid_id", "geometry"]], on="grid_id", how="left")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=grid.crs)

    missing = int(gdf.geometry.isna().sum())
    print("Joined shape:", gdf.shape)
    print("Missing geometry rows:", missing)

    if missing >= len(gdf):
        raise RuntimeError("All joined rows have missing geometry.")

    if gdf.crs is None:
        print("Grid CRS missing. Setting EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    geojson_path = out_map_dir / "slow_recovery_risk_map.geojson"
    gpkg_path = out_map_dir / "slow_recovery_risk_map.gpkg"
    centroid_csv_path = out_map_dir / "slow_recovery_risk_map_with_centroid.csv"

    print("\nWriting joined map files...")
    gdf.to_file(geojson_path, driver="GeoJSON")
    gdf.to_file(gpkg_path, driver="GPKG")

    centroid_src = gdf.to_crs("EPSG:3857")
    centroids = centroid_src.geometry.centroid
    cent_gdf = gdf.copy()
    cent_gdf["geometry"] = gpd.GeoSeries(centroids, crs="EPSG:3857").to_crs("EPSG:4326")
    cent_gdf["lon"] = cent_gdf.geometry.x
    cent_gdf["lat"] = cent_gdf.geometry.y
    cent_gdf.drop(columns="geometry").to_csv(centroid_csv_path, index=False, encoding="utf-8-sig")

    print("GeoJSON:", geojson_path)
    print("GPKG:", gpkg_path)
    print("Centroid CSV:", centroid_csv_path)

    print("\nAggregating by grid_id for cleaner map figures...")
    agg = gdf.groupby("grid_id", as_index=False).agg(
        pred_delayed_prob=("pred_delayed_prob", "max"),
        pred_recovery_percentile=("pred_recovery_percentile", "mean"),
    )
    grid_map = grid.merge(agg, on="grid_id", how="left")
    grid_map = gpd.GeoDataFrame(grid_map, geometry="geometry", crs=grid.crs).to_crs("EPSG:4326")
    grid_map_path = out_map_dir / "slow_recovery_risk_grid_aggregated.geojson"
    grid_map.to_file(grid_map_path, driver="GeoJSON")
    print("Aggregated grid GeoJSON:", grid_map_path)

    print("\nCreating figures with non-GUI Agg backend...")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        print("[ERROR] Missing matplotlib.")
        print("Run:")
        print("  conda install -c conda-forge matplotlib -y")
        print("or:")
        print("  pip install matplotlib")
        raise

    fig1 = out_fig_dir / "map_predicted_slow_recovery_risk.png"
    fig2 = out_fig_dir / "map_top10_slow_recovery_risk.png"

    fig, ax = plt.subplots(figsize=(9, 11))
    grid_map.plot(
        column="pred_delayed_prob",
        ax=ax,
        legend=True,
        linewidth=0,
        cmap="YlOrRd",
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Predicted Slow-Recovery Risk from Early NTL Trajectories")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(fig1, dpi=250)
    plt.close()

    threshold = float(grid_map["pred_delayed_prob"].quantile(0.90))
    grid_map["top10_risk"] = grid_map["pred_delayed_prob"] >= threshold

    fig, ax = plt.subplots(figsize=(9, 11))
    grid_map.plot(ax=ax, color="lightgrey", linewidth=0)
    grid_map[grid_map["top10_risk"]].plot(
        ax=ax,
        column="pred_delayed_prob",
        legend=True,
        linewidth=0,
        cmap="Reds",
    )
    ax.set_title("Top 10% Predicted Slow-Recovery Risk Areas")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(fig2, dpi=250)
    plt.close()

    print("Figure 1:", fig1)
    print("Figure 2:", fig2)
    print("[DONE] Map generation completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FAILED]", type(e).__name__, str(e))
        sys.exit(1)
