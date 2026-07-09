from __future__ import annotations

from typing import Any

from ntlpol.extractors.gee_common import submit_drive_table_export

GHSL_POP_DATASET = "JRC/GHSL/P2023A/GHS_POP"
GHSL_BUILT_DATASET = "JRC/GHSL/P2023A/GHS_BUILT_S"


def nearest_ghsl_epoch(year: int) -> int:
    epochs = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
    return min(epochs, key=lambda e: abs(e - int(year)))


def build_ghsl_static_image(
    ee: Any,
    *,
    epoch_year: int = 2020,
    pop_dataset_id: str = GHSL_POP_DATASET,
    built_dataset_id: str = GHSL_BUILT_DATASET,
) -> Any:
    epoch = nearest_ghsl_epoch(epoch_year)
    pop = ee.Image(f"{pop_dataset_id}/{epoch}").select("population_count").rename("population_count_sum")
    built = ee.Image(f"{built_dataset_id}/{epoch}").select("built_surface").rename("built_surface_sum_m2")
    return pop.addBands([built]).set({"ghsl_epoch_year": epoch})


def build_ghsl_grid_static_collection(
    ee: Any,
    *,
    grid_fc: Any,
    epoch_year: int = 2020,
    scale_m: int = 100,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    pop_dataset_id: str = GHSL_POP_DATASET,
    built_dataset_id: str = GHSL_BUILT_DATASET,
) -> Any:
    img = build_ghsl_static_image(
        ee,
        epoch_year=epoch_year,
        pop_dataset_id=pop_dataset_id,
        built_dataset_id=built_dataset_id,
    )
    reduced = img.reduceRegions(
        collection=grid_fc,
        reducer=ee.Reducer.sum(),
        scale=scale_m,
        tileScale=tile_scale,
    )

    def add_density_and_ratio(f: Any) -> Any:
        area_m2 = ee.Number(f.geometry().area(1))
        area_km2 = area_m2.divide(1_000_000)
        pop = ee.Number(f.get("population_count_sum"))
        built = ee.Number(f.get("built_surface_sum_m2"))
        return f.set(
            {
                "grid_id": f.get(id_property),
                "population_density": pop.divide(area_km2.max(1e-6)),
                "built_up_ratio": built.divide(area_m2.max(1)).min(1),
                "ghsl_epoch_year": nearest_ghsl_epoch(epoch_year),
            }
        )

    return reduced.map(add_density_and_ratio)


def export_ghsl_grid_static_to_drive(
    ee: Any,
    *,
    grid_fc: Any,
    drive_folder: str,
    epoch_year: int = 2020,
    file_name_prefix: str = "grid_static_ghsl_features",
    description: str = "export_grid_static_ghsl_features",
    scale_m: int = 100,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    pop_dataset_id: str = GHSL_POP_DATASET,
    built_dataset_id: str = GHSL_BUILT_DATASET,
) -> Any:
    fc = build_ghsl_grid_static_collection(
        ee,
        grid_fc=grid_fc,
        epoch_year=epoch_year,
        scale_m=scale_m,
        tile_scale=tile_scale,
        id_property=id_property,
        pop_dataset_id=pop_dataset_id,
        built_dataset_id=built_dataset_id,
    )
    selectors = [
        "grid_id",
        "population_count_sum",
        "built_surface_sum_m2",
        "population_density",
        "built_up_ratio",
        "ghsl_epoch_year",
    ]
    return submit_drive_table_export(
        ee=ee,
        collection=fc,
        description=description,
        folder=drive_folder,
        file_name_prefix=file_name_prefix,
        selectors=selectors,
    )
