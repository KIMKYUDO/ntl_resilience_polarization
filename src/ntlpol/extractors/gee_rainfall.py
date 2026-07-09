from __future__ import annotations

from typing import Any

from ntlpol.extractors.gee_common import submit_drive_table_export

GPM_IMERG_DATASET = "NASA/GPM_L3/IMERG_V07"
PRECIP_BAND = "precipitation"


def _accum_mm_image(ee: Any, *, start_date: str, end_date: str, band: str, name: str, dataset_id: str) -> Any:
    start = ee.Date(start_date)
    # End date is treated as inclusive at day level by advancing one day.
    end = ee.Date(end_date).advance(1, "day")
    ic = ee.ImageCollection(dataset_id).filterDate(start, end).select(band)
    # IMERG V07 precipitation is mm/hr at 30-minute cadence, so multiply by 0.5 hour.
    return ic.map(lambda img: img.multiply(0.5)).sum().rename(name)


def build_gpm_event_accumulation_image(
    ee: Any,
    *,
    start_date: str,
    end_date: str,
    dataset_id: str = GPM_IMERG_DATASET,
    band: str = PRECIP_BAND,
) -> Any:
    during = _accum_mm_image(
        ee,
        start_date=start_date,
        end_date=end_date,
        band=band,
        name="rainfall_accum_event_mm",
        dataset_id=dataset_id,
    )
    pre_start = ee.Date(start_date).advance(-3, "day").format("YYYY-MM-dd")
    pre_end = ee.Date(start_date).advance(-1, "day").format("YYYY-MM-dd")
    post_start = ee.Date(end_date).advance(1, "day").format("YYYY-MM-dd")
    post_end = ee.Date(end_date).advance(7, "day").format("YYYY-MM-dd")
    pre3 = _accum_mm_image(
        ee,
        start_date=pre_start,
        end_date=pre_end,
        band=band,
        name="rainfall_accum_pre_event_3d_mm",
        dataset_id=dataset_id,
    )
    post7 = _accum_mm_image(
        ee,
        start_date=post_start,
        end_date=post_end,
        band=band,
        name="rainfall_accum_post_event_7d_mm",
        dataset_id=dataset_id,
    )
    return during.addBands([pre3, post7])


def build_gpm_grid_event_collection(
    ee: Any,
    *,
    grid_fc: Any,
    events: list[dict[str, str]],
    scale_m: int = 11132,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    dataset_id: str = GPM_IMERG_DATASET,
    band: str = PRECIP_BAND,
) -> Any:
    collections = []
    reducer = ee.Reducer.mean()
    for event in events:
        event_id = event["event_id"]
        img = build_gpm_event_accumulation_image(
            ee,
            start_date=event["start_date"],
            end_date=event["end_date"],
            dataset_id=dataset_id,
            band=band,
        )
        fc = img.reduceRegions(
            collection=grid_fc,
            reducer=reducer,
            scale=scale_m,
            tileScale=tile_scale,
        ).map(lambda f: f.set({"event_id": event_id, "grid_id": f.get(id_property)}))
        collections.append(fc)
    return ee.FeatureCollection(collections).flatten()


def export_gpm_grid_event_rainfall_to_drive(
    ee: Any,
    *,
    grid_fc: Any,
    events: list[dict[str, str]],
    drive_folder: str,
    file_name_prefix: str = "grid_event_gpm_rainfall",
    description: str = "export_gpm_grid_event_rainfall",
    scale_m: int = 11132,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    dataset_id: str = GPM_IMERG_DATASET,
    band: str = PRECIP_BAND,
) -> Any:
    fc = build_gpm_grid_event_collection(
        ee,
        grid_fc=grid_fc,
        events=events,
        scale_m=scale_m,
        tile_scale=tile_scale,
        id_property=id_property,
        dataset_id=dataset_id,
        band=band,
    )
    selectors = [
        "event_id",
        "grid_id",
        "rainfall_accum_event_mm",
        "rainfall_accum_pre_event_3d_mm",
        "rainfall_accum_post_event_7d_mm",
    ]
    return submit_drive_table_export(
        ee=ee,
        collection=fc,
        description=description,
        folder=drive_folder,
        file_name_prefix=file_name_prefix,
        selectors=selectors,
    )
