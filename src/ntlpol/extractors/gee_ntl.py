from __future__ import annotations

from typing import Any

from ntlpol.extractors.gee_common import month_starts, submit_drive_table_export

VNP46A2_DATASET = "NASA/VIIRS/002/VNP46A2"
RAW_BAND = "DNB_BRDF_Corrected_NTL"
GAP_FILLED_BAND = "Gap_Filled_DNB_BRDF_Corrected_NTL"
MANDATORY_QA_BAND = "Mandatory_Quality_Flag"
CLOUD_QA_BAND = "QF_Cloud_Mask"
SNOW_BAND = "Snow_Flag"


def _prep_vnp46a2_daily(ee: Any, image: Any, *, use_gap_filled: bool = True) -> Any:
    radiance_band = GAP_FILLED_BAND if use_gap_filled else RAW_BAND
    raw = image.select(radiance_band).rename("raw_radiance")

    mandatory = image.select(MANDATORY_QA_BAND)
    # 0 = high-quality persistent lights, 1 = high-quality ephemeral lights.
    good_mandatory = mandatory.lte(1)

    cloud_mask = image.select(CLOUD_QA_BAND)
    # Bits 6-7: 0 confident clear, 1 probably clear, 2 probably cloudy, 3 cloudy.
    cloud_state = cloud_mask.rightShift(6).bitwiseAnd(3)
    clear = cloud_state.lte(1)

    no_snow = image.select(SNOW_BAND).eq(0)
    good = good_mandatory.And(clear).And(no_snow).rename("valid_obs")

    clean = raw.updateMask(good).rename("cleaned_radiance")
    bad = good.Not().rename("cloud_or_quality_bad_count")
    return raw.addBands([clean, good.rename("valid_obs_count"), bad])



def build_daily_vnp46a2_image(
    ee: Any,
    *,
    date: Any,
    dataset_id: str = VNP46A2_DATASET,
    use_gap_filled: bool = True,
) -> Any:
    """Build one quality-masked daily VNP46A2 image with project-standard bands."""
    start = ee.Date(date)
    end = start.advance(1, "day")
    img = (
        ee.ImageCollection(dataset_id)
        .filterDate(start, end)
        .first()
    )
    img = _prep_vnp46a2_daily(ee, ee.Image(img), use_gap_filled=use_gap_filled)
    return img.set({"date": start.format("YYYY-MM-dd"), "source": "VNP46A2"})


def build_vnp46a2_grid_daily_collection_for_month(
    ee: Any,
    *,
    grid_fc: Any,
    year: int,
    month: int,
    scale_m: int = 500,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    dataset_id: str = VNP46A2_DATASET,
    use_gap_filled: bool = True,
) -> Any:
    """Return grid-day VNP46A2 rows for one calendar month.

    Important implementation detail:
    Some VNP46A2 months can have missing calendar days. The previous version
    iterated over every calendar day and used ImageCollection.first(); when a
    day had no image, first() was null and image.select(...) failed in GEE with
    "Parameter 'input' is required and may not be null".

    This version first obtains the actual available observation dates from the
    ImageCollection and reduces only those dates, so missing days are skipped
    safely. Monthly aggregation later uses valid_obs_count/coverage information.
    """
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    reducer = ee.Reducer.mean()

    month_ic = ee.ImageCollection(dataset_id).filterDate(start, end)

    # Build a server-side list of actually available dates in this month.
    # Using date strings avoids duplicate same-day timestamps and prevents
    # reduce_one_day from being called for missing days.
    date_strings = (
        month_ic.aggregate_array("system:time_start")
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
        .distinct()
        .sort()
    )

    def reduce_one_available_day(date_string: Any) -> Any:
        date = ee.Date.parse("YYYY-MM-dd", date_string)
        date_str = date.format("YYYY-MM-dd")
        year_month = date.format("YYYY-MM")

        # Mosaic is safe here because the date is known to exist in month_ic.
        # It also handles rare same-day duplicates without returning null.
        daily_img = ee.ImageCollection(dataset_id).filterDate(
            date, date.advance(1, "day")
        ).mosaic()
        daily = _prep_vnp46a2_daily(
            ee, daily_img, use_gap_filled=use_gap_filled
        ).set({"date": date_str, "source": "VNP46A2"})

        fc = daily.reduceRegions(
            collection=grid_fc,
            reducer=reducer,
            scale=scale_m,
            tileScale=tile_scale,
        )
        return fc.map(
            lambda f: f.set(
                {
                    "grid_id": f.get(id_property),
                    "date": date_str,
                    "year_month": year_month,
                    "source": "VNP46A2",
                }
            )
        )

    return ee.FeatureCollection(date_strings.map(reduce_one_available_day)).flatten()


def export_vnp46a2_grid_daily_month_to_drive(
    ee: Any,
    *,
    grid_fc: Any,
    year: int,
    month: int,
    drive_folder: str,
    file_name_prefix: str | None = None,
    description: str | None = None,
    scale_m: int = 500,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    dataset_id: str = VNP46A2_DATASET,
    use_gap_filled: bool = True,
) -> Any:
    ym = f"{year:04d}_{month:02d}"
    fc = build_vnp46a2_grid_daily_collection_for_month(
        ee,
        grid_fc=grid_fc,
        year=year,
        month=month,
        scale_m=scale_m,
        tile_scale=tile_scale,
        id_property=id_property,
        dataset_id=dataset_id,
        use_gap_filled=use_gap_filled,
    )
    selectors = [
        "grid_id",
        "date",
        "year_month",
        "raw_radiance",
        "cleaned_radiance",
        "valid_obs_count",
        "cloud_or_quality_bad_count",
        "source",
    ]
    return submit_drive_table_export(
        ee=ee,
        collection=fc,
        description=description or f"export_vnp46a2_grid_daily_{ym}",
        folder=drive_folder,
        file_name_prefix=file_name_prefix or f"grid_daily_ntl_{ym}",
        selectors=selectors,
    )

def build_monthly_vnp46a2_image(
    ee: Any,
    *,
    year: int,
    month: int,
    dataset_id: str = VNP46A2_DATASET,
    use_gap_filled: bool = True,
) -> Any:
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    ic = (
        ee.ImageCollection(dataset_id)
        .filterDate(start, end)
        .map(lambda img: _prep_vnp46a2_daily(ee, img, use_gap_filled=use_gap_filled))
    )

    raw = ic.select("raw_radiance").median().rename("raw_radiance")
    clean = ic.select("cleaned_radiance").median().rename("cleaned_radiance")
    valid = ic.select("valid_obs_count").sum().rename("valid_obs_count")
    bad = ic.select("cloud_or_quality_bad_count").sum().rename("cloud_or_quality_bad_count")
    coverage = valid.divide(valid.add(bad).max(1)).rename("coverage_ratio")
    return raw.addBands([clean, valid, bad, coverage]).set(
        {"year_month": f"{year:04d}-{month:02d}", "source": "VNP46A2"}
    )


def build_vnp46a2_grid_monthly_collection(
    ee: Any,
    *,
    grid_fc: Any,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    scale_m: int = 500,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    dataset_id: str = VNP46A2_DATASET,
    use_gap_filled: bool = True,
) -> Any:
    collections = []
    reducer = ee.Reducer.mean()
    for year, month in month_starts(start_year, start_month, end_year, end_month):
        img = build_monthly_vnp46a2_image(
            ee,
            year=year,
            month=month,
            dataset_id=dataset_id,
            use_gap_filled=use_gap_filled,
        )
        ym = f"{year:04d}-{month:02d}"
        fc = img.reduceRegions(
            collection=grid_fc,
            reducer=reducer,
            scale=scale_m,
            tileScale=tile_scale,
        ).map(lambda f: f.set({"year_month": ym, "source": "VNP46A2", "grid_id": f.get(id_property)}))
        collections.append(fc)
    return ee.FeatureCollection(collections).flatten()


def export_vnp46a2_grid_monthly_to_drive(
    ee: Any,
    *,
    grid_fc: Any,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    drive_folder: str,
    file_name_prefix: str = "grid_monthly_ntl",
    description: str = "export_vnp46a2_grid_monthly_ntl",
    scale_m: int = 500,
    tile_scale: int = 4,
    id_property: str = "grid_id",
    dataset_id: str = VNP46A2_DATASET,
    use_gap_filled: bool = True,
) -> Any:
    fc = build_vnp46a2_grid_monthly_collection(
        ee,
        grid_fc=grid_fc,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        scale_m=scale_m,
        tile_scale=tile_scale,
        id_property=id_property,
        dataset_id=dataset_id,
        use_gap_filled=use_gap_filled,
    )
    selectors = [
        "grid_id",
        "year_month",
        "raw_radiance",
        "cleaned_radiance",
        "valid_obs_count",
        "cloud_or_quality_bad_count",
        "coverage_ratio",
        "source",
    ]
    return submit_drive_table_export(
        ee=ee,
        collection=fc,
        description=description,
        folder=drive_folder,
        file_name_prefix=file_name_prefix,
        selectors=selectors,
    )
