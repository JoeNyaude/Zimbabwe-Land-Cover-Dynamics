"""Google Earth Engine baseline analysis for annual Zimbabwe land cover."""
from datetime import date
from functools import lru_cache
import os

CLASS_NAMES = (
    "Water",
    "Forest",
    "Grass",
    "Flooded vegetation",
    "Crops",
    "Shrub and scrub",
    "Urban",
    "Bare ground",
    "Snow and ice",
)
CLASS_PALETTE = (
    "#419BDF",
    "#397D49",
    "#88B053",
    "#7A87C6",
    "#E49635",
    "#DFC35A",
    "#C4281B",
    "#A59B8F",
    "#B39FE1",
)
PIXEL_AREA_KM2 = 0.0001  # Dynamic World nominal 10 m pixels.
MAX_FOREST_LOSS_POINTS = 300


@lru_cache(maxsize=1)
def _earth_engine():
    """Initialize EE with the current user's ADC credentials and configured project."""
    from django.conf import settings

    project = getattr(settings, "EARTH_ENGINE_PROJECT", "")
    if not project:
        raise RuntimeError(
            "Set EARTH_ENGINE_PROJECT to a registered Google Cloud project, then "
            "authenticate Earth Engine for the account running Django."
        )
    try:
        import ee
    except ImportError as error:
        raise RuntimeError(
            "Earth Engine support is not installed. Install it with "
            "'pip install earthengine-api'."
        ) from error
    try:
        ee.Initialize(project=project)
    except Exception as error:
        raise RuntimeError(
            "Earth Engine could not initialize. Run 'earthengine authenticate' "
            "as the account running Django and check EARTH_ENGINE_PROJECT."
        ) from error
    return ee


def _annual_images(ee, geometry, year):
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    dynamic_world = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(geometry)
    )
    image_count = dynamic_world.size().getInfo()
    if not image_count:
        raise RuntimeError(f"Dynamic World has no scenes for {year} in this area.")

    # The mode label makes an annual categorical map from the per-scene labels.
    labels = (
        dynamic_world.select("label")
        .reduce(ee.Reducer.mode())
        .rename("land_cover")
        .clip(geometry)
    )

    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(geometry)
    )

    def mask_clouds(image):
        scl = image.select("SCL")
        clear = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(clear).select(["B4", "B3", "B2"])

    rgb = (
        sentinel.map(mask_clouds)
        .median()
        .clip(geometry)
    )
    return labels, rgb, image_count


def _tile_url(image, vis_params):
    map_info = image.getMapId(vis_params)
    return map_info["tile_fetcher"].url_format


def _summarize_transitions(histogram):
    """Summarize Dynamic World pixel transitions and Tree-class loss."""
    class_count = len(CLASS_NAMES)
    transition_counts = {}
    before_counts = [0] * class_count
    after_counts = [0] * class_count
    total_pixels = 0
    forest_loss_pixels = 0

    for encoded, count in histogram.items():
        code = int(float(encoded))
        before_class, after_class = divmod(code, class_count)
        if not (0 <= before_class < class_count and 0 <= after_class < class_count):
            raise ValueError(f"Dynamic World returned an invalid transition code: {encoded}")
        pixels = int(count)
        transition_counts[(before_class, after_class)] = pixels
        before_counts[before_class] += pixels
        after_counts[after_class] += pixels
        total_pixels += pixels
        if before_class == 1 and after_class != 1:
            forest_loss_pixels += pixels

    transitions = []
    changed_pixels = 0
    for (before_class, after_class), pixels in transition_counts.items():
        if before_class == after_class or pixels <= 0:
            continue
        changed_pixels += pixels
        transitions.append({
            "from_class": CLASS_NAMES[before_class],
            "to_class": CLASS_NAMES[after_class],
            "area_km2": round(pixels * PIXEL_AREA_KM2, 3),
            "percentage": round(100 * pixels / total_pixels, 2) if total_pixels else 0,
            "pixels": pixels,
        })
    transitions.sort(key=lambda item: item["pixels"], reverse=True)

    def class_areas(counts):
        return [
            {
                "name": name,
                "color": color,
                "area_km2": round(count * PIXEL_AREA_KM2, 3),
                "percentage": round(100 * count / total_pixels, 1) if total_pixels else 0,
                "pixels": count,
            }
            for name, color, count in zip(CLASS_NAMES, CLASS_PALETTE, counts)
            if count
        ]

    forest_loss_summary = {
        "detected": forest_loss_pixels > 0,
        "pixels": forest_loss_pixels,
        "area_km2": round(forest_loss_pixels * PIXEL_AREA_KM2, 3),
        "percentage": round(100 * forest_loss_pixels / total_pixels, 2) if total_pixels else 0,
        "from_class": "Forest",
        "to_class": "Any non-Forest class",
        "interpretation": "Potential forest loss, not independently verified deforestation.",
    }
    return {
        "before_classes": class_areas(before_counts),
        "after_classes": class_areas(after_counts),
        "transitions": transitions,
        "changed_percentage": round(100 * changed_pixels / total_pixels, 2) if total_pixels else 0,
        "changed_area_km2": round(changed_pixels * PIXEL_AREA_KM2, 3),
        "classified_pixel_count": total_pixels,
        "forest_loss": forest_loss_summary,
        "before_dominant_class": CLASS_NAMES[int(max(range(class_count), key=before_counts.__getitem__))] if total_pixels else None,
        "after_dominant_class": CLASS_NAMES[int(max(range(class_count), key=after_counts.__getitem__))] if total_pixels else None,
    }


def _forest_loss_coordinates(sample):
    """Return valid [latitude, longitude] pairs from a sampled EE feature collection."""
    points = []
    for feature in sample.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        longitude, latitude = coordinates[:2]
        if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            continue
        points.append([round(latitude, 6), round(longitude, 6)])
    return points


def run_gee_analysis(bounds, before_year, after_year):
    """Create two annual maps, a change mask, and area summaries for an AOI."""
    ee = _earth_engine()
    coordinates = [
        [bounds["west"], bounds["south"]],
        [bounds["east"], bounds["south"]],
        [bounds["east"], bounds["north"]],
        [bounds["west"], bounds["north"]],
        [bounds["west"], bounds["south"]],
    ]
    geometry = ee.Geometry.Polygon([coordinates], proj="EPSG:4326", geodesic=False)
    before_labels, before_rgb, before_count = _annual_images(ee, geometry, before_year)
    after_labels, after_rgb, after_count = _annual_images(ee, geometry, after_year)

    transition = before_labels.multiply(len(CLASS_NAMES)).add(after_labels).rename("transition")
    histogram_result = transition.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=10,
        maxPixels=10_000_000_000,
        tileScale=4,
    ).get("transition").getInfo()
    histogram = histogram_result or {}

    summary = _summarize_transitions(histogram)
    change_mask = before_labels.neq(after_labels).selfMask()
    forest_loss_mask = before_labels.eq(1).And(after_labels.neq(1)).selfMask()
    forest_loss_samples = (
        forest_loss_mask.rename("forest_loss")
        .uint8()
        .stratifiedSample(
            numPoints=MAX_FOREST_LOSS_POINTS,
            classBand="forest_loss",
            region=geometry,
            scale=30,
            seed=42,
            dropNulls=True,
            tileScale=4,
            geometries=True,
        )
        .getInfo()
    )

    return {
        "dynamic_world_before_imagery_layer": _tile_url(
            before_rgb, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
        ),
        "dynamic_world_after_imagery_layer": _tile_url(
            after_rgb, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
        ),
        "dynamic_world_before_land_cover_layer": _tile_url(
            before_labels, {"min": 0, "max": 8, "palette": list(CLASS_PALETTE)}
        ),
        "dynamic_world_after_land_cover_layer": _tile_url(
            after_labels, {"min": 0, "max": 8, "palette": list(CLASS_PALETTE)}
        ),
        "dynamic_world_change_layer": _tile_url(change_mask, {"palette": ["#E53935"]}),
        "dynamic_world_forest_loss_layer": _tile_url(
            forest_loss_mask, {"palette": ["#F44336"]}
        ),
        "forest_loss_points": _forest_loss_coordinates(forest_loss_samples),
        **summary,
        "before_scene_count": before_count,
        "after_scene_count": after_count,
        "current_year": date.today().year,
        "classes": list(CLASS_NAMES),
        "area_note": "Areas are estimated from the nominal 10 m Dynamic World pixel grid.",
    }
