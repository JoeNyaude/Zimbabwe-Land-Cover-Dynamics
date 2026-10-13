from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
import re

import numpy as np
from PIL import Image

IMAGE_SIZE = (256, 256)
CNN_IMAGE_SIZE = (224, 224)
CHANGE_PATCH_STRIDE = 192
CHANGE_INFERENCE_BATCH_SIZE = 8
CHANGE_THRESHOLD = 0.65
FOREST_LOSS_TILE_CONFIDENCE = 0.60
CLASS_NAMES = ["Deforestation", "Forest", "Urban", "Water"]
CNN_MODEL_FILENAME = "best_land_cover_cnn.h5"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT
MEDIA_ROOT = PROJECT_ROOT / "core" / "media"

WAYBACK_CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_TILE_URL = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{release}/{z}/{y}/{x}"
TILE_SIZE = 256
MAX_TILES_PER_SIDE = 7    # cap at 7×7 = 49 tiles to keep fetching fast
TILE_TIMEOUT = 15          # seconds per tile
TILE_DOWNLOAD_WORKERS = 8
TILE_REQUEST_ATTEMPTS = 3
TILE_RETRY_BACKOFF = 0.4


def _conv_block(inputs, filters):
    import tensorflow as tf
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def _build_change_unet():
    import tensorflow as tf
    image_a = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_a")
    image_b = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image_b")
    inputs = tf.keras.layers.Concatenate(name="temporal_pair")([image_a, image_b])

    skip1 = _conv_block(inputs, 32)
    skip2 = _conv_block(tf.keras.layers.MaxPooling2D()(skip1), 64)
    skip3 = _conv_block(tf.keras.layers.MaxPooling2D()(skip2), 128)
    bottleneck = _conv_block(tf.keras.layers.MaxPooling2D()(skip3), 256)

    up3 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(bottleneck)
    up3 = _conv_block(tf.keras.layers.Concatenate()([up3, skip3]), 128)
    up2 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(up3)
    up2 = _conv_block(tf.keras.layers.Concatenate()([up2, skip2]), 64)
    up1 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(up2)
    up1 = _conv_block(tf.keras.layers.Concatenate()([up1, skip1]), 32)
    output = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="change_mask")(up1)
    return tf.keras.Model(inputs=[image_a, image_b], outputs=output)


def _load_cnn_model_compat(tf, model_path):
    """Load the land-cover CNN with its legacy arithmetic ops."""
    class LegacyTFOpLambda(tf.keras.layers.Layer):
        def __init__(self, function, **kwargs):
            self.legacy_function = function
            super().__init__(**kwargs)

        def call(self, inputs, y=None, **kwargs):
            operation = self.legacy_function
            if operation in ("math.truediv", "truediv"):
                return inputs / y
            if operation in ("math.subtract", "subtract"):
                return inputs - y
            raise ValueError(f"Unsupported legacy CNN operation: {operation}")

    try:
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as original_error:
        try:
            # Keras 2.13 serialized Functional under keras.src.engine.functional;
            # Keras 3 moved the class to keras.src.models.functional.
            from keras.src.models.functional import Functional

            return tf.keras.models.load_model(
                str(model_path), compile=False,
                custom_objects={
                    "TFOpLambda": LegacyTFOpLambda,
                    "Functional": Functional,
                },
            )
        except Exception as compatibility_error:
            tensorflow_version = getattr(tf, "__version__", "unknown")
            keras_version = getattr(tf.keras, "__version__", "unknown")
            raise RuntimeError(
                f"Could not load {model_path.name} with TensorFlow {tensorflow_version} "
                f"and Keras {keras_version}. Install the pinned dependencies from "
                f"requirements-ml.txt. Initial load: {original_error}; "
                f"compatibility load: {compatibility_error}"
            ) from compatibility_error


def _load_change_weights_compat(model, weights_path):
    """Load the notebook checkpoint layout used by the saved UNet weights."""
    import h5py
    with h5py.File(weights_path, "r") as weights_file:
        loaded_layers = 0
        for layer in model.layers:
            if not layer.weights:
                continue
            group_name = f"_layer_checkpoint_dependencies/{layer.name}"
            match = re.match(r"(conv2d|batch_normalization)(?:_(\d+))?$", layer.name)
            if match:
                prefix, suffix = match.groups()
                checkpoint_index = int(suffix or 0) * 2
                checkpoint_name = f"{prefix}_{checkpoint_index}" if checkpoint_index else prefix
                group_name = f"_layer_checkpoint_dependencies/{checkpoint_name}"
            elif layer.name == "change_mask":
                group_name = "_layer_checkpoint_dependencies/conv2d_28"

            fallback_candidates = {
                group_name,
                group_name.replace("/", "\\"),
                group_name.replace("\\", "/"),
            }
            layout_group = next((candidate for candidate in fallback_candidates if candidate in weights_file), None)
            if layout_group is None:
                raise ValueError(f"Missing checkpoint layer: {layer.name}")

            variables = weights_file[layout_group]["vars"]
            values = [variables[str(index)][()] for index in range(len(layer.weights))]
            layer.set_weights(values)
            loaded_layers += 1
    if loaded_layers == 0:
        raise ValueError("No UNet weights were loaded")


@lru_cache(maxsize=1)
def _load_models():
    import tensorflow as tf
    change_weights = MODEL_ROOT / "change_unet_pos10.weights.h5"
    cnn_path = MODEL_ROOT / CNN_MODEL_FILENAME
    if not change_weights.exists():
        raise FileNotFoundError(f"Missing change model weights: {change_weights}")
    if not cnn_path.exists():
        raise FileNotFoundError(f"Missing land-cover model: {cnn_path}")

    change_model = _build_change_unet()
    _load_change_weights_compat(change_model, str(change_weights))
    cnn_model = _load_cnn_model_compat(tf, cnn_path)
    return change_model, cnn_model


def _load_change_image(path):
    import tensorflow as tf
    image = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    return tf.cast(image, tf.float32) / 255.0


def _preprocess_cnn_image(image):
    """Resize, pad, and rescale RGB pixels to the model's 0..1 training range."""
    import tensorflow as tf

    image = tf.cast(image, tf.float32) / 255.0
    target_height, target_width = CNN_IMAGE_SIZE
    shape = tf.shape(image)
    source_height = tf.cast(shape[0], tf.float32)
    source_width = tf.cast(shape[1], tf.float32)
    scale = tf.minimum(target_height / source_height, target_width / source_width)
    resized_height = tf.maximum(1, tf.cast(tf.round(source_height * scale), tf.int32))
    resized_width = tf.maximum(1, tf.cast(tf.round(source_width * scale), tf.int32))
    image = tf.image.resize(image, [resized_height, resized_width])
    offset_height = (target_height - resized_height) // 2
    offset_width = (target_width - resized_width) // 2
    image = tf.image.pad_to_bounding_box(
        image, offset_height, offset_width, target_height, target_width
    )
    return image.numpy()


def _patch_starts(length, patch_size, stride):
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _predict_change_probability(change_model, before_path, after_path):
    """Predict change on overlapping model-sized windows at the source resolution."""
    with Image.open(before_path) as image:
        before = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(after_path) as image:
        after = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if before.shape != after.shape:
        raise ValueError("Before and after snapshots must have identical dimensions.")

    height, width = before.shape[:2]
    y_starts = _patch_starts(height, IMAGE_SIZE[0], CHANGE_PATCH_STRIDE)
    x_starts = _patch_starts(width, IMAGE_SIZE[1], CHANGE_PATCH_STRIDE)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    probability_count = np.zeros((height, width), dtype=np.float32)
    before_batch, after_batch, patch_positions = [], [], []

    def predict_batch():
        if not before_batch:
            return
        before_inputs = np.stack(before_batch)
        after_inputs = np.stack(after_batch)
        paired_inputs = (
            np.concatenate([before_inputs, after_inputs]),
            np.concatenate([after_inputs, before_inputs]),
        )
        predictions = change_model.predict(paired_inputs, verbose=0)[..., 0]
        forward_predictions, reverse_predictions = np.split(predictions, 2, axis=0)
        predictions = (forward_predictions + reverse_predictions) / 2.0
        for prediction, (top, left, patch_height, patch_width) in zip(predictions, patch_positions):
            probability_sum[top:top + patch_height, left:left + patch_width] += prediction[:patch_height, :patch_width]
            probability_count[top:top + patch_height, left:left + patch_width] += 1
        before_batch.clear()
        after_batch.clear()
        patch_positions.clear()

    for top in y_starts:
        for left in x_starts:
            patch_height = min(IMAGE_SIZE[0], height - top)
            patch_width = min(IMAGE_SIZE[1], width - left)
            before_patch = before[top:top + patch_height, left:left + patch_width]
            after_patch = after[top:top + patch_height, left:left + patch_width]
            pad_height = IMAGE_SIZE[0] - patch_height
            pad_width = IMAGE_SIZE[1] - patch_width
            padding = ((0, pad_height), (0, pad_width), (0, 0))
            before_batch.append(np.pad(before_patch, padding, mode="edge"))
            after_batch.append(np.pad(after_patch, padding, mode="edge"))
            patch_positions.append((top, left, patch_height, patch_width))
            if len(before_batch) >= CHANGE_INFERENCE_BATCH_SIZE:
                predict_batch()
    predict_batch()
    return probability_sum / np.maximum(probability_count, 1)


def _load_cnn_inputs(path):
    """Prepare one complete, aspect-preserved snapshot at the CNN training scale."""
    import tensorflow as tf

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        image_input = _preprocess_cnn_image(
            tf.convert_to_tensor(np.asarray(image))
        )
    return image_input[None, ...], np.asarray([width * height], dtype=np.float64)


def _load_cnn_spatial_inputs(path):
    """Split a snapshot into model-sized chips and retain each chip's center."""
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)

    height, width = image.shape[:2]
    inputs = []
    centers = []
    for top in range(0, height, CNN_IMAGE_SIZE[0]):
        for left in range(0, width, CNN_IMAGE_SIZE[1]):
            patch_height = min(CNN_IMAGE_SIZE[0], height - top)
            patch_width = min(CNN_IMAGE_SIZE[1], width - left)
            patch = image[top:top + patch_height, left:left + patch_width]
            padding = (
                (0, CNN_IMAGE_SIZE[0] - patch_height),
                (0, CNN_IMAGE_SIZE[1] - patch_width),
                (0, 0),
            )
            inputs.append(np.pad(patch, padding, mode="edge").astype(np.float32) / 255.0)
            centers.append((left + patch_width / 2, top + patch_height / 2))
    return np.stack(inputs), np.asarray(centers, dtype=np.float64), (width, height)


def _pixel_center_to_lat_lon(x, y, image_size, bounds):
    """Convert a source-image pixel center from its Web Mercator crop to WGS84."""
    width, height = image_size
    north_mercator = math.log(math.tan(math.pi / 4 + math.radians(bounds["north"]) / 2))
    south_mercator = math.log(math.tan(math.pi / 4 + math.radians(bounds["south"]) / 2))
    mercator_y = north_mercator + (y / height) * (south_mercator - north_mercator)
    latitude = math.degrees(2 * math.atan(math.exp(mercator_y)) - math.pi / 2)
    longitude = bounds["west"] + (x / width) * (bounds["east"] - bounds["west"])
    return [round(latitude, 6), round(longitude, 6)]


def _predict_local_forest_loss_points(cnn_model, before_path, after_path, bounds):
    """Locate paired CNN chips predicted Forest before and Deforestation after."""
    before_inputs, centers, before_size = _load_cnn_spatial_inputs(before_path)
    after_inputs, after_centers, after_size = _load_cnn_spatial_inputs(after_path)
    if before_size != after_size or not np.array_equal(centers, after_centers):
        raise ValueError("Before and after snapshots must produce matching spatial CNN chips.")

    predictions = _normalize_cnn_predictions(cnn_model.predict(
        np.concatenate([before_inputs, after_inputs], axis=0), verbose=0
    ))
    before_scores, after_scores = np.split(predictions, 2, axis=0)
    forest_index = CLASS_NAMES.index("Forest")
    deforestation_index = CLASS_NAMES.index("Deforestation")
    points = []
    for before, after, (x, y) in zip(before_scores, after_scores, centers):
        if (
            before[forest_index] >= FOREST_LOSS_TILE_CONFIDENCE
            and after[deforestation_index] >= FOREST_LOSS_TILE_CONFIDENCE
        ):
            points.append(_pixel_center_to_lat_lon(x, y, after_size, bounds))
    return points


def _normalize_cnn_predictions(predictions):
    """Validate the CNN's four class scores in CLASS_NAMES order."""
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.ndim != 2:
        raise ValueError(f"Expected batched CNN scores, received shape {predictions.shape}.")
    if predictions.shape[1] == len(CLASS_NAMES):
        return predictions
    raise ValueError(
        f"{CNN_MODEL_FILENAME} must return {len(CLASS_NAMES)} class scores "
        f"in {CLASS_NAMES} order; received {predictions.shape[1]} outputs."
    )


def _aggregate_class_predictions(patch_probabilities, patch_areas):
    """Return area-weighted softmax scores and rounded tile class percentages."""
    patch_probabilities = np.asarray(patch_probabilities, dtype=np.float32)
    patch_areas = np.asarray(patch_areas, dtype=np.float64)
    if patch_probabilities.size == 0 or patch_areas.size == 0:
        raise ValueError("No CNN tiles were produced for the supplied image.")
    if np.any(patch_areas <= 0):
        raise ValueError("CNN tile areas must be positive to aggregate land-cover scores.")

    average_probabilities = np.average(patch_probabilities, axis=0, weights=patch_areas)
    patch_classes = np.argmax(patch_probabilities, axis=1)
    class_areas = np.bincount(
        patch_classes, weights=patch_areas, minlength=len(CLASS_NAMES)
    )
    total_area = float(class_areas.sum())
    if total_area <= 0:
        raise ValueError("The total area of the CNN tiles is zero.")
    exact_tenths = class_areas / total_area * 1000
    rounded_tenths = np.floor(exact_tenths).astype(np.int64)
    remainder = 1000 - int(rounded_tenths.sum())
    for class_index in np.argsort(exact_tenths - rounded_tenths)[::-1][:remainder]:
        rounded_tenths[class_index] += 1
    composition = {
        name: float(share) / 10
        for name, share in zip(CLASS_NAMES, rounded_tenths)
    }
    return average_probabilities, composition


def _summarize_class_transitions(before_probabilities, after_probabilities, patch_areas):
    """Estimate class transitions by pairing corresponding local CNN tiles."""
    if before_probabilities.shape != after_probabilities.shape:
        raise ValueError("Before and after snapshots must produce matching CNN tiles.")
    if len(patch_areas) != len(before_probabilities):
        raise ValueError("Every paired CNN tile must have a source-image area.")

    class_count = len(CLASS_NAMES)
    before_labels = np.argmax(before_probabilities, axis=1)
    after_labels = np.argmax(after_probabilities, axis=1)
    transition_codes = before_labels * class_count + after_labels
    transition_areas = np.bincount(
        transition_codes,
        weights=patch_areas,
        minlength=class_count * class_count,
    )
    total_area = float(np.sum(patch_areas))
    transitions = []
    changed_area = 0.0
    for code, area in enumerate(transition_areas):
        before_class, after_class = divmod(code, class_count)
        if before_class == after_class or area <= 0:
            continue
        changed_area += float(area)
        transitions.append({
            "from_class": CLASS_NAMES[before_class],
            "to_class": CLASS_NAMES[after_class],
            "percentage": round(float(area) / total_area * 100, 2),
            "tile_count": int(np.count_nonzero(transition_codes == code)),
        })
    transitions.sort(key=lambda item: item["percentage"], reverse=True)
    return {
        "changed_percentage": round(changed_area / total_area * 100, 2),
        "tile_count": len(patch_areas),
        "transitions": transitions,
        "method": "area_weighted_paired_cnn_tiles",
    }


def _save_rgb(array, path):
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def _save_gray(array, path, colormap=False):
    values = np.clip(array * 255 if array.max() <= 1 else array, 0, 255).astype(np.uint8)
    if colormap:
        rgba = Image.fromarray(values, mode="L").convert("RGBA")
        rgba.save(path)
    else:
        Image.fromarray(values, mode="L").save(path)


def _deg_to_tile(lat, lon, zoom):
    """Convert a latitude/longitude to Web Mercator tile coordinates at the given zoom."""
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y


def _deg_to_global_pixel(lat, lon, zoom):
    """Convert WGS84 coordinates to global Web Mercator pixel coordinates."""
    latitude = max(min(lat, 85.05112878), -85.05112878)
    scale = TILE_SIZE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(latitude))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def _choose_zoom(bounds):
    """Pick the highest zoom where the bounding box fits within MAX_TILES_PER_SIDE tiles."""
    for zoom in range(19, 7, -1):
        x0, y1 = _deg_to_tile(bounds["south"], bounds["west"], zoom)
        x1, y0 = _deg_to_tile(bounds["north"], bounds["east"], zoom)
        cols = abs(x1 - x0) + 1
        rows = abs(y1 - y0) + 1
        if cols <= MAX_TILES_PER_SIDE and rows <= MAX_TILES_PER_SIDE:
            return zoom
    return 8


@lru_cache(maxsize=1)
def _load_wayback_releases():
    """Load Esri's public catalog of dated World Imagery releases."""
    req = urllib.request.Request(
        WAYBACK_CONFIG_URL,
        headers={"User-Agent": "Mozilla/5.0 (ZIMLCD/1.0)"},
    )
    config = json.loads(_read_url_with_retries(req, TILE_TIMEOUT).decode("utf-8"))

    releases = []
    for release_number, item in config.items():
        if not str(release_number).isdigit() or not isinstance(item, dict):
            continue
        match = re.search(r"Wayback\s+(\d{4}-\d{2}-\d{2})", item.get("itemTitle", ""))
        if match:
            releases.append((match.group(1), int(release_number)))
    if not releases:
        raise ValueError("Esri Wayback returned no dated imagery releases.")
    return tuple(sorted(releases))


def _wayback_release_for_year(year):
    candidates = [release for release in _load_wayback_releases() if release[0].startswith(f"{year}-")]
    if not candidates:
        raise ValueError(f"Esri Wayback has no imagery release published in {year}.")
    return candidates[-1]


def _read_url_with_retries(request, timeout):
    for attempt in range(TILE_REQUEST_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            retryable = error.code in (408, 429) or error.code >= 500
            if not retryable:
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        if attempt + 1 == TILE_REQUEST_ATTEMPTS:
            raise RuntimeError(
                f"Imagery request failed after {TILE_REQUEST_ATTEMPTS} attempts: {last_error}"
            ) from last_error
        time.sleep(TILE_RETRY_BACKOFF * (2 ** attempt))


def _fetch_map_mosaic_at_zoom(bounds, release_number, zoom):
    """Download and crop one complete Wayback snapshot at a specific zoom."""
    # Tile grid covering the bounding box
    x_min, y_max = _deg_to_tile(bounds["south"], bounds["west"], zoom)
    x_max, y_min = _deg_to_tile(bounds["north"], bounds["east"], zoom)

    # Safety clamp (should already be within MAX_TILES_PER_SIDE due to _choose_zoom)
    x_max = min(x_max, x_min + MAX_TILES_PER_SIDE - 1)
    y_max = min(y_max, y_min + MAX_TILES_PER_SIDE - 1)

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1

    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    headers = {"User-Agent": "Mozilla/5.0 (ZIMLCD/1.0)"}

    tile_positions = [
        (tx, ty)
        for ty in range(y_min, y_max + 1)
        for tx in range(x_min, x_max + 1)
    ]

    def download_tile(position):
        tx, ty = position
        url = WAYBACK_TILE_URL.format(
            release=release_number, z=zoom, y=ty, x=tx
        )
        req = urllib.request.Request(url, headers=headers)
        tile_bytes = _read_url_with_retries(req, TILE_TIMEOUT)
        with Image.open(io.BytesIO(tile_bytes)) as tile_image:
            tile_image.load()
            tile = tile_image.convert("RGB")
        return position, tile

    # Tile requests are independent; bounded concurrency avoids waiting for up
    # to 49 network round trips in series while keeping load on Esri modest.
    with ThreadPoolExecutor(max_workers=min(TILE_DOWNLOAD_WORKERS, len(tile_positions))) as pool:
        for (tx, ty), tile in pool.map(download_tile, tile_positions):
            col = tx - x_min
            row = ty - y_min
            mosaic.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))

    west_px, north_px = _deg_to_global_pixel(bounds["north"], bounds["west"], zoom)
    east_px, south_px = _deg_to_global_pixel(bounds["south"], bounds["east"], zoom)
    tile_origin_x = x_min * TILE_SIZE
    tile_origin_y = y_min * TILE_SIZE
    crop_box = (
        max(0, math.floor(west_px - tile_origin_x)),
        max(0, math.floor(north_px - tile_origin_y)),
        min(mosaic.width, math.ceil(east_px - tile_origin_x)),
        min(mosaic.height, math.ceil(south_px - tile_origin_y)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("The selected bounds produced an empty imagery crop.")
    return mosaic.crop(crop_box)


def _fetch_wayback_snapshot_pair(bounds, before_year, after_year):
    """Fetch both years at one common zoom, falling back if a tile is absent."""
    before_release, before_release_number = _wayback_release_for_year(before_year)
    after_release, after_release_number = _wayback_release_for_year(after_year)
    preferred_zoom = _choose_zoom(bounds)
    missing_tile = None

    for zoom in range(preferred_zoom, 7, -1):
        try:
            before_snapshot = _fetch_map_mosaic_at_zoom(
                bounds, before_release_number, zoom
            )
            after_snapshot = _fetch_map_mosaic_at_zoom(
                bounds, after_release_number, zoom
            )
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            missing_tile = error
            continue
        return (
            (before_snapshot, before_release),
            (after_snapshot, after_release),
            zoom,
        )

    raise RuntimeError(
        f"Wayback imagery is missing tiles for {before_year} or {after_year} "
        f"at every zoom from {preferred_zoom} to 8. Last missing tile: {missing_tile.url}"
    ) from missing_tile


def run_inference_from_bounds(bounds, analysis_id, before_year, after_year):
    """
    Compare dated Esri Wayback snapshots using the local CNN and change model.
    """

    (before_snapshot, before_release), (after_snapshot, after_release), imagery_zoom = (
        _fetch_wayback_snapshot_pair(bounds, before_year, after_year)
    )
    output_dir = MEDIA_ROOT / "analyses" / str(analysis_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = output_dir / "before-source.png"
    after_path = output_dir / "after-source.png"
    before_snapshot.save(str(before_path))
    after_snapshot.save(str(after_path))

    outputs = run_inference(before_path, after_path, analysis_id, bounds)
    outputs["before_release_date"] = before_release
    outputs["after_release_date"] = after_release
    outputs["imagery_zoom"] = imagery_zoom
    outputs["change_detection_source"] = "Local change U-Net applied to Esri Wayback imagery"
    return outputs


def run_inference(before_path, after_path, analysis_id, bounds=None):
    change_model, cnn_model = _load_models()
    probability = _predict_change_probability(change_model, before_path, after_path)
    mask = probability >= CHANGE_THRESHOLD

    before_cnn_inputs, before_patch_areas = _load_cnn_inputs(before_path)
    after_cnn_inputs, after_patch_areas = _load_cnn_inputs(after_path)
    before_patch_count = len(before_cnn_inputs)
    cnn_inputs = np.concatenate([before_cnn_inputs, after_cnn_inputs], axis=0)
    class_probabilities = _normalize_cnn_predictions(
        cnn_model.predict(cnn_inputs, verbose=0)
    )
    before_class_probabilities_array, before_land_cover_composition = _aggregate_class_predictions(
        class_probabilities[:before_patch_count], before_patch_areas
    )
    if not np.array_equal(before_patch_areas, after_patch_areas):
        raise ValueError("Before and after snapshots must produce matching CNN tiles.")
    after_class_probabilities_array, after_land_cover_composition = _aggregate_class_predictions(
        class_probabilities[before_patch_count:], after_patch_areas
    )
    local_forest_loss_points = (
        _predict_local_forest_loss_points(cnn_model, before_path, after_path, bounds)
        if bounds is not None else []
    )
    local_class_transitions = _summarize_class_transitions(
        class_probabilities[:before_patch_count],
        class_probabilities[before_patch_count:],
        before_patch_areas,
    )
    local_class_transitions["method"] = "paired_snapshot_cnn_classes"
    local_class_transitions["snapshot_count"] = local_class_transitions.pop("tile_count")
    for transition in local_class_transitions["transitions"]:
        transition["snapshot_count"] = transition.pop("tile_count")
    before_class = max(before_land_cover_composition, key=before_land_cover_composition.get)
    after_class = max(after_land_cover_composition, key=after_land_cover_composition.get)
    before_class_probabilities = {
        name: round(float(score) * 100, 2)
        for name, score in zip(CLASS_NAMES, before_class_probabilities_array)
    }
    after_class_probabilities = {
        name: round(float(score) * 100, 2)
        for name, score in zip(CLASS_NAMES, after_class_probabilities_array)
    }
    after_sorted_scores = np.sort(after_class_probabilities_array)

    output_dir = MEDIA_ROOT / "analyses" / str(analysis_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(before_path) as image:
        before_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(after_path) as image:
        after_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    difference = np.abs(after_rgb.astype(np.int16) - before_rgb.astype(np.int16)).astype(np.float32) / 255.0
    difference_rgb = np.clip(difference / max(float(difference.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
    mask_rgb = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    overlay = after_rgb.copy()
    overlay[mask] = np.array([190, 45, 35], dtype=np.uint8)
    overlay = (0.55 * after_rgb + 0.45 * overlay).astype(np.uint8)

    names = {
        "before_image": "before.png",
        "after_image": "after.png",
        "absolute_difference": "absolute-difference.png",
        "change_probability": "change-probability.png",
        "change_mask": "change-mask.png",
        "overlay": "overlay.png",
    }
    _save_rgb(before_rgb, output_dir / names["before_image"])
    _save_rgb(after_rgb, output_dir / names["after_image"])
    _save_rgb(difference_rgb, output_dir / names["absolute_difference"])
    _save_rgb(np.repeat((probability * 255).astype(np.uint8)[..., None], 3, axis=2), output_dir / names["change_probability"])
    _save_rgb(mask_rgb, output_dir / names["change_mask"])
    _save_rgb(overlay, output_dir / names["overlay"])
    np.save(output_dir / "change-probability.npy", probability.astype(np.float32))

    return {
        **{key: f"/media/analyses/{analysis_id}/{filename}" for key, filename in names.items()},
        "before_predicted_class": before_class,
        "after_predicted_class": after_class,
        "before_class_probabilities": before_class_probabilities,
        "after_class_probabilities": after_class_probabilities,
        "before_land_cover_composition": before_land_cover_composition,
        "after_land_cover_composition": after_land_cover_composition,
        "local_class_transitions": local_class_transitions,
        "local_forest_loss_points": local_forest_loss_points,
        "classification_method": "whole_snapshot_aspect_preserved",
        "before_class_confidence": before_class_probabilities[before_class],
        "after_class_confidence": after_class_probabilities[after_class],
        "after_class_margin": round(float(after_sorted_scores[-1] - after_sorted_scores[-2]) * 100, 2),
        "predicted_class": after_class,
        "changed_percentage": round(float(mask.mean() * 100), 2),
        "change_threshold": CHANGE_THRESHOLD,
        "maximum_change_probability": round(float(probability.max() * 100), 2),
    }


def apply_change_threshold(analysis_id, threshold):
    """Rebuild the saved mask and overlay from stored change probabilities."""
    output_dir = MEDIA_ROOT / "analyses" / str(analysis_id)
    probability_path = output_dir / "change-probability.npy"
    if probability_path.exists():
        probability = np.load(probability_path)
    else:
        # Support analyses created before the raw probability array was saved.
        with Image.open(output_dir / "change-probability.png") as image:
            probability = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    mask = probability >= float(threshold)
    mask_rgb = np.repeat(mask.astype(np.uint8)[..., None] * 255, 3, axis=2)
    with Image.open(output_dir / "after.png") as image:
        after_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = after_rgb.copy()
    overlay[mask] = np.array([190, 45, 35], dtype=np.uint8)
    overlay = (0.55 * after_rgb + 0.45 * overlay).astype(np.uint8)
    _save_rgb(mask_rgb, output_dir / "change-mask.png")
    _save_rgb(overlay, output_dir / "overlay.png")
    return round(float(mask.mean() * 100), 2), round(float(probability.max() * 100), 2)
