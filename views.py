import json
import uuid
import csv
import math
import shutil
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from PIL import Image
from django.conf import settings
from django.core.paginator import Paginator
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .ml_inference import MEDIA_ROOT, CLASS_NAMES, apply_change_threshold, run_inference_from_bounds
from .agric_inference import predict_image as predict_agric_image
from .models import LandCoverAnalysis

ZIMBABWE_BOUNDS = {
    'south': -22.5,
    'west': 25.2,
    'north': -15.5,
    'east': 33.1,
}

ZIMBABWE_PLACES = [
    ('Harare', -17.8252, 31.0335),
    ('Chitungwiza', -18.0127, 31.0756),
    ('Bindura', -17.3019, 31.3306),
    ('Marondera', -18.1853, 31.5519),
    ('Mutare', -18.9707, 32.6709),
    ('Masvingo', -20.0744, 30.8326),
    ('Bulawayo', -20.1325, 28.6265),
    ('Gweru', -19.4552, 29.8149),
    ('Kwekwe', -18.9281, 29.8149),
    ('Kadoma', -18.3333, 29.9167),
    ('Zvishavane', -20.3267, 30.0665),
    ('Victoria Falls', -17.9243, 25.8567),
    ('Hwange', -18.3693, 26.5019),
    ('Kariba', -16.5167, 28.8),
]

# Selected reference sites for the national overview map. These are point
# markers, not surveyed waterbody or forest boundaries.
WATER_BODY_POINTS = [
    ('Lake Kariba', -16.52, 28.76),
    ('Lake Chivero', -17.9003, 30.7911),
    ('Lake Manyame', -17.78, 30.68),
    ('Lake Mazvikadei', -17.2605, 30.4254),
    ('Lake Mutirikwi', -20.2489, 31.0333),
    ('Osborne Dam', -18.7664, 32.4856),
    ('Sebakwe Dam', -19.0333, 30.25),
    ('Manjirenji Dam', -20.6262, 31.6094),
    ('Manyuchi Dam', -21.054, 30.37),
    ('Bangala Dam', -20.6947, 31.2327),
]

FOREST_POINTS = [
    ('Chirinda Forest', -20.4360, 32.7020),
    ('Bunga Forest Botanical Reserve', -19.1331, 32.7525),
    ('Mafungabusi Forest', -18.4430, 28.8758),
    ('Mukuvisi Woodland', -17.8333, 31.0833),
    ('Nyanga Forest', -18.3637, 32.8217),
]

FOREST_ALERT_AREAS = [
    {
        'name': 'Mapfungautsi Forest / Gokwe South',
        'latitude': -18.1433,
        'longitude': 28.6582,
        'evidence': 'Documented historical forest loss, 2000–2020; approximate district anchor, not a loss coordinate.',
        'source_label': 'Peer-reviewed study (2022)',
        'source_url': 'https://doi.org/10.33425/2770-2928.1013',
        'coordinate_url': 'https://www.openstreetmap.org/relation/5554779',
    },
    {
        'name': 'Murehwa District',
        'latitude': -17.8044,
        'longitude': 31.8388,
        'evidence': 'Community-reported deforestation dynamics mapped for 1990 and 2021; approximate district anchor.',
        'source_label': 'Peer-reviewed study (2024)',
        'source_url': 'https://doi.org/10.1051/cagri/2024011',
        'coordinate_url': 'https://www.openstreetmap.org/relation/5581946',
    },
    {
        'name': 'Mutasa District',
        'latitude': -18.5871,
        'longitude': 32.6824,
        'evidence': 'Documented forest-conversion pressure associated with tobacco farming; approximate district anchor, not a measured loss site.',
        'source_label': 'Peer-reviewed study (2020)',
        'source_url': 'https://doi.org/10.11648/j.eeb.20200501.12',
        'coordinate_url': 'https://www.openstreetmap.org/relation/5555252',
    },
    {
        'name': 'Bindura District',
        'latitude': -17.2425,
        'longitude': 31.2996,
        'evidence': 'Satellite land-cover study found woodland decline and increasing fragmentation between 1973 and 2000; approximate district anchor.',
        'source_label': 'Peer-reviewed study (2008)',
        'source_url': 'https://doi.org/10.1002/ldr.761',
        'coordinate_url': 'https://www.openstreetmap.org/relation/5596168',
    },
    {
        'name': 'Hurungwe District',
        'latitude': -16.5226,
        'longitude': 29.5800,
        'evidence': 'Remote-sensing study mapped forest-loss patterns in 1989, 2002, 2010 and 2020; approximate district anchor.',
        'source_label': 'Remote-sensing study (2022)',
        'source_url': 'https://doi.org/10.4236/jgis.2022.146037',
        'coordinate_url': 'https://www.openstreetmap.org/relation/7857103',
    },
    {
        'name': 'Chimanimani / Gwindingwe plantations',
        'latitude': -19.8041,
        'longitude': 32.8707,
        'evidence': 'Reported plantation fire damage: 326 ha in 2017 and 262 ha in 2018; approximate Chimanimani anchor, not a fire location.',
        'source_label': 'Global Press Journal report',
        'source_url': 'https://globalpressjournal.com/africa/zimbabwe/unregulated-deforestation-may-decimating-zimbabwes-timber-industry/',
        'coordinate_url': 'https://www.openstreetmap.org/relation/4056297',
    },
    {
        'name': 'Chirinda Forest',
        'latitude': -20.4331,
        'longitude': 32.6975,
        'evidence': 'Historical study documents community forest-resource use and conservation conflict in 1980–2000; forest reference point.',
        'source_label': 'Peer-reviewed historical study (2019)',
        'source_url': 'https://doi.org/10.1080/03057070.2019.1601867',
        'coordinate_url': 'https://www.openstreetmap.org/node/14188777401',
    },
]


def _nearest_named_place(bounds):
    """Return the nearest mapped town to the centre of a selected rectangle."""
    latitude = (bounds['south'] + bounds['north']) / 2
    longitude = (bounds['west'] + bounds['east']) / 2
    latitude_rad = math.radians(latitude)
    return min(
        ZIMBABWE_PLACES,
        key=lambda place: (
            (math.radians(place[1] - latitude)) ** 2
            + (math.cos(latitude_rad) * math.radians(place[2] - longitude)) ** 2
        ),
    )[0]


def _openstreetmap_url(bounds):
    latitude = (bounds['south'] + bounds['north']) / 2
    longitude = (bounds['west'] + bounds['east']) / 2
    return f'https://www.openstreetmap.org/?mlat={latitude:.6f}&mlon={longitude:.6f}#map=13/{latitude:.6f}/{longitude:.6f}'


def _reverse_geocode_locality(bounds):
    """Resolve a rectangle centre to a locality when an ArcGIS key is configured."""
    api_key = getattr(settings, 'ARCGIS_API_KEY', '')
    if not api_key:
        return ''
    latitude = (bounds['south'] + bounds['north']) / 2
    longitude = (bounds['west'] + bounds['east']) / 2
    query = urlencode({
        'f': 'json',
        'location': f'{longitude},{latitude}',
        'outFields': 'Neighborhood,City,Subregion,PlaceName,CountryCode',
        'token': api_key,
    })
    request = Request(
        f'https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode?{query}',
        headers={'User-Agent': 'ZIMLCD land-cover analysis'},
    )
    try:
        with urlopen(request, timeout=4) as response:
            address = json.load(response).get('address', {})
    except (OSError, ValueError, KeyError):
        return ''
    if address.get('CountryCode') not in ('ZWE', 'ZW'):
        return ''
    neighborhood = (address.get('Neighborhood') or address.get('PlaceName') or '').strip()
    city = (address.get('City') or address.get('Subregion') or '').strip()
    if neighborhood and city and neighborhood.casefold() != city.casefold():
        return f'{neighborhood}, {city}'
    return neighborhood or city

def home(request):
    return render(request, 'home.html', {
        'place_points': ZIMBABWE_PLACES,
        'water_body_points': WATER_BODY_POINTS,
        'forest_points': FOREST_POINTS,
        'forest_alert_areas': FOREST_ALERT_AREAS,
        'place_geocoding_enabled': bool(getattr(settings, 'ARCGIS_API_KEY', '')),
    })


def agriculture_prediction(request):
    """Run Agric's trained U-Net on a user supplied satellite image."""
    if request.method != 'POST':
        return render(request, 'agriculture_prediction.html')
    uploaded = request.FILES.get('image')
    if not uploaded:
        return render(request, 'agriculture_prediction.html', {'error': 'Choose a satellite image to classify.'})
    if uploaded.size > 20 * 1024 * 1024:
        return render(request, 'agriculture_prediction.html', {'error': 'Image files must be 20 MB or smaller.'})
    analysis_id = uuid.uuid4().hex
    output_dir = MEDIA_ROOT / 'agric-predictions' / analysis_id
    try:
        with Image.open(uploaded) as image:
            image.verify()
        uploaded.seek(0)
        source_path = output_dir / 'source.png'
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(uploaded) as image:
            image.convert('RGB').save(source_path)
        mask_path, composition = predict_agric_image(source_path, output_dir)
    except Exception as error:
        shutil.rmtree(output_dir, ignore_errors=True)
        return render(request, 'agriculture_prediction.html', {
            'error': f'Could not create a prediction: {error}'
        })
    return render(request, 'agriculture_prediction.html', {
        'source_url': f'{settings.MEDIA_URL}agric-predictions/{analysis_id}/source.png',
        'prediction_url': f'{settings.MEDIA_URL}agric-predictions/{analysis_id}/prediction.png',
        'composition': composition,
    })


def _rectangle_area_hectares(bounds):
    """Approximate rectangle area on a spherical Earth, returned in hectares."""
    earth_radius_m = 6_371_008.8
    south = math.radians(bounds['south'])
    north = math.radians(bounds['north'])
    longitude_width = math.radians(bounds['east'] - bounds['west'])
    area_m2 = earth_radius_m ** 2 * longitude_width * (math.sin(north) - math.sin(south))
    return round(area_m2 / 10_000, 2)


def _changed_area_hectares(area_hectares, outputs):
    dynamic_world_area_km2 = outputs.get('dynamic_world_changed_area_km2')
    if dynamic_world_area_km2 is not None:
        return round(float(dynamic_world_area_km2) * 100, 2)
    return round(
        area_hectares * float(outputs.get('changed_percentage', 0)) / 100,
        2,
    )


def _ensure_session_key(request):
    if not request.session.session_key:
        request.session.create()
    request.session.set_expiry(60 * 60 * 24 * 365)
    return request.session.session_key


def dashboard(request):
    session_key = request.session.session_key
    if not session_key and request.session.get('latest_analysis'):
        session_key = _ensure_session_key(request)

    # Database rows are authoritative. Do not recreate records from session
    # snapshots: that would resurrect analyses intentionally deleted in admin.
    if session_key:
        saved_ids = set(LandCoverAnalysis.objects.filter(
            session_key=session_key,
        ).values_list('analysis_id', flat=True))
        latest_in_session = request.session.get('latest_analysis')
        if latest_in_session and latest_in_session.get('analysis_id') not in saved_ids:
            request.session.pop('latest_analysis', None)
        request.session['analysis_history'] = [
            item for item in request.session.get('analysis_history', [])
            if item.get('analysis_id') in saved_ids
        ]

    records = LandCoverAnalysis.objects.filter(session_key=session_key).order_by('-created_at') if session_key else LandCoverAnalysis.objects.none()
    latest_record = records.first()
    analysis = latest_record.as_analysis() if latest_record else None
    page_obj = Paginator(records, 5).get_page(request.GET.get('page'))
    for record in page_obj.object_list:
        record.map_url = _openstreetmap_url(record.bounds)
    recent_records = list(records[:8])
    maximum_area_hectares = max((record.area_hectares for record in recent_records), default=1) or 1
    class_chart = []
    if analysis:
        outputs = analysis.get('outputs', {})
        if outputs.get('dynamic_world_enabled'):
            from .gee_analysis import CLASS_NAMES as DYNAMIC_WORLD_CLASSES

            before_shares = {
                item['name']: item['percentage']
                for item in outputs.get('dynamic_world_before_classes', [])
            }
            after_shares = {
                item['name']: item['percentage']
                for item in outputs.get('dynamic_world_after_classes', [])
            }
            class_chart = [
                {
                    'name': class_name,
                    'before': before_shares.get(class_name, 0),
                    'after': after_shares.get(class_name, 0),
                }
                for class_name in DYNAMIC_WORLD_CLASSES
            ]
        else:
            before_scores = outputs.get('before_class_probabilities', {})
            after_scores = outputs.get('after_class_probabilities', {})
            class_chart = [
                {
                    'name': class_name,
                    'before': before_scores.get(class_name, 0),
                    'after': after_scores.get(class_name, 0),
                }
                for class_name in after_scores
            ]
    change_chart = []
    for index, record in enumerate(reversed(recent_records), start=1):
        item = record.as_analysis()
        bounds = item.get('bounds', {})
        center_lat = (bounds.get('south', 0) + bounds.get('north', 0)) / 2
        center_lon = (bounds.get('west', 0) + bounds.get('east', 0)) / 2
        change_chart.append({
            'label': record.location_name or f'Area {index}',
            'center': f'{center_lat:.5f}, {center_lon:.5f}',
            'map_url': _openstreetmap_url(bounds),
            'changed': item.get('outputs', {}).get('changed_percentage', 0),
            'area_hectares': record.area_hectares,
            'changed_hectares': record.changed_hectares,
            'area_width': round(record.area_hectares / maximum_area_hectares * 100, 2),
            'changed_width': round(record.changed_hectares / maximum_area_hectares * 100, 2),
            'before_year': item.get('before_year'),
            'after_year': item.get('after_year'),
            'class_name': item.get('outputs', {}).get(
                'dynamic_world_after_dominant_class',
                item.get('outputs', {}).get('after_predicted_class', 'Unknown'),
            ),
            'bounds': bounds,
            'analysis_id': record.analysis_id,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M'),
        })
    change_values = [float(row['changed'] or 0) for row in change_chart]
    most_changed = max(change_chart, key=lambda row: float(row['changed'] or 0), default=None)
    change_summary = {
        'screened_count': len(change_chart),
        'average_changed': round(sum(change_values) / len(change_values), 1) if change_values else 0,
        'at_or_above_10': sum(value >= 10 for value in change_values),
        'most_changed': most_changed,
    }
    return render(request, 'dashboard.html', {
        'analysis': analysis,
        'analysis_map_url': _openstreetmap_url(analysis['bounds']) if analysis and analysis.get('bounds') else '',
        'page_obj': page_obj,
        'analysis_count': records.count(),
        'maximum_area_hectares': maximum_area_hectares,
        'change_chart': change_chart,
        'change_summary': change_summary,
        'class_chart': class_chart,
    })


@require_POST
def delete_analysis(request, analysis_id):
    """Delete one saved analysis owned by this browser session."""
    session_key = request.session.session_key or ''
    record = LandCoverAnalysis.objects.filter(
        analysis_id=analysis_id,
        session_key=session_key,
    ).first()

    history = request.session.get('analysis_history', [])
    request.session['analysis_history'] = [
        item for item in history if item.get('analysis_id') != analysis_id
    ]
    latest = request.session.get('latest_analysis')
    if latest and latest.get('analysis_id') == analysis_id:
        request.session.pop('latest_analysis', None)

    if record:
        analyses_root = (MEDIA_ROOT / 'analyses').resolve()
        analysis_dir = (analyses_root / record.analysis_id).resolve()
        if analysis_dir.parent == analyses_root and analysis_dir.name == record.analysis_id:
            try:
                shutil.rmtree(analysis_dir)
            except OSError:
                pass
        record.delete()

        if latest and latest.get('analysis_id') == analysis_id:
            replacement = LandCoverAnalysis.objects.filter(
                session_key=session_key,
            ).first()
            if replacement:
                request.session['latest_analysis'] = replacement.as_analysis()

    return redirect('dashboard')


def analysis_results(request, analysis_id=None):
    if analysis_id:
        record = LandCoverAnalysis.objects.filter(
            analysis_id=analysis_id, session_key=request.session.session_key or ''
        ).first()
        if not record:
            raise Http404('Saved analysis not found.')
        analysis = record.as_analysis()
    else:
        analysis = request.session.get('latest_analysis')
    if not analysis:
        return render(request, 'results.html', {'error': 'No analysis area has been selected yet.'})
    if 'area_hectares' not in analysis and analysis.get('bounds'):
        analysis['area_hectares'] = _rectangle_area_hectares(analysis['bounds'])
        analysis['changed_hectares'] = round(
            analysis['area_hectares'] * float(analysis.get('outputs', {}).get('changed_percentage', 0)) / 100,
            2,
        )
    review_class_names = CLASS_NAMES
    if analysis.get('outputs', {}).get('dynamic_world_enabled'):
        from .gee_analysis import CLASS_NAMES as DYNAMIC_WORLD_CLASSES
        review_class_names = DYNAMIC_WORLD_CLASSES
    return render(request, 'results.html', {
        'analysis': analysis,
        'persistent_record': bool(analysis_id),
        'class_names': review_class_names,
        'location_notice': request.session.pop('location_name_notice', ''),
        'review_notice': request.session.pop('class_review_notice', ''),
        'change_threshold_notice': request.session.pop('change_threshold_notice', ''),
    })


def _analysis_for_update(request):
    requested_id = request.POST.get('analysis_id', '')
    if requested_id:
        record = LandCoverAnalysis.objects.filter(
            analysis_id=requested_id, session_key=request.session.session_key or ''
        ).first()
        if record:
            return record.as_analysis(), record
    analysis = request.session.get('latest_analysis')
    if not analysis or (requested_id and analysis.get('analysis_id') != requested_id):
        return None, None
    record = LandCoverAnalysis.objects.filter(
        analysis_id=analysis.get('analysis_id'), session_key=request.session.session_key or ''
    ).first()
    return analysis, record


def _save_analysis_update(request, analysis, record):
    if record:
        record.location_name = analysis.get('location_name', record.location_name)
        record.outputs = analysis['outputs']
        record.reviewed_classes = analysis.get('reviewed_classes', {})
        record.changed_hectares = _changed_area_hectares(
            record.area_hectares, analysis['outputs']
        )
        record.save(update_fields=['location_name', 'outputs', 'reviewed_classes', 'changed_hectares'])
        analysis = record.as_analysis()
    latest = request.session.get('latest_analysis')
    if latest and latest.get('analysis_id') == analysis.get('analysis_id'):
        request.session['latest_analysis'] = analysis
    history = request.session.get('analysis_history', [])
    for index, item in enumerate(history):
        if item.get('analysis_id') == analysis.get('analysis_id'):
            history[index] = analysis
            break
    request.session['analysis_history'] = history
    if record:
        return redirect('analysis-result-detail', analysis_id=analysis['analysis_id'])
    return redirect('analysis-results')


@require_POST
def update_analysis_location(request):
    analysis, record = _analysis_for_update(request)
    location_name = request.POST.get('location_name', '').strip()
    if not analysis:
        request.session['location_name_notice'] = 'Saved analysis not found in this browser session.'
        return redirect('dashboard')
    if len(location_name) > 120:
        request.session['location_name_notice'] = 'Location names must be 120 characters or fewer.'
        return redirect('analysis-result-detail', analysis_id=analysis['analysis_id']) if record else redirect('analysis-results')

    analysis['location_name'] = location_name
    request.session['location_name_notice'] = (
        f'Location name saved as “{location_name}”.' if location_name
        else 'Location name cleared.'
    )
    return _save_analysis_update(request, analysis, record)


@require_POST
def record_class_review(request):
    analysis, record = _analysis_for_update(request)
    snapshot = request.POST.get('snapshot', '')
    reference_class = request.POST.get('reference_class', '')
    valid_classes = set(CLASS_NAMES)
    if analysis and analysis.get('outputs', {}).get('dynamic_world_enabled'):
        from .gee_analysis import CLASS_NAMES as DYNAMIC_WORLD_CLASSES
        valid_classes.update(DYNAMIC_WORLD_CLASSES)
    if not analysis or snapshot not in ('before', 'after') or reference_class not in valid_classes:
        request.session['class_review_notice'] = 'Choose a snapshot and one of the model classes before saving.'
        return redirect('analysis-results')

    reviewed_classes = analysis.setdefault('reviewed_classes', {})
    reviewed_classes[snapshot] = reference_class
    reviewed_classes[f'{snapshot}_reviewed_at'] = timezone.localtime().strftime('%Y-%m-%d %H:%M')
    request.session['class_review_notice'] = f'Saved {reference_class} as the reviewed reference label for the {snapshot} snapshot.'
    return _save_analysis_update(request, analysis, record)


@require_POST
def update_change_threshold(request):
    analysis, record = _analysis_for_update(request)
    if not analysis:
        request.session['change_threshold_notice'] = 'Run an analysis before changing the mask threshold.'
        return redirect('analysis-results')
    try:
        threshold = float(request.POST.get('threshold', ''))
    except (TypeError, ValueError):
        threshold = None
    if threshold is None or not 0.10 <= threshold <= 0.90:
        request.session['change_threshold_notice'] = 'Choose a threshold between 0.10 and 0.90.'
        return redirect('analysis-results')

    try:
        changed_percentage, maximum_probability = apply_change_threshold(
            analysis['analysis_id'], threshold
        )
    except (OSError, ValueError) as error:
        request.session['change_threshold_notice'] = f'Could not rebuild the mask: {error}'
        return redirect('analysis-results')

    analysis['outputs']['change_threshold'] = round(threshold, 2)
    analysis['outputs']['wayback_unet_changed_percentage'] = changed_percentage
    if 'dynamic_world_changed_percentage' not in analysis['outputs']:
        analysis['outputs']['changed_percentage'] = changed_percentage
    analysis['outputs']['maximum_change_probability'] = maximum_probability
    request.session['change_threshold_notice'] = (
        f'Wayback CNN mask rebuilt at threshold {threshold:.2f}; '
        f'{changed_percentage}% of pixels are marked as changed.'
    )
    return _save_analysis_update(request, analysis, record)


def land_cover_maps(request):
    return render(request, 'maps.html', {
        'analysis': request.session.get('latest_analysis'),
    })


def _latest_saved_session_analysis(request):
    if not request.session.session_key:
        if not request.session.get('latest_analysis'):
            return None
        _ensure_session_key(request)
    record = LandCoverAnalysis.objects.filter(
        session_key=request.session.session_key,
    ).first()
    return record.as_analysis() if record else None


def reports(request):
    analysis = _latest_saved_session_analysis(request)
    chart_classes = []
    composition_classes = []
    has_land_cover_composition = False
    change_percentage = 0
    time_series = []
    time_series_by_year = {}
    time_series_segments = []
    score_bins = []
    score_bin_max = 1
    score_bin_source = 'Wayback U-Net change probabilities'
    current_year = timezone.now().year
    timeline_years = []
    if analysis:
        outputs = analysis.get('outputs', {})
        before = outputs.get('before_class_probabilities', {})
        after = outputs.get('after_class_probabilities', {})
        dynamic_world_before = outputs.get('dynamic_world_before_classes') or []
        dynamic_world_after = outputs.get('dynamic_world_after_classes') or []
        if outputs.get('classification_method') == 'whole_snapshot_aspect_preserved':
            before_composition = before
            after_composition = after
        else:
            before_composition = outputs.get('before_land_cover_composition') or {}
            after_composition = outputs.get('after_land_cover_composition') or {}
        has_land_cover_composition = bool(before_composition or after_composition)
        chart_classes = [
            {'name': name, 'before': max(0, min(100, float(before.get(name, 0) or 0))),
             'after': max(0, min(100, float(after.get(name, 0) or 0)))}
            for name in CLASS_NAMES
        ]
        if outputs.get('dynamic_world_enabled'):
            from .gee_analysis import CLASS_NAMES as DYNAMIC_WORLD_CLASSES, CLASS_PALETTE

            before_percentages = {item['name']: item['percentage'] for item in dynamic_world_before}
            after_percentages = {item['name']: item['percentage'] for item in dynamic_world_after}
            has_land_cover_composition = bool(dynamic_world_before or dynamic_world_after)
            composition_classes = [
                {
                    'name': name,
                    'before': max(0, min(100, float(before_percentages.get(name, 0) or 0))),
                    'after': max(0, min(100, float(after_percentages.get(name, 0) or 0))),
                    'color': color,
                }
                for name, color in zip(DYNAMIC_WORLD_CLASSES, CLASS_PALETTE)
            ]
        else:
            cnn_colors = {
                'Deforestation': '#c46b4b',
                'Forest': '#4f8a62',
                'Urban': '#4d79a8',
                'Water': '#4aa6a8',
            }
            composition_classes = [
                {
                    'name': name,
                    'before': max(0, min(100, float(before_composition.get(name, 0) or 0))),
                    'after': max(0, min(100, float(after_composition.get(name, 0) or 0))),
                    'color': cnn_colors[name],
                }
                for name in CLASS_NAMES
            ]
        probability_path = MEDIA_ROOT / 'analyses' / str(analysis.get('analysis_id')) / 'change-probability.npy'
        if probability_path.exists():
            import numpy as np
            probability = np.load(probability_path, mmap_mode='r')
            counts, _ = np.histogram(probability, bins=10, range=(0, 1))
            score_bins = [
                {'label': f'{start}–{start + 10}%', 'count': int(count)}
                for start, count in zip(range(0, 100, 10), counts)
            ]
            score_bin_max = max((item['count'] for item in score_bins), default=1) or 1
        else:
            # Older analyses may not have the raw probability raster; retain
            # a useful histogram of their recorded class scores as a fallback.
            scores = [value for item in chart_classes for value in (item['before'], item['after'])]
            score_bins = [{'label': f'{start}–{start + 10}%', 'count': sum(start <= score < start + 10 for score in scores)}
                          for start in range(0, 100, 10)]
            score_bins[-1]['count'] += sum(score == 100 for score in scores)
            score_bin_source = 'class scores (older analysis)'
            score_bin_max = max((item['count'] for item in score_bins), default=1) or 1
        change_percentage = max(0, min(100, float(outputs.get('changed_percentage', 0) or 0)))
        current_bounds = analysis.get('bounds')
        records = LandCoverAnalysis.objects.filter(
            session_key=request.session.session_key or '',
            bounds=current_bounds,
        ).order_by('after_year', 'created_at')
        analyses = [record.as_analysis() for record in records]
        if not any(item.get('analysis_id') == analysis.get('analysis_id') for item in analyses):
            analyses.append(analysis)
        # Use the later year of each comparison as the point on the timeline.
        primary_source = bool(outputs.get('dynamic_world_enabled'))
        for item in analyses:
            item_outputs = item.get('outputs', {})
            if bool(item_outputs.get('dynamic_world_enabled')) != primary_source:
                continue
            try:
                year = int(item.get('after_year'))
                percentage = max(0, min(100, float(item_outputs.get('changed_percentage', 0) or 0)))
            except (TypeError, ValueError):
                continue
            time_series_by_year[year] = {
                'year': year,
                'changed': percentage,
                'label': item.get('location_name') or 'Selected area',
            }
        time_series = sorted(time_series_by_year.values(), key=lambda point: point['year'])
        timeline_start = 2023
        timeline_end = max(current_year, timeline_start + 1)
        timeline_years = []
        for year in range(timeline_start, timeline_end + 1):
            timeline_years.append({
                'year': year,
                'x': round(40 + (year - timeline_start) / (timeline_end - timeline_start) * 420, 1),
            })
        if time_series:
            for point in time_series:
                point['x'] = round(40 + (point['year'] - timeline_start) / (timeline_end - timeline_start) * 420, 1)
                point['y'] = round(150 - point['changed'] * 1.25, 1)
            time_series[0]['path_start'] = True
        for earlier, later in zip(time_series, time_series[1:]):
            if later['year'] == earlier['year'] + 1:
                time_series_segments.append(
                    f"{earlier['x']},{earlier['y']} {later['x']},{later['y']}"
                )
    return render(request, 'reports.html', {
        'analysis': analysis,
        'analysis_map_url': _openstreetmap_url(analysis['bounds']) if analysis and analysis.get('bounds') else '',
        'chart_classes': chart_classes,
        'composition_classes': composition_classes,
        'has_land_cover_composition': has_land_cover_composition,
        'change_percentage': change_percentage,
        'unchanged_percentage': 100 - change_percentage,
        'time_series': time_series,
        'time_series_segments': time_series_segments,
        'score_bins': score_bins,
        'score_bin_max': score_bin_max,
        'score_bin_source': score_bin_source,
        'timeline_years': timeline_years,
        'timeline_end_year': max(current_year, 2024),
    })


def download_report(request):
    analysis = _latest_saved_session_analysis(request)
    if not analysis:
        return render(request, 'reports.html', {
            'analysis': None,
            'error': 'Run an analysis before downloading a report.',
        }, status=400)

    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="zimlcd-analysis-report.csv"'
    writer = csv.writer(response)
    writer.writerow(['ZIMLCD analysis report'])
    writer.writerow(['Location name', analysis.get('location_name', '')])
    writer.writerow(['Earlier year', analysis['before_year']])
    writer.writerow(['Earlier Wayback release', analysis['outputs'].get('before_release_date', '')])
    writer.writerow(['Supplementary unvalidated Wayback CNN estimate (earlier)', analysis['outputs'].get('before_predicted_class', '')])
    writer.writerow(['Supplementary Wayback CNN softmax score (earlier, %)', analysis['outputs'].get('before_class_confidence', '')])
    classification_method = analysis['outputs'].get('classification_method')
    before_class_values = (
        analysis['outputs'].get('before_class_probabilities') or {}
        if classification_method == 'whole_snapshot_aspect_preserved'
        else analysis['outputs'].get('before_land_cover_composition') or {}
    )
    earlier_value_label = (
        'Earlier {class_name} CNN score (%)'
        if classification_method == 'whole_snapshot_aspect_preserved'
        else 'Earlier {class_name} share (%)'
    )
    for class_name in CLASS_NAMES:
        writer.writerow([
            earlier_value_label.format(class_name=class_name),
            before_class_values.get(class_name, ''),
        ])
    writer.writerow(['Later year', analysis['after_year']])
    writer.writerow(['Later Wayback release', analysis['outputs'].get('after_release_date', '')])
    writer.writerow(['Supplementary unvalidated Wayback CNN estimate (later)', analysis['outputs'].get('after_predicted_class', '')])
    writer.writerow(['Supplementary Wayback CNN softmax score (later, %)', analysis['outputs'].get('after_class_confidence', '')])
    later_value_label = (
        'Later {class_name} CNN score (%)'
        if classification_method == 'whole_snapshot_aspect_preserved'
        else 'Later {class_name} share (%)'
    )
    after_class_values = (
        analysis['outputs'].get('after_class_probabilities') or {}
        if classification_method == 'whole_snapshot_aspect_preserved'
        else analysis['outputs'].get('after_land_cover_composition') or {}
    )
    for class_name in CLASS_NAMES:
        writer.writerow([
            later_value_label.format(class_name=class_name),
            after_class_values.get(class_name, ''),
        ])
    dynamic_world_before = analysis['outputs'].get('dynamic_world_before_classes') or []
    dynamic_world_after = analysis['outputs'].get('dynamic_world_after_classes') or []
    if dynamic_world_before or dynamic_world_after:
        writer.writerow([])
        writer.writerow(['Primary land-cover source', 'Google Dynamic World annual 10 m classifications'])
        writer.writerow(['Earlier Dynamic World dominant class', analysis['outputs'].get('dynamic_world_before_dominant_class', '')])
        writer.writerow(['Later Dynamic World dominant class', analysis['outputs'].get('dynamic_world_after_dominant_class', '')])
        writer.writerow(['Dynamic World class', f'Earlier {analysis["before_year"]} (%)', f'Later {analysis["after_year"]} (%)'])
        before_shares = {item['name']: item['percentage'] for item in dynamic_world_before}
        after_shares = {item['name']: item['percentage'] for item in dynamic_world_after}
        for class_name in dict.fromkeys([*before_shares, *after_shares]):
            writer.writerow([class_name, before_shares.get(class_name, 0), after_shares.get(class_name, 0)])
        writer.writerow(['Dynamic World class-transition pixels (%)', analysis['outputs'].get('dynamic_world_changed_percentage', '')])
        forest_loss = analysis['outputs'].get('dynamic_world_forest_loss') or {}
        writer.writerow(['Potential forest loss pixels (%)', forest_loss.get('percentage', '')])
        writer.writerow(['Potential forest loss area (km2)', forest_loss.get('area_km2', '')])
        writer.writerow(['Forest loss interpretation', forest_loss.get('interpretation', '')])
        writer.writerow(['Forest transition source', 'Dynamic World Forest/Trees class to any non-Forest class'])
    local_transitions = analysis['outputs'].get('local_class_transitions') or {}
    if local_transitions:
        writer.writerow([])
        if local_transitions.get('method') == 'paired_snapshot_cnn_classes':
            writer.writerow(['Paired whole-snapshot CNN class comparison (unvalidated on Wayback imagery)'])
            writer.writerow(['Snapshot pairs with different predicted classes (%)', local_transitions.get('changed_percentage', '')])
            writer.writerow(['Earlier snapshot class', 'Later snapshot class', 'Compared pairs'])
            count_key = 'snapshot_count'
        else:
            writer.writerow(['Supplementary local CNN tile transitions (unvalidated on Wayback imagery)'])
            writer.writerow(['Classified tile area changing class (%)', local_transitions.get('changed_percentage', '')])
            writer.writerow(['From class', 'To class', 'Image area share (%)', 'Tile count'])
            count_key = 'tile_count'
        for transition in local_transitions.get('transitions', []):
            writer.writerow([
                transition.get('from_class', ''),
                transition.get('to_class', ''),
                transition.get('snapshot_count' if count_key == 'snapshot_count' else 'percentage', ''),
                transition.get(count_key, ''),
            ])
        reviewed_classes = analysis.get('reviewed_classes', {})
    writer.writerow(['Earlier reviewed reference label', reviewed_classes.get('before', '')])
    writer.writerow(['Later reviewed reference label', reviewed_classes.get('after', '')])
    writer.writerow(['Primary Dynamic World pixels with class change (%)', analysis['outputs'].get('dynamic_world_changed_percentage', analysis['outputs'].get('changed_percentage', ''))])
    writer.writerow(['Wayback U-Net change-mask pixels (%)', analysis['outputs'].get('wayback_unet_changed_percentage', '')])
    writer.writerow(['Change probability threshold', analysis['outputs'].get('change_threshold', '')])
    writer.writerow(['Selected area (estimated hectares)', analysis.get('area_hectares', '')])
    writer.writerow(['Changed area (estimated hectares)', analysis.get('changed_hectares', '')])
    writer.writerow(['Selected area south', analysis['bounds']['south']])
    writer.writerow(['Selected area west', analysis['bounds']['west']])
    writer.writerow(['Selected area north', analysis['bounds']['north']])
    writer.writerow(['Selected area east', analysis['bounds']['east']])
    writer.writerow([])
    writer.writerow(['Note', 'Predictions are model outputs and should be checked against local reference data.'])
    return response


def help_page(request):
    return render(request, 'help.html')


def about_page(request):
    return render(request, 'about.html')


@require_POST
def create_analysis(request):
    """Run the supplied CNN and change detector on the selected image pair."""
    try:
        payload = json.loads(request.body)
        bounds = payload['bounds']
        before_year = int(payload['before_year'])
        after_year = int(payload['after_year'])
        location_name = payload.get('location_name', '')
        location_name_is_manual = payload.get(
            'location_name_is_manual', bool(location_name.strip()) if isinstance(location_name, str) else False
        ) is True
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return JsonResponse({'error': 'Provide valid years and a selected map area.'}, status=400)

    required_bounds = {'south', 'west', 'north', 'east'}
    if set(bounds) != required_bounds:
        return JsonResponse({'error': 'The selected map area is incomplete.'}, status=400)

    if not 2023 <= before_year <= 2026 or not 2023 <= after_year <= 2026:
        return JsonResponse({'error': 'Years must be between 2023 and 2026.'}, status=400)

    if before_year >= after_year:
        return JsonResponse({'error': 'The later year must be after the earlier year.'}, status=400)

    if not isinstance(location_name, str):
        return JsonResponse({'error': 'The location name must be text.'}, status=400)
    location_name = location_name.strip()
    if len(location_name) > 120:
        return JsonResponse({'error': 'Location names must be 120 characters or fewer.'}, status=400)

    try:
        bounds = {key: float(bounds[key]) for key in required_bounds}
    except (TypeError, ValueError):
        return JsonResponse({'error': 'The selected map area contains invalid coordinates.'}, status=400)

    if bounds['south'] >= bounds['north'] or bounds['west'] >= bounds['east']:
        return JsonResponse({'error': 'The selected map area is invalid.'}, status=400)

    if (
        bounds['south'] < ZIMBABWE_BOUNDS['south']
        or bounds['west'] < ZIMBABWE_BOUNDS['west']
        or bounds['north'] > ZIMBABWE_BOUNDS['north']
        or bounds['east'] > ZIMBABWE_BOUNDS['east']
    ):
        return JsonResponse({'error': 'The selected area must be within Zimbabwe.'}, status=400)

    if not location_name_is_manual:
        cached_analysis = LandCoverAnalysis.objects.filter(
            session_key=request.session.session_key or '', bounds=bounds,
        ).exclude(location_name__startswith='Near ').order_by('-created_at').first()
        location_name = cached_analysis.location_name if cached_analysis else ''
        if not location_name:
            location_name = _reverse_geocode_locality(bounds)
        if not location_name:
            location_name = f'Near {_nearest_named_place(bounds)} (approx.)'

    analysis_id = uuid.uuid4().hex
    try:
        outputs = run_inference_from_bounds(
            bounds, analysis_id, before_year, after_year
        )
    except Exception as error:
        return JsonResponse({'error': f'Land-cover analysis failed: {error}'}, status=503)

    outputs.setdefault('local_forest_loss_points', [])
    # Apply the Agric semantic-segmentation model to the same dated snapshots
    # used by the change analysis. Keep its failure independent of core analysis.
    try:
        agric_dir = MEDIA_ROOT / 'analyses' / analysis_id
        _, outputs['agric_before_composition'] = predict_agric_image(
            agric_dir / 'before-source.png', agric_dir / 'agric-before'
        )
        _, outputs['agric_after_composition'] = predict_agric_image(
            agric_dir / 'after-source.png', agric_dir / 'agric-after'
        )
        outputs['agric_before_dominant_class'] = max(
            outputs['agric_before_composition'], key=lambda item: item['percentage']
        )['name']
        outputs['agric_after_dominant_class'] = max(
            outputs['agric_after_composition'], key=lambda item: item['percentage']
        )['name']
        outputs['agric_before_prediction'] = f'/media/analyses/{analysis_id}/agric-before/prediction.png'
        outputs['agric_after_prediction'] = f'/media/analyses/{analysis_id}/agric-after/prediction.png'
        outputs['agric_prediction_model'] = 'Agric U-Net (best_unet.pt)'
    except Exception as error:
        outputs['agric_prediction_error'] = str(error)

    if settings.EARTH_ENGINE_PROJECT:
        try:
            from .gee_analysis import run_gee_analysis

            dynamic_world = run_gee_analysis(bounds, before_year, after_year)
        except Exception as error:
            outputs['forest_loss_detection_error'] = (
                f'Potential forest-loss detection could not run: {error}'
            )
        else:
            outputs['wayback_unet_changed_percentage'] = outputs['changed_percentage']
            outputs.update({
                'dynamic_world_enabled': True,
                'dynamic_world_before_imagery_layer': dynamic_world['dynamic_world_before_imagery_layer'],
                'dynamic_world_after_imagery_layer': dynamic_world['dynamic_world_after_imagery_layer'],
                'dynamic_world_before_land_cover_layer': dynamic_world['dynamic_world_before_land_cover_layer'],
                'dynamic_world_after_land_cover_layer': dynamic_world['dynamic_world_after_land_cover_layer'],
                'dynamic_world_change_layer': dynamic_world['dynamic_world_change_layer'],
                'dynamic_world_forest_loss_layer': dynamic_world['dynamic_world_forest_loss_layer'],
                'dynamic_world_before_classes': dynamic_world['before_classes'],
                'dynamic_world_after_classes': dynamic_world['after_classes'],
                'dynamic_world_transitions': dynamic_world['transitions'],
                'dynamic_world_before_dominant_class': dynamic_world['before_dominant_class'],
                'dynamic_world_after_dominant_class': dynamic_world['after_dominant_class'],
                'dynamic_world_forest_loss': dynamic_world['forest_loss'],
                'dynamic_world_forest_loss_points': dynamic_world['forest_loss_points'],
                'dynamic_world_changed_percentage': dynamic_world['changed_percentage'],
                'dynamic_world_changed_area_km2': dynamic_world['changed_area_km2'],
                'changed_percentage': dynamic_world['changed_percentage'],
                'area_note': dynamic_world['area_note'],
            })
    area_hectares = _rectangle_area_hectares(bounds)
    changed_hectares = _changed_area_hectares(area_hectares, outputs)
    analysis = {
        'status': 'analysis_complete',
        'message': 'The selected imagery pair was analyzed successfully.',
        'created_at': timezone.localtime().strftime('%Y-%m-%d %H:%M'),
        'before_year': before_year,
        'after_year': after_year,
        'bounds': bounds,
        'outputs': outputs,
        'analysis_id': analysis_id,
        'location_name': location_name,
        'area_hectares': area_hectares,
        'changed_hectares': changed_hectares,
    }
    session_key = _ensure_session_key(request)
    LandCoverAnalysis.objects.create(
        analysis_id=analysis_id,
        session_key=session_key,
        location_name=location_name,
        before_year=before_year,
        after_year=after_year,
        bounds=bounds,
        outputs=outputs,
        area_hectares=area_hectares,
        changed_hectares=changed_hectares,
    )
    history = request.session.get('analysis_history', [])
    previous_analysis = request.session.get('latest_analysis')
    if previous_analysis and previous_analysis.get('analysis_id') != analysis_id:
        history = [item for item in history if item.get('analysis_id') != previous_analysis.get('analysis_id')]
        history.insert(0, previous_analysis)
    history = [item for item in history if item.get('analysis_id') != analysis_id]
    history.insert(0, analysis)
    request.session['latest_analysis'] = analysis
    request.session['analysis_history'] = history[:8]
    return JsonResponse({**analysis, 'results_url': reverse('analysis-result-detail', args=[analysis_id])})
