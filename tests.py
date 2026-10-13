from django.test import TestCase, override_settings
from django.urls import reverse
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from unittest.mock import MagicMock, patch

from .ml_inference import (
	CLASS_NAMES,
	_aggregate_class_predictions,
	_load_cnn_inputs,
	_load_cnn_spatial_inputs,
	_pixel_center_to_lat_lon,
	_preprocess_cnn_image,
	_predict_local_forest_loss_points,
	run_inference_from_bounds,
	_summarize_class_transitions,
	_read_url_with_retries,
	_fetch_wayback_snapshot_pair,
)
from .gee_analysis import _forest_loss_coordinates, _summarize_transitions
from .agric_inference import CLASS_COLORS, predict_image
from .models import LandCoverAnalysis
from .views import _changed_area_hectares


class AgricInferenceTests(TestCase):
	def test_unet_predicts_overlapping_windows_without_downscaling_snapshot(self):
		import torch

		class ColorModel:
			def __call__(self, images):
				logits = torch.zeros((images.shape[0], len(CLASS_COLORS), *images.shape[2:]))
				is_red = images[:, 0] > 0
				logits[:, 1] = 1
				logits[:, 0][is_red] = 2
				return logits

		with tempfile.TemporaryDirectory() as temp_dir:
			source = Path(temp_dir) / 'snapshot.png'
			output_dir = Path(temp_dir) / 'output'
			image = np.zeros((256, 512, 3), dtype=np.uint8)
			image[:, :256] = (255, 0, 0)
			image[:, 256:] = (0, 0, 255)
			Image.fromarray(image).save(source)

			with patch(
				'dynamics.agric_inference._load_model',
				return_value=(torch, ColorModel(), torch.device('cpu'), 256),
			):
				mask_path, composition = predict_image(source, output_dir)

			with Image.open(mask_path) as mask:
				prediction = np.asarray(mask)

		self.assertEqual(prediction.shape, image.shape)
		self.assertTrue(np.all(prediction[:, :256] == CLASS_COLORS[0]))
		self.assertTrue(np.all(prediction[:, 256:] == CLASS_COLORS[1]))
		self.assertEqual(composition[0]['percentage'], 50.0)
		self.assertEqual(composition[1]['percentage'], 50.0)


class LandCoverCompositionTests(TestCase):
	def test_composition_weights_classified_tiles_by_area(self):
		probabilities = np.array([
			[0.01, 0.01, 0.97, 0.01],
			[0.01, 0.97, 0.01, 0.01],
		])

		scores, composition = _aggregate_class_predictions(
			probabilities, np.array([2, 1])
		)

		self.assertEqual(composition[CLASS_NAMES[2]], 66.7)
		self.assertEqual(composition[CLASS_NAMES[1]], 33.3)
		self.assertEqual(sum(composition.values()), 100.0)
		np.testing.assert_allclose(scores, [0.01, 0.33, 0.65, 0.01])

	def test_transitions_pair_tiles_and_weight_partial_tile_area(self):
		before = np.array([
			[0.01, 0.96, 0.01, 0.02],
			[0.01, 0.96, 0.01, 0.02],
			[0.01, 0.01, 0.01, 0.97],
		])
		after = np.array([
			[0.01, 0.01, 0.96, 0.02],
			[0.01, 0.96, 0.01, 0.02],
			[0.01, 0.01, 0.01, 0.97],
		])

		result = _summarize_class_transitions(before, after, np.array([2, 1, 1]))

		self.assertEqual(result['changed_percentage'], 50.0)
		self.assertEqual(result['tile_count'], 3)
		self.assertEqual(result['transitions'], [{
			'from_class': 'Forest',
			'to_class': 'Urban',
			'percentage': 50.0,
			'tile_count': 1,
		}])

	def test_cnn_preprocessing_scales_pixels_to_model_range(self):
		image = np.zeros((224, 224, 3), dtype=np.uint8)
		image[0, 0] = [0, 0, 0]
		image[0, 1] = [255, 255, 255]
		image[1, 0] = [128, 64, 32]
		image[1, 1] = [64, 128, 192]

		processed = _preprocess_cnn_image(tf.convert_to_tensor(image))

		self.assertTrue(np.all(processed >= 0.0))
		self.assertTrue(np.all(processed <= 1.0))
		self.assertEqual(processed[0, 0, 0], 0.0)
		self.assertEqual(processed[0, 1, 0], 1.0)
		self.assertAlmostEqual(processed[1, 0, 1], 64 / 255.0)

	def test_cnn_input_preserves_whole_snapshot_aspect_ratio_and_scale(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			path = Path(temp_dir) / 'wide-snapshot.png'
			Image.new('RGB', (896, 448), (255, 128, 0)).save(path)

			inputs, snapshot_areas = _load_cnn_inputs(path)

		self.assertEqual(inputs.shape, (1, 224, 224, 3))
		self.assertEqual(snapshot_areas.tolist(), [896 * 448])
		np.testing.assert_array_equal(inputs[0, 0, 100], [0.0, 0.0, 0.0])
		np.testing.assert_allclose(
			inputs[0, 100, 100],
			[1.0, 128 / 255.0, 0.0],
			atol=1e-6,
		)

	def test_dynamic_world_counts_forest_loss_against_total_pixel_area(self):
		summary = _summarize_transitions({
			'10': 90,
			'15': 5,
			'11': 3,
			'60': 2,
		})

		self.assertEqual(summary['changed_percentage'], 8.0)
		self.assertEqual(summary['forest_loss']['pixels'], 8)
		self.assertEqual(summary['forest_loss']['percentage'], 8.0)
		self.assertEqual(summary['before_dominant_class'], 'Forest')
		self.assertEqual(summary['after_dominant_class'], 'Forest')
		self.assertEqual(
			[(item['from_class'], item['to_class'], item['percentage'])
			 for item in summary['transitions']],
			[('Forest', 'Urban', 5.0), ('Forest', 'Grass', 3.0)],
		)
		self.assertEqual(summary['forest_loss']['interpretation'],
			'Potential forest loss, not independently verified deforestation.')

	def test_dynamic_world_rejects_out_of_range_transition_classes(self):
		with self.assertRaises(ValueError):
			_summarize_transitions({'81': 1})

	def test_forest_loss_sample_coordinates_are_returned_as_latitude_longitude(self):
		points = _forest_loss_coordinates({
			'features': [
				{'geometry': {'coordinates': [31.12345678, -18.23456789]}},
				{'geometry': None},
			],
		})

		self.assertEqual(points, [[-18.234568, 31.123457]])

	def test_spatial_cnn_chips_retain_source_pixel_centers(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			path = Path(temp_dir) / 'snapshot.png'
			Image.new('RGB', (300, 260), (128, 64, 32)).save(path)

			inputs, centers, size = _load_cnn_spatial_inputs(path)

		self.assertEqual(inputs.shape, (4, 224, 224, 3))
		self.assertEqual(size, (300, 260))
		np.testing.assert_array_equal(centers, [
			[112, 112], [262, 112], [112, 242], [262, 242],
		])

	def test_pixel_center_is_converted_from_web_mercator_to_lat_lon(self):
		bounds = {'south': -20, 'west': 28, 'north': -16, 'east': 32}

		self.assertEqual(
			_pixel_center_to_lat_lon(50, 50, (100, 100), bounds),
			[-18.011348, 30.0],
		)

	def test_local_forest_loss_points_require_paired_confident_classifications(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			before_path = Path(temp_dir) / 'before.png'
			after_path = Path(temp_dir) / 'after.png'
			Image.new('RGB', (224, 224)).save(before_path)
			Image.new('RGB', (224, 224)).save(after_path)
			model = MagicMock()
			model.predict.return_value = np.array([
				[0.05, 0.80, 0.10, 0.05],
				[0.80, 0.10, 0.05, 0.05],
			])

			points = _predict_local_forest_loss_points(
				model, before_path, after_path,
				{'south': -20, 'west': 28, 'north': -16, 'east': 32},
			)

		self.assertEqual(points, [[-18.011348, 30.0]])


class WaybackSnapshotTests(TestCase):
	@patch('dynamics.ml_inference.time.sleep')
	@patch('dynamics.ml_inference.urllib.request.urlopen')
	def test_transient_request_error_is_retried(self, urlopen, sleep):
		response = MagicMock()
		response.__enter__.return_value = response
		response.read.return_value = b'tile-data'
		urlopen.side_effect = [URLError('temporary network error'), response]

		payload = _read_url_with_retries('tile-url', 1)

		self.assertEqual(payload, b'tile-data')
		self.assertEqual(urlopen.call_count, 2)
		sleep.assert_called_once()

	@patch('dynamics.ml_inference._fetch_map_mosaic_at_zoom')
	@patch('dynamics.ml_inference._choose_zoom', return_value=19)
	@patch('dynamics.ml_inference._wayback_release_for_year')
	def test_missing_high_zoom_tile_retries_both_years_at_shared_zoom(
		self, release_for_year, choose_zoom, fetch_mosaic
	):
		release_for_year.side_effect = [('2023-01-01', 1), ('2024-01-01', 2)]
		image = Image.new('RGB', (32, 24))

		def fetch(bounds, release_number, zoom):
			if zoom == 19 and release_number == 2:
				raise HTTPError('tile', 404, 'missing', None, None)
			return image.copy()

		fetch_mosaic.side_effect = fetch
		bounds = {'south': -18.0, 'west': 31.0, 'north': -17.9, 'east': 31.1}

		(before_image, before_release), (after_image, after_release), zoom = (
			_fetch_wayback_snapshot_pair(bounds, 2023, 2024)
		)

		self.assertEqual(zoom, 18)
		self.assertEqual(before_release, '2023-01-01')
		self.assertEqual(after_release, '2024-01-01')
		self.assertEqual(before_image.size, after_image.size)
		self.assertEqual(
			[(call.args[1], call.args[2]) for call in fetch_mosaic.call_args_list],
			[(1, 19), (2, 19), (1, 18), (2, 18)],
		)


class ReportsPageTests(TestCase):
	def test_report_loads_saved_analysis_and_shows_class_composition(self):
		session = self.client.session
		session.save()
		bounds = {'south': -17.9, 'west': 31.0, 'north': -17.8, 'east': 31.1}
		LandCoverAnalysis.objects.create(
			analysis_id='report-composition-test',
			session_key=session.session_key,
			location_name='Test area',
			before_year=2023,
			after_year=2024,
			bounds=bounds,
			outputs={
				'before_predicted_class': 'Forest',
				'after_predicted_class': 'Urban',
				'before_class_probabilities': {'Forest': 80},
				'after_class_probabilities': {'Urban': 75},
				'before_land_cover_composition': {'Forest': 100},
				'after_land_cover_composition': {'Forest': 33, 'Urban': 67},
				'local_class_transitions': {
					'changed_percentage': 60.0,
					'tile_count': 3,
					'transitions': [{
						'from_class': 'Forest',
						'to_class': 'Urban',
						'percentage': 60.0,
						'tile_count': 2,
					}],
				},
				'changed_percentage': 10,
			},
			area_hectares=100,
			changed_hectares=10,
		)

		response = self.client.get(reverse('reports'))

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'Estimated land-cover composition')
		self.assertContains(response, 'Forest 33.0%')
		self.assertContains(response, 'Urban 67.0%')

		results_response = self.client.get(reverse(
			'analysis-result-detail', args=['report-composition-test']
		))
		self.assertContains(results_response, 'Local land-cover transitions')
		self.assertContains(results_response, 'Classified tile area changing class:')
		self.assertContains(results_response, '60.00%')

		csv_response = self.client.get(reverse('download-report'))

		self.assertEqual(csv_response.status_code, 200)
		self.assertContains(csv_response, 'Earlier Urban share (%)')
		self.assertContains(csv_response, 'Later Urban share (%)')
		self.assertContains(csv_response, 'Supplementary local CNN tile transitions')
		self.assertContains(csv_response, 'Forest,Urban,60.0,2')

	def test_results_summary_uses_best_unet_dominant_classes_when_available(self):
		session = self.client.session
		session.save()
		LandCoverAnalysis.objects.create(
			analysis_id='best-unet-summary-test',
			session_key=session.session_key,
			before_year=2023,
			after_year=2026,
			bounds={'south': -18.0, 'west': 31.0, 'north': -17.9, 'east': 31.1},
			outputs={
				'before_predicted_class': 'CNN earlier class',
				'after_predicted_class': 'CNN later class',
				'agric_before_dominant_class': 'Forest',
				'agric_after_dominant_class': 'Agriculture',
				'changed_percentage': 0.0,
			},
			area_hectares=179.45,
			changed_hectares=0.0,
		)

		response = self.client.get(reverse(
			'analysis-result-detail', args=['best-unet-summary-test']
		))

		self.assertEqual(response.status_code, 200)
		summary = response.content.decode().split(
			'<div class="results-summary">', 1
		)[1].split('</div>\n\n  <section', 1)[0]
		self.assertIn('Earlier Best U-Net estimate', summary)
		self.assertIn('Later Best U-Net estimate', summary)
		self.assertIn('Forest</span>', summary)
		self.assertIn('Agriculture</span>', summary)
		self.assertIn('Pixels flagged by unvalidated Wayback change model', summary)
		self.assertContains(response, '179.45 ha')
		self.assertContains(response, '0.00 ha')
		self.assertNotIn('CNN earlier class', summary)
		self.assertNotIn('CNN later class', summary)


class LocalModelIntegrationTests(TestCase):
	@patch('dynamics.ml_inference.run_inference')
	@patch('dynamics.ml_inference._fetch_wayback_snapshot_pair')
	def test_analysis_uses_local_models_without_calling_earth_engine(
		self, fetch_snapshots, local_inference
	):
		image = Image.new('RGB', (16, 16))
		fetch_snapshots.return_value = (
			(image, '2023-01-01'),
			(image.copy(), '2024-01-01'),
			17,
		)
		local_inference.return_value = {
			'changed_percentage': 82.0,
			'maximum_change_probability': 99.0,
			'before_predicted_class': 'Forest',
			'after_predicted_class': 'Urban',
		}

		with tempfile.TemporaryDirectory() as temp_dir:
			with patch('dynamics.ml_inference.MEDIA_ROOT', Path(temp_dir)):
				outputs = run_inference_from_bounds(
					{'south': -18, 'west': 31, 'north': -17.9, 'east': 31.1},
					'local-model-integration-test',
					2023,
					2024,
				)

		self.assertEqual(outputs['changed_percentage'], 82.0)
		self.assertEqual(outputs['before_predicted_class'], 'Forest')
		self.assertEqual(outputs['after_predicted_class'], 'Urban')
		self.assertEqual(outputs['before_release_date'], '2023-01-01')
		self.assertEqual(outputs['after_release_date'], '2024-01-01')
		self.assertEqual(outputs['imagery_zoom'], 17)
		self.assertEqual(
			outputs['change_detection_source'],
			'Local change U-Net applied to Esri Wayback imagery',
		)
		local_inference.assert_called_once()


class DynamicWorldResultsTests(TestCase):
	def test_home_offers_2026_as_a_comparison_year(self):
		response = self.client.get(reverse('home'))

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, '<option value="2026">2026</option>', html=True)

	def test_home_describes_local_model_analysis(self):
		response = self.client.get(reverse('home'))

		self.assertContains(response, 'local CNN and change-detection models')

	def test_home_map_shows_source_linked_approximate_forest_alert_regions(self):
		response = self.client.get(reverse('home'))

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'High-alert forest-loss regions (approx.)')
		self.assertContains(response, 'Mapfungautsi Forest / Gokwe South')
		self.assertContains(response, 'Murehwa District')
		self.assertContains(response, 'Mutasa District')
		self.assertContains(response, 'Bindura District')
		self.assertContains(response, 'Hurungwe District')
		self.assertContains(response, 'Chimanimani / Gwindingwe plantations')
		self.assertContains(response, 'Chirinda Forest')
		self.assertContains(response, 'https://doi.org/10.33425/2770-2928.1013')
		self.assertNotContains(response, 'Historic loss / reported pressure, not live alerts.')

	@patch('dynamics.views.run_inference_from_bounds')
	def test_prepare_analysis_endpoint_runs_local_analysis_without_earth_engine(self, run_analysis):
		run_analysis.return_value = {
			'changed_percentage': 12.5,
			'before_predicted_class': 'Forest',
			'after_predicted_class': 'Urban',
		}

		response = self.client.post(reverse('create-analysis'), data={
			'before_year': 2023,
			'after_year': 2024,
			'bounds': {
				'south': -18.0,
				'west': 31.0,
				'north': -17.9,
				'east': 31.1,
			},
		}, content_type='application/json')

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.json()['outputs']['changed_percentage'], 12.5)
		self.assertIn('local_forest_loss_points', response.json()['outputs'])
		run_analysis.assert_called_once()

	@override_settings(EARTH_ENGINE_PROJECT='test-earth-engine-project')
	@patch('dynamics.gee_analysis.run_gee_analysis')
	@patch('dynamics.views.run_inference_from_bounds')
	def test_prepare_analysis_includes_dynamic_world_forest_loss_points(
		self, run_analysis, run_dynamic_world
	):
		run_analysis.return_value = {'changed_percentage': 12.5}
		run_dynamic_world.return_value = {
			'dynamic_world_before_imagery_layer': 'before imagery',
			'dynamic_world_after_imagery_layer': 'after imagery',
			'dynamic_world_before_land_cover_layer': 'before labels',
			'dynamic_world_after_land_cover_layer': 'after labels',
			'dynamic_world_change_layer': 'change mask',
			'dynamic_world_forest_loss_layer': 'forest loss',
			'before_classes': [],
			'after_classes': [],
			'transitions': [],
			'before_dominant_class': 'Forest',
			'after_dominant_class': 'Grass',
			'forest_loss': {'pixels': 3},
			'forest_loss_points': [[-18.2, 31.1]],
			'changed_percentage': 4.2,
			'changed_area_km2': 0.3,
			'area_note': 'Estimated at 10 m.',
		}

		response = self.client.post(reverse('create-analysis'), data={
			'before_year': 2023,
			'after_year': 2024,
			'bounds': {
				'south': -18.0,
				'west': 31.0,
				'north': -17.9,
				'east': 31.1,
			},
		}, content_type='application/json')

		self.assertEqual(response.status_code, 200)
		outputs = response.json()['outputs']
		self.assertTrue(outputs['dynamic_world_enabled'])
		self.assertEqual(outputs['wayback_unet_changed_percentage'], 12.5)
		self.assertEqual(outputs['changed_percentage'], 4.2)
		self.assertEqual(outputs['dynamic_world_forest_loss_points'], [[-18.2, 31.1]])

	def test_changed_hectares_uses_dynamic_world_pixel_area(self):
		self.assertEqual(
			_changed_area_hectares(10000, {
				'changed_percentage': 50,
				'dynamic_world_changed_area_km2': 0.25,
			}),
			25.0,
		)
		self.assertEqual(
			_changed_area_hectares(10000, {'changed_percentage': 2.5}),
			250.0,
		)

	def test_dynamic_world_is_primary_and_local_cnn_is_not_presented_as_truth(self):
		session = self.client.session
		session.save()
		LandCoverAnalysis.objects.create(
			analysis_id='dynamic-world-result-test',
			session_key=session.session_key,
			location_name='Forest test area',
			before_year=2023,
			after_year=2024,
			bounds={'south': -18.0, 'west': 31.0, 'north': -17.9, 'east': 31.1},
			outputs={
				'dynamic_world_enabled': True,
				'dynamic_world_before_dominant_class': 'Forest',
				'dynamic_world_after_dominant_class': 'Forest',
				'dynamic_world_changed_percentage': 2.5,
				'dynamic_world_before_classes': [
					{'name': 'Forest', 'percentage': 95.0, 'color': '#397D49'},
					{'name': 'Urban', 'percentage': 5.0, 'color': '#C4281B'},
				],
				'dynamic_world_after_classes': [
					{'name': 'Forest', 'percentage': 92.5, 'color': '#397D49'},
					{'name': 'Urban', 'percentage': 7.5, 'color': '#C4281B'},
				],
				'dynamic_world_transitions': [{
					'from_class': 'Forest',
					'to_class': 'Urban',
					'percentage': 2.5,
					'area_km2': 0.25,
				}],
				'dynamic_world_forest_loss': {
					'percentage': 2.5,
					'area_km2': 0.25,
					'interpretation': 'Potential forest loss, not independently verified deforestation.',
				},
				'dynamic_world_forest_loss_points': [[-18.2, 31.1]],
				'dynamic_world_before_land_cover_layer': 'https://earthengine.test/before/{z}/{x}/{y}',
				'dynamic_world_after_land_cover_layer': 'https://earthengine.test/after/{z}/{x}/{y}',
				'dynamic_world_change_layer': 'https://earthengine.test/change/{z}/{x}/{y}',
				'dynamic_world_forest_loss_layer': 'https://earthengine.test/forest-loss/{z}/{x}/{y}',
				'changed_percentage': 2.5,
				'before_predicted_class': 'Urban',
				'after_predicted_class': 'Urban',
			},
			area_hectares=100,
			changed_hectares=2.5,
		)

		response = self.client.get(reverse(
			'analysis-result-detail', args=['dynamic-world-result-test']
		))

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'Primary source: Dynamic World')
		self.assertContains(response, 'Potential forest loss')
		self.assertContains(response, 'Forest 92.5%')
		self.assertContains(response, 'dynamic-world-map')
		self.assertContains(response, 'Potential forest-loss mask')
		self.assertContains(response, 'Red dots mark sampled potential Forest-to-non-Forest transitions')
		self.assertContains(response, '[-18.2, 31.1]')
		self.assertNotContains(response, 'Overall result: Urban')

		report_response = self.client.get(reverse('reports'))
		self.assertContains(report_response, 'Annual Dynamic World pixel-class shares')
		self.assertContains(report_response, 'Google Dynamic World annual Sentinel-2 classifications')


class ResultsSidebarTests(TestCase):
	def test_results_sidebar_is_disabled_before_analysis(self):
		response = self.client.get(reverse('home'))

		self.assertContains(response, 'Results unavailable until an analysis is complete')
		self.assertNotContains(response, 'href="/results/"')

	def test_results_sidebar_links_to_results_after_analysis(self):
		session = self.client.session
		session['latest_analysis'] = {'analysis_id': 'completed-analysis'}
		session.save()

		response = self.client.get(reverse('home'))

		self.assertContains(response, 'href="/results/"')
		results_response = self.client.get(reverse('analysis-results'))
		self.assertContains(
			results_response,
			'title="Results" aria-label="Results" class="active" href="/results/"',
		)
		self.assertContains(
			results_response,
			'title="Change Analysis" aria-label="Change Analysis" class="" href="/"',
		)
