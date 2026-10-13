from django.db import models


class LandCoverAnalysis(models.Model):
    analysis_id = models.CharField(max_length=64, unique=True)
    session_key = models.CharField(max_length=40, db_index=True)
    location_name = models.CharField(max_length=120, blank=True, default='')
    before_year = models.PositiveSmallIntegerField()
    after_year = models.PositiveSmallIntegerField()
    bounds = models.JSONField()
    outputs = models.JSONField()
    reviewed_classes = models.JSONField(default=dict, blank=True)
    area_hectares = models.FloatField()
    changed_hectares = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']

    def as_analysis(self):
        return {
            'status': 'analysis_complete',
            'message': 'The selected imagery pair was analyzed successfully.',
            'analysis_id': self.analysis_id,
            'location_name': self.location_name,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
            'before_year': self.before_year,
            'after_year': self.after_year,
            'bounds': self.bounds,
            'outputs': self.outputs,
            'reviewed_classes': self.reviewed_classes,
            'area_hectares': self.area_hectares,
            'changed_hectares': self.changed_hectares,
        }
