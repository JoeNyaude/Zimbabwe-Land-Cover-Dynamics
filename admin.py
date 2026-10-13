from django.contrib import admin
from .models import LandCoverAnalysis


@admin.register(LandCoverAnalysis)
class LandCoverAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        'location_name',
        'before_year',
        'after_year',
        'changed_percentage',
        'area_hectares',
        'created_at',
    )
    list_filter = ('before_year', 'after_year', 'created_at')
    search_fields = ('location_name', 'analysis_id', 'session_key')
    readonly_fields = ('analysis_id', 'session_key', 'created_at')
    ordering = ('-created_at',)

    @admin.display(description='Changed pixels (%)')
    def changed_percentage(self, obj):
        return obj.outputs.get('changed_percentage', '-')
