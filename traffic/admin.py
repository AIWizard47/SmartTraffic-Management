from django.contrib import admin
from .models import TrafficLightState, ManualMask, VideoUpload, VehicleCount, SnapshotImage
# Register your models here.
admin.site.register(TrafficLightState)
admin.site.register(ManualMask)
admin.site.register(VideoUpload)
admin.site.register(VehicleCount)
admin.site.register(SnapshotImage)