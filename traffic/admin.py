from django.contrib import admin
from .models import TrafficLightState, ManualMask, VideoUpload, VehicleCount, SnapshotImage, Road , StaticTime, numberplate , maskforplate , videoam,chalan
# Register your models here.
admin.site.register(TrafficLightState)
admin.site.register(ManualMask)
admin.site.register(VideoUpload)
admin.site.register(VehicleCount)
admin.site.register(SnapshotImage)
admin.site.register(Road)
admin.site.register(StaticTime)
admin.site.register(numberplate)
admin.site.register(maskforplate)
admin.site.register(chalan)
admin.site.register(videoam)

