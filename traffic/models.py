# traffic/models.py
from django.db import models

class VideoUpload(models.Model):
    road_name = models.CharField(max_length=255)
    video = models.FileField(upload_to='videos/')
    upload_time = models.DateTimeField(auto_now_add=True)

class TrafficLightState(models.Model):
    road_name = models.CharField(max_length=255)
    state = models.CharField(max_length=10, choices=[('green', 'Green'), ('red', 'Red')])
    countdown_timer = models.IntegerField()

class VehicleCount(models.Model):
    road_name = models.CharField(max_length=255)
    vehicle_type = models.CharField(max_length=50)
    count = models.IntegerField()

class ManualMask(models.Model):
    road_name = models.CharField(max_length=255)
    mask_points = models.JSONField()  # Store the polygon points as JSON

class SnapshotImage(models.Model):
    area_name = models.CharField(max_length=255)
    road_1image = models.ImageField(upload_to='output_frames/')
    road_2image = models.ImageField(upload_to='output_frames/')
    road_3image = models.ImageField(upload_to='output_frames/')
    road_4image = models.ImageField(upload_to='output_frames/')

