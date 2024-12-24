# traffic/models.py
from django.db import models
from django.db.models import F

class VideoUpload(models.Model):
    road_name = models.CharField(max_length=255)
    video = models.FileField(upload_to='videos/')
    upload_time = models.DateTimeField(auto_now_add=True)

class TrafficLightState(models.Model):
    road_name = models.CharField(max_length=255)
    state = models.CharField(max_length=10, choices=[('green', 'Green'), ('red', 'Red'), ('yellow', 'Yellow')])
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

class Road(models.Model):
    road_name = models.CharField(max_length=50)  # Road name (e.g., Road 1, Road 2, etc.)
    time_in_sec = models.IntegerField()  # Time in seconds

    def __str__(self):
        return f"{self.road_name} - {self.time_in_sec} seconds"


    @staticmethod
    def update_or_add_time(road_name, additional_time):
        """
        Add to the existing countdown time for the given road or create a new record.
        """
        road, created = Road.objects.get_or_create(road_name=road_name)
        if not created:
            road.time_in_sec = F('time_in_sec') + additional_time  # Use F to avoid race conditions
            road.save()
            road.refresh_from_db()  # Refresh after F operation
        else:
            road.time_in_sec = additional_time
            road.save()
        return road

    def reset_all_times():
        """
        Reset the countdown time for all roads to 0.
        """
        Road.objects.all().update(time_in_sec=0)


class StaticTime(models.Model):
    static_name = models.CharField(max_length=30)
    data = models.JSONField()
    avg_time = models.IntegerField(max_length=10)

class maskforplate(models.Model):
    area_name = models.CharField(max_length=10)
    mask = models.JSONField()

class numberplate(models.Model):
    area_name = models.CharField(max_length=20)
    person_name = models.CharField(max_length=30)
    num_plate = models.CharField(max_length=9)
    phone_number = models.CharField(max_length=10)

class videoam(models.Model):
    video = models.FileField(upload_to='videos/')
class chalan(models.Model):
    video = models.FileField(upload_to='videos/')

class uter(models.Model):
    video = models.FileField(upload_to='videos/')
