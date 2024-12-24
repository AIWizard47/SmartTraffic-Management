from rest_framework import serializers
from .models import ManualMask, SnapshotImage

# Serializer for ManualMask
class ManualMaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = ManualMask
        fields = ['road_name', 'mask_points']

# Serializer for SnapshotImage
class SnapshotImageSerializer(serializers.ModelSerializer):
    images = serializers.SerializerMethodField()

    class Meta:
        model = SnapshotImage
        fields = ['area_name', 'images']

    def get_images(self, obj):
        return [
            {'label': 'Road 1', 'url': obj.road_1image.url if obj.road_1image else None},
            {'label': 'Road 2', 'url': obj.road_2image.url if obj.road_2image else None},
            {'label': 'Road 3', 'url': obj.road_3image.url if obj.road_3image else None},
            {'label': 'Road 4', 'url': obj.road_4image.url if obj.road_4image else None},
        ]
