from django import forms
from .models import VideoUpload , ManualMask, SnapshotImage
class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['road_name', 'video']
class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = SnapshotImage
        fields = ['area_name', 'road_1image', 'road_2image', 'road_3image', 'road_4image']
class ManualMaskForm(forms.ModelForm):
    class Meta:
        model = ManualMask
        fields = ['road_name', 'mask_points']

    # Custom field to handle the polygon points input as JSON
    mask_points = forms.CharField(widget=forms.Textarea, help_text="Enter polygon points as JSON")