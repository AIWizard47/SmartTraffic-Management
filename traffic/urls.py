# traffic/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # New index route
    path('upload/', views.upload_video, name='upload_video'),
    path('process_video/', views.process_video, name='process_video'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('create_mask/', views.create_mask, name='create_mask'),
    path('process_image/', views.process_image, name='process_image'), #http://127.0.0.1:8000/process_image/?image_url=http://127.0.0.1:8000/media/output_frames/frame_1_7C5J3KX.jpg&road_name=road1
    path('image_upload/', views.upload_image, name='upload_image'),
    path('start_process/', views.start_process, name='start_process'),
    path('start_process_all/', views.process_all_roads, name='start_process_all'),
    path('get-updates/', views.get_updates, name='get_updates'),
    path('api/current-state/', views.get_current_state, name='get_current_state'),
    # path('update_timer/<int:light_id>/', views.update_timer, name='update_timer'),
]