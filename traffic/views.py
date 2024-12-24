import time
from io import BytesIO
from django.http import HttpResponseBadRequest, HttpResponse
from django.shortcuts import render, redirect
import subprocess
from .models import VideoUpload, VehicleCount, ManualMask, TrafficLightState, SnapshotImage, Road,videoam,chalan
from ultralytics import YOLO
import cv2
import os
import numpy as np
from rest_framework.permissions import IsAuthenticated
from .forms import VideoUploadForm, ManualMaskForm, ImageUploadForm
from django.contrib import messages
import requests
from PIL import Image
from django.http import JsonResponse
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from background_task import background
from django.contrib.auth.decorators import login_required
from django.core.management import call_command
import threading
from django.core.cache import cache
from django.core.files.base import ContentFile
import io
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.db import connection
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ManualMaskSerializer, SnapshotImageSerializer
times_for_video = 0

# def create_mask(request):
#     mask_instance, created = ManualMask.objects.get_or_create(road_name=road_name)
#
#     if request.method == 'POST':
#         # Process form data for each road's mask points
#         road_1mask_points = request.POST.get('road_1mask_points')
#         road_2mask_points = request.POST.get('road_2mask_points')
#         road_3mask_points = request.POST.get('road_3mask_points')
#         road_4mask_points = request.POST.get('road_4mask_points')
#
#         # Store the mask points as JSON
#         mask_instance.road_1mask_points = road_1mask_points if road_1mask_points else []
#         mask_instance.road_2mask_points = road_2mask_points if road_2mask_points else []
#         mask_instance.road_3mask_points = road_3mask_points if road_3mask_points else []
#         mask_instance.road_4mask_points = road_4mask_points if road_4mask_points else []
#
#         mask_instance.save()
#     # Fetch the image associated with the road or a default image
#     #     # road_image = RoadImage.objects.first()  # Adjust as per your application's logic
#     #
#     #     # if road_image and road_image.image:
#     #     #     image_url = road_image.image.url  # Ensure MEDIA_URL is correctly configured
#     #     # else:
#     # Example image URLs for the four images (replace with actual logic)
#     image_url1 = '/media/output_frames/frame_1.jpg'
#     image_url2 = '/media/output_frames/frame_2.jpg'
#     image_url3 = '/media/output_frames/frame_3.jpg'
#     image_url4 = '/media/output_frames/frame_4.jpg'
#
#     context = {
#         'form': form,
#         'image_url1': image_url1,
#         'image_url2': image_url2,
#         'image_url3': image_url3,
#         'image_url4': image_url4,
#     }
#     return render(request, 'create_mask.html', context)

def chalaan(request):
    vid = chalan.objects.first()  # Fetch a single video (e.g., the first video)

    return render(request,"numberplate.html",{'vid': vid})
def ambulance(request):
    vid = videoam.objects.first()  # Fetch a single video (e.g., the first video)
    return render(request, "ambulance.html", {'vid': vid})
def index(request):
    # Get distinct area names and road names from the SnapshotImage model
    area_names = SnapshotImage.objects.values_list('area_name', flat=True).distinct()
    road_names = ['1', '2', '3', '4']  # Assuming roads are labeled as 1, 2, 3, 4

    # Get image URLs for each area
    snapshot_images = SnapshotImage.objects.all()

    context = {
        'area_names': area_names,
        'road_names': road_names,
        'snapshot_images': snapshot_images,
    }
    return render(request, 'index.html', context)
@login_required
def create_mask(request):
    if request.method == 'POST':
        form = ManualMaskForm(request.POST)
        if form.is_valid():
            # Get the road name and mask points from the form
            road_name = form.cleaned_data['road_name']
            mask_points = form.cleaned_data['mask_points']

            # Save or update the mask for the given road name
            ManualMask.objects.update_or_create(
                road_name=road_name,
                defaults={'mask_points': mask_points}
            )
            messages.success(request, "Mask saved successfully!")
            return redirect('create_mask')
        else:
            messages.error(request, "There was an error saving the mask. Please try again.")
    else:
        form = ManualMaskForm()

    # Fetch the SnapshotImage object based on the area_name
    area_name = request.GET.get('area_name')  # or request.POST.get('area_name')
    snapshot_image = SnapshotImage.objects.filter(area_name=area_name).first()

    # If a SnapshotImage object exists, create the dropdown options
    if snapshot_image:
        images = [
            {'label': 'Road 1', 'url': snapshot_image.road_1image.url},
            {'label': 'Road 2', 'url': snapshot_image.road_2image.url},
            {'label': 'Road 3', 'url': snapshot_image.road_3image.url},
            {'label': 'Road 4', 'url': snapshot_image.road_4image.url},
        ]
    else:
        images = []

    context = {
        'form': form,
        'images': images,
    }
    return render(request, 'create_mask.html', context)
# @login_required
def process_video(request):
    # Get the uploaded video
    video = VideoUpload.objects.last()

    # Load YOLO model
    model = YOLO('Yolo-weight/yolov8l.pt')

    # Open video file
    cap = cv2.VideoCapture(video.video.path)

    # Create output directory if it doesn't exist
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Get total frames and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration_seconds = total_frames / fps

    # Determine the start time for capturing (last 5 seconds)
    start_capture_time = duration_seconds - 5
    start_capture_frame = int(start_capture_time * fps)

    frame_count = 0
    captured_frames = 0

    # Get mask points from the database
    mask_obj = ManualMask.objects.filter(road_name=video.road_name).first()
    data = json.loads(mask_obj.mask_points)
    mask_points = [(int(item['x']), int(item['y'])) for item in data] if mask_obj else None

    # Create the mask (example image dimensions; adjust as necessary)
    success, img = cap.read()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if mask_points:
        # Convert the points to the correct format and type
        contour = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [contour], 255)

    # Prepare for capturing frames
    while True:
        success, img = cap.read()
        if not success:
            break

        # Capture frames only in the last 5 seconds
        if frame_count == start_capture_frame:
            # captured_frames += 1
            # frame_filename = os.path.join(output_dir, f"frame_{captured_frames}.jpg")
            # cv2.imwrite(frame_filename, img)
            # print(f"Saved frame {captured_frames} at time {frame_count // fps + 1} seconds")
            captured_frames += 1

            # Generate a unique filename for the frame
            frame_filename = f"frame_{captured_frames}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save the frame to disk
            cv2.imwrite(frame_path, img)
            print(f"Saved frame {captured_frames} at time {frame_count // fps + 1} seconds")

            # Read the saved frame to save it in the database
            with open(frame_path, 'rb') as f:
                # Fetch or create the SnapshotImage object for the given area_name
                snapshot, created = SnapshotImage.objects.update_or_create(
                    area_name="neelbad",  # Adjust the area_name as needed
                    defaults={}  # No need to set anything else in defaults for now
                )

                # Overwrite or create the road_1image field
                snapshot.road_1image.save(frame_filename, ContentFile(f.read()))
                snapshot.save()

        frame_count += 1

    # Release video capture
    cap.release()

    # Dictionary to store the count of each vehicle type
    vehicle_counts = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
    total_count = 0
    # Process each saved image to count vehicles
    for filename in os.listdir(output_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_dir, filename)
            img = cv2.imread(img_path)

            # Run YOLO model on the saved image
            results = model(img, stream=True)

            # Process detection results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Check if the center of the bounding box is within the mask
                    if mask is not None:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
                            idx = box.cls[0]
                            nameOfVehicle = model.names[int(idx)]

                            if nameOfVehicle in vehicle_counts:
                                vehicle_counts[nameOfVehicle] += 1
                                total_count += 1

    # Save the counts to the database
    for vehicle_type, count in vehicle_counts.items():
        VehicleCount.objects.update_or_create(
            # road_name=road_name,
            vehicle_type=vehicle_type,
            defaults={
                'road_name':  video.road_name,
                'count': count,
            }
        )

    # Update or create the TrafficLightState based on the total vehicle count
    TrafficLightState.objects.update_or_create(
        road_name=video.road_name,
        defaults={
            'state': 'Green' if total_count > 0 else 'Red',
            'countdown_timer': max(min(45, total_count), 10)  # or whatever logic you want for the countdown timer
        }
    )
    # Return some response or redirect
    # return render(request, 'dashboard.html')
    return redirect("dashboard")
@login_required
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('dashboard')  # Redirect to the dashboard or any other page after upload
    else:
        form = VideoUploadForm()

    return render(request, 'upload_video.html', {'form': form})

@login_required
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            area_name = form.cleaned_data['area_name']

            # Update or create the SnapshotImage entry
            snapshot_image, created = SnapshotImage.objects.update_or_create(
                area_name=area_name,
                defaults={
                    'road_1image': form.cleaned_data.get('road_1image'),
                    'road_2image': form.cleaned_data.get('road_2image'),
                    'road_3image': form.cleaned_data.get('road_3image'),
                    'road_4image': form.cleaned_data.get('road_4image'),
                }
            )

            if created:
                messages.success(request, "New image set created successfully!")
            else:
                messages.success(request, "Image set updated successfully!")

            create_mask_url = reverse('create_mask')  # Get the base URL for create_mask
            url_with_params = f"{create_mask_url}?area_name={area_name}"

            # Redirect to the create_mask page with the query parameter
            return HttpResponseRedirect(url_with_params)
        else:
            messages.error(request, "There was an error uploading the image. Please try again.")
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
    }
    return render(request, 'upload_image.html', context)
# @login_required
def process_image(request):
    # Get the image URL and road name from GET parameters
    image_url = request.GET.get('image_url')
    road_name = request.GET.get('road_name')

    if not image_url or not road_name:
        return HttpResponseBadRequest("Missing 'image_url' or 'road_name' parameter.")

    # Load YOLO model
    model = YOLO('Yolo-weight/yolov8l.pt')
    # print(model.names)
    # Fetch image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Convert the image to a numpy array (OpenCV format)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Get mask points from the database
    mask_obj = ManualMask.objects.filter(road_name=road_name).first()
    if mask_obj:
        data = json.loads(mask_obj.mask_points)
        mask_points = [(int(item['x']), int(item['y'])) for item in data]
    else:
        return HttpResponseBadRequest("No mask found for the provided road name.")

    # Create the mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    contour = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [contour], 255)

    # Run YOLO model on the image
    results = model(img, stream=True)

    # Dictionary to store the count of each vehicle type
    vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    total_count = 0

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check if the center of the bounding box is within the mask
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
                idx = box.cls[0]
                nameOfVehicle = model.names[int(idx)]

                if nameOfVehicle in vehicle_counts:
                    vehicle_counts[nameOfVehicle] += 1
                    total_count += 1

    # Save the counts to the database
    for vehicle_type, count in vehicle_counts.items():
        VehicleCount.objects.update_or_create(
            # road_name=road_name,
            vehicle_type=vehicle_type,
            defaults={
                'road_name':  road_name,
                'count': count,
            }
        )

    countdown_timer = max(min(45, total_count), 10)

    # Get or create the traffic light state for the given road
    traffic_light, created = TrafficLightState.objects.update_or_create(
        road_name=road_name,
        defaults={
            'countdown_timer': countdown_timer,
            # 'last_updated': timezone.now(),  # You may want to track when it was last updated
        }
    )

    # Start the countdown
    # start_countdown(traffic_light)

    return redirect("/")
@login_required
def dashboard(request):
    # Get all traffic light states
    traffic_lights = TrafficLightState.objects.all()

    # Find the green light traffic
    green_light = traffic_lights.filter(state='Green').first()

    # Fetch the video for the green light road
    green_road_video = None
    if green_light:
        green_road_video = VideoUpload.objects.filter(road_name=green_light.road_name).first()

    # Fetch the videos for roads with red lights
    red_light_videos = []
    red_lights = traffic_lights.filter(state='Red')
    if red_lights.exists():
        red_light_videos = VideoUpload.objects.filter(road_name__in=red_lights.values_list('road_name', flat=True))

    # Get all vehicle counts
    vehicle_counts = VehicleCount.objects.all()

    # Get manual masks
    masks = ManualMask.objects.all()

    # Create the context
    context = {
        'traffic_lights': traffic_lights,
        'vehicle_counts': vehicle_counts,
        'masks': masks,
        'green_road_video': green_road_video,
        'red_light_videos': red_light_videos,
    }

    return render(request, 'dashboard.html', context)


@background(schedule=5)  # No delay in scheduling
def process_all_roads_task():
    road_names = ["road_1", "road_2", "road_3", "road_4"]
    model = YOLO('Yolo-weight/yolov8l.pt')
    base_url = "http://127.0.0.1:8000/media/"

    snapshot_image = SnapshotImage.objects.filter(area_name='neelbad').first()
    # print(snapshot_image)
    if not snapshot_image:
        print('Image not found')
        return

    i = 10
    print('Task is running')
    while i:
        i -= 1

        for road_name in road_names:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "traffic_updates_group",
                {
                    "type": "send_update",
                }
            )
            image_url = getattr(snapshot_image, f'road_{road_name[-1]}image').url
            full_image_url = base_url + image_url.split('/media/')[1]

            response = requests.get(full_image_url)
            if response.status_code != 200:
                continue

            img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            mask_obj = ManualMask.objects.filter(road_name=road_name).first()
            if mask_obj:
                data = json.loads(mask_obj.mask_points)
                mask_points = [(int(item['x']), int(item['y'])) for item in data]
            else:
                continue

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            contour = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour], 255)

            results = model(img, stream=True)
            vehicle_counts = {'person': 0, 'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0,'elephant': 0, 'cat': 0, 'dog': 0, 'sheep': 0, 'cow':0}
            total_count = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
                        idx = box.cls[0]
                        nameOfVehicle = model.names[int(idx)]
                        # print(model.names)
                        if nameOfVehicle in vehicle_counts:
                            vehicle_counts[nameOfVehicle] += 1
                            total_count += 1

            for vehicle_type, count in vehicle_counts.items():

                if vehicle_type == 'bus':
                    total_count += 5
                elif vehicle_type == 'truck':
                    total_count += 3
                elif vehicle_type == 'car':
                    total_count += 2
                elif vehicle_type == 'cow' and count > 4:
                    total_count += 4
                elif vehicle_type == 'dog' and count > 10:
                    total_count += 5
                elif vehicle_type == 'motorcycle':
                    total_count += 2
                elif vehicle_type == 'person':
                    total_count += abs(count-vehicle_counts['motorcycle'])+10

                print(total_count,vehicle_counts,vehicle_type)
                VehicleCount.objects.update_or_create(
                    vehicle_type=vehicle_type,
                    defaults={
                        'road_name': road_name,
                        'count': count,
                    }
                )



                traffic_light, created = TrafficLightState.objects.update_or_create(
                    road_name=road_name,
                    defaults={
                        'state': 'Green' if total_count > 0 else 'Red',
                        'countdown_timer': max(min(45, total_count), 10),
                    }
                )
            traffic_light_map = {
                'road_1': TrafficLightState.objects.filter(road_name='road_2').first(),
                'road_2': TrafficLightState.objects.filter(road_name='road_3').first(),
                'road_3': TrafficLightState.objects.filter(road_name='road_4').first(),
                'road_4': TrafficLightState.objects.filter(road_name='road_1').first(),
            }
            traffic_light_next = traffic_light_map.get(road_name)
            start_countdown(traffic_light,traffic_light_next)
@login_required
def process_all_roads(request):
    # # Define road names in the order of processing
    # road_names = ["road_1", "road_2", "road_3", "road_4"]
    #
    # # Load YOLO model
    # model = YOLO('Yolo-weight/yolov8l.pt')
    # model = YOLO('Yolo-weight/yolov8l.pt')
    # model = YOLO('Yolo-weight/yolov8l.pt')
    #
    # # Get base URL for media files
    # base_url = request.build_absolute_uri('/media/')
    #
    # # Fetch SnapshotImage object
    # snapshot_image = SnapshotImage.objects.filter(area_name='neelbad').first()
    #
    # if not snapshot_image:
    #     return HttpResponseBadRequest("No snapshot image found.")
    #
    # # Process each road sequentially
    # i = 10
    # while i:
    #
    #     i -= 1
    #     for road_name in road_names:
    #         # Inside process_video or process_image
    #         channel_layer = get_channel_layer()
    #         async_to_sync(channel_layer.group_send)(
    #             "traffic_updates_group",
    #             {
    #                 "type": "send_update",
    #             }
    #         )
    #         # Fetch the image URL from the snapshot_image
    #         image_url = getattr(snapshot_image, f'road_{road_name[-1]}image').url
    #
    #         # Construct the full URL
    #         full_image_url = base_url + image_url.split('/media/')[1]
    #
    #         if not full_image_url:
    #             return HttpResponseBadRequest(f"No image found for {road_name}.")
    #
    #         # Get the image and process it
    #         response = requests.get(full_image_url)
    #         if response.status_code != 200:
    #             return HttpResponseBadRequest(f"Failed to fetch image from {full_image_url}")
    #
    #         img = Image.open(BytesIO(response.content))
    #         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #
    #         # Get mask points from the database
    #         mask_obj = ManualMask.objects.filter(road_name=road_name).first()
    #         if mask_obj:
    #             data = json.loads(mask_obj.mask_points)
    #             mask_points = [(int(item['x']), int(item['y'])) for item in data]
    #         else:
    #             return HttpResponseBadRequest(f"No mask found for {road_name}.")
    #
    #         # Create the mask
    #         mask = np.zeros(img.shape[:2], dtype=np.uint8)
    #         contour = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))
    #         cv2.fillPoly(mask, [contour], 255)
    #
    #         # Run YOLO model on the image
    #         results = model(img, stream=True)
    #
    #         # Dictionary to store the count of each vehicle type
    #         vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    #         total_count = 0
    #
    #         # Process detection results
    #         for r in results:
    #             boxes = r.boxes
    #             for box in boxes:
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #
    #                 # Check if the center of the bounding box is within the mask
    #                 center_x = (x1 + x2) // 2
    #                 center_y = (y1 + y2) // 2
    #                 if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
    #                     idx = box.cls[0]
    #                     nameOfVehicle = model.names[int(idx)]
    #
    #                     if nameOfVehicle in vehicle_counts:
    #                         vehicle_counts[nameOfVehicle] += 1
    #                         total_count += 1
    #
    #         # Save the counts to the database
    #         for vehicle_type, count in vehicle_counts.items():
    #             if vehicle_type == 'bus':
    #                 count += 5
    #                 total_count += 5
    #             elif vehicle_type == 'truck':
    #                 count += 3
    #                 total_count += 3
    #             elif vehicle_type == 'car':
    #                 count += 2
    #                 total_count += 2
    #
    #             VehicleCount.objects.update_or_create(
    #                 vehicle_type=vehicle_type,
    #                 defaults={
    #                     'road_name': road_name,
    #                     'count': count,
    #                 }
    #             )
    #
    #         # Update or create the traffic light state for the given road
    #         countdown_timer = max(min(45, total_count), 10)
    #         traffic_light, created = TrafficLightState.objects.update_or_create(
    #             road_name=road_name,
    #             defaults={
    #                 'countdown_timer': countdown_timer,
    #             }
    #         )
    #
    #         # Start the countdown for the current road
    #         start_countdown(traffic_light)
    #
    #         # If total_count is 0, move to the next road
    #         if total_count == 0:
    #             continue  # Move to the next road in the sequence
    # return redirect("/")



    # process_all_roads_task(schedule=0)  # This starts the background task
    # process_all_roads_task(schedule=0)
    # process_all_roads_task = process_all_roads_task.now
    # return redirect('dashboard')
    Road.reset_all_times()
    road_names = ["road_1", "road_2", "road_3", "road_4"]
    for i in range(4):
        traffic_light, created = TrafficLightState.objects.update_or_create(
            road_name=road_names[i],
            defaults={
                'state': 'Red',
                'countdown_timer': 0,
            }
        )
    with connection.cursor() as cursor:
        cursor.execute("UPDATE background_task SET locked_by = NULL, locked_at = NULL;")
    messages.success(request, "Task locks cleared successfully!")
    venv_python = os.path.join('D:\\VS_code\\VSCode\\ObjectDitections\\venv12.5', 'Scripts', 'python.exe')
    try:
        # Run the custom management command to start the task worker
        subprocess.Popen([venv_python, 'manage.py', 'process_tasks'])
        print('WORKING +++++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        return redirect('dashboard')  # Redirect after starting the worker

    except Exception as e:
        print(e)
        return HttpResponse(f"An error occurred: {e}")




# def start_countdown(traffic_light):
#     # Implement the logic to start the countdown timer
#     timer = traffic_light.countdown_timer
#     while timer >= 0:
#         # Update the countdown timer in the database
#         traffic_light.countdown_timer = timer
#         traffic_light.save()
#         time.sleep(1)  # Countdown by 1 second
#         timer -= 1
#
#     # After countdown ends, update the state
#     traffic_light.state = 'Red'  # or any logic for state change
#     traffic_light.save()

def start_yellow_light(traffic_light, channel_layer):
    # Yellow light countdown for 5 seconds
    for i in range(5,0,-1):
        traffic_light.state = 'Yellow'
        traffic_light.countdown_timer = i  # Display the yellow light countdown as 5 seconds
        traffic_light.save()

        # Send the countdown update to the WebSocket
        async_to_sync(channel_layer.group_send)(
            "traffic_lights",
            {
                "type": "send_traffic_update",
                "data": {
                    "light_id": traffic_light.id,
                    "countdown_timer": traffic_light.countdown_timer,
                    "state": traffic_light.state,
                },
            },
        )
        time.sleep(1)


def start_countdown(traffic_light, traffic_light_next):
    global times_for_video
    timer = traffic_light.countdown_timer
    times_for_video += timer
    road_name = traffic_light_next.road_name
    channel_layer = get_channel_layer()
    # Update or add time for the next road
    road = Road.update_or_add_time(traffic_light.road_name, timer+1)

    # Notify via channels (if needed)
    async_to_sync(channel_layer.group_send)(
        "traffic_lights",
        {
            "type": "traffic.update",
            "road_name": road_name,
            "time_in_sec": road.time_in_sec,
        }
    )
    while timer > 0:
        traffic_light.state = 'Green'
        traffic_light.countdown_timer = timer
        traffic_light.save()

        # Send the countdown update to the WebSocket
        async_to_sync(channel_layer.group_send)(
            "traffic_lights",
            {
                "type": "send_traffic_update",
                "data": {
                    "light_id": traffic_light.id,
                    "countdown_timer": timer,
                    "state": traffic_light.state,
                },
            },
        )

        # When the timer reaches 5 seconds, stcd art the yellow light countdown in parallel
        if timer == 5:
            process_videooo(roadname=road_name, specific_times=times_for_video)
            # threading.Thread(target=process_videooo, args=(roadname==road_name, specific_times==times_for_video)).start()
            threading.Thread(target=start_yellow_light, args=(traffic_light, channel_layer)).start()
            threading.Thread(target=start_yellow_light, args=(traffic_light_next, channel_layer)).start()
            # extract_frame_at_timer()
        time.sleep(1)
        timer -= 1

    # After countdown ends, update the state to Red
    traffic_light.state = 'Red'
    traffic_light.countdown_timer = 0
    traffic_light.save()

    # Send the final state update to the WebSocket
    async_to_sync(channel_layer.group_send)(
        "traffic_lights",
        {
            "type": "send_traffic_update",
            "data": {
                "light_id": traffic_light.id,
                "countdown_timer": traffic_light.countdown_timer,
                "state": traffic_light.state,
            },
        },
    )
# @csrf_exempt
# def update_timer(request, light_id):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         countdown_timer = data.get('countdown_timer')
#
#         # Update the traffic light state in the database
#         TrafficLightState.objects.filter(id=light_id).update(curr_countdown_timer=countdown_timer)
#
#         return JsonResponse({'status': 'success'})


def get_updates(request):
    vehicle_counts = VehicleCount.objects.all().values('vehicle_type', 'count')
    traffic_lights = TrafficLightState.objects.all().values('id', 'road_name', 'state', 'countdown_timer')

    data = {
        'vehicle_counts': list(vehicle_counts),
        'traffic_lights': list(traffic_lights),
    }

    return JsonResponse(data)
@login_required
def get_current_state(request):
    vehicle_counts = VehicleCount.objects.all()
    traffic_lights = TrafficLightState.objects.all()

    vehicle_data = {vc.vehicle_type: vc.count for vc in vehicle_counts}
    traffic_light_data = {
        tls.road_name: {
            'state': tls.state,
            'countdown_timer': tls.countdown_timer
        } for tls in traffic_lights
    }

    response_data = {
        'vehicle_counts': vehicle_data,
        'traffic_lights': traffic_light_data,
    }

    return JsonResponse(response_data)

@login_required
def capture_snapshot(video_obj, timestamp_seconds, road_name):
    """
    Captures a snapshot from the given video at the specified timestamp and
    saves or updates it in the SnapshotImage model.

    :param video_obj: VideoUpload object containing video information.
    :param timestamp_seconds: Time period in seconds where the snapshot should be captured.
    :param road_name: The name of the road (e.g., 'road_1') to update the snapshot.
    :return: None
    """

    # Open video file using OpenCV
    video_path = video_obj.video.path
    cap = cv2.VideoCapture(video_path)

    # Calculate the frame number based on the timestamp
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = int(fps * timestamp_seconds)

    # Set the video capture to the specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    if not success:
        print("Failed to capture the frame.")
        cap.release()
        return

    # Convert the frame to a PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Save the image to an in-memory file
    img_io = io.BytesIO()
    pil_img.save(img_io, format='JPEG')
    img_content = ContentFile(img_io.getvalue(), f"{road_name}_snapshot.jpg")

    # Fetch or create the SnapshotImage entry for the given road name
    snapshot, created = SnapshotImage.objects.update_or_create(
        area_name=video_obj.area_name,  # Assuming the area_name is related to the video
        defaults={f'{road_name}image': img_content}
    )

    # Release the video capture
    cap.release()
    print(f"Snapshot for {road_name} captured and saved successfully.")

# def api_dashboard_data(request):
#     # Fetch traffic light states
#     traffic_lights = TrafficLightState.objects.all()
#     traffic_light_data = [
#         {
#             "road_name": traffic_light.road_name,
#             "state": traffic_light.state,
#             "countdown_timer": traffic_light.countdown_timer,
#         }
#         for traffic_light in traffic_lights
#     ]
#
#     # Fetch vehicle counts
#     vehicle_counts = VehicleCount.objects.all()
#     vehicle_count_data = [
#         {
#             "road_name": vehicle_count.road_name,
#             "vehicle_type": vehicle_count.vehicle_type,
#             "count": vehicle_count.count,
#         }
#         for vehicle_count in vehicle_counts
#     ]
#
#     # Fetch manual masks
#     masks = ManualMask.objects.all()
#     mask_data = [
#         {
#             "road_name": mask.road_name,
#             "mask_points": json.loads(mask.mask_points),  # Convert JSON string to a Python object
#         }
#         for mask in masks
#     ]
#
#     # Fetch snapshot images
#     snapshot_images = SnapshotImage.objects.all()
#     snapshot_image_data = [
#         {
#             "area_name": snapshot_image.area_name,
#             "road_1image": snapshot_image.road_1image.url if snapshot_image.road_1image else None,
#             "road_2image": snapshot_image.road_2image.url if snapshot_image.road_2image else None,
#             "road_3image": snapshot_image.road_3image.url if snapshot_image.road_3image else None,
#             "road_4image": snapshot_image.road_4image.url if snapshot_image.road_4image else None,
#         }
#         for snapshot_image in snapshot_images
#     ]
#
#     # Combine all the data
#     response_data = {
#         "traffic_lights": traffic_light_data,
#         "vehicle_counts": vehicle_count_data,
#         "masks": mask_data,
#         "snapshot_images": snapshot_image_data,
#     }
#
#     return JsonResponse(response_data, safe=False)
#



class DashboardDataAPIView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            # Fetch traffic light states
            traffic_lights = TrafficLightState.objects.all()
            traffic_light_data = [
                {
                    "road_name": traffic_light.road_name,
                    "state": traffic_light.state,
                    "countdown_timer": traffic_light.countdown_timer,
                }
                for traffic_light in traffic_lights
            ]

            # Fetch vehicle counts
            vehicle_counts = VehicleCount.objects.all()
            vehicle_count_data = [
                {
                    "road_name": vehicle_count.road_name,
                    "vehicle_type": vehicle_count.vehicle_type,
                    "count": vehicle_count.count,
                }
                for vehicle_count in vehicle_counts
            ]

            # Fetch manual masks
            masks = ManualMask.objects.all()
            mask_data = [
                {
                    "road_name": mask.road_name,
                    "mask_points": json.loads(mask.mask_points),  # Convert JSON string to a Python object
                }
                for mask in masks
            ]

            # Fetch snapshot images
            snapshot_images = SnapshotImage.objects.all()
            snapshot_image_data = [
                {
                    "area_name": snapshot_image.area_name,
                    "road_1image": snapshot_image.road_1image.url if snapshot_image.road_1image else None,
                    "road_2image": snapshot_image.road_2image.url if snapshot_image.road_2image else None,
                    "road_3image": snapshot_image.road_3image.url if snapshot_image.road_3image else None,
                    "road_4image": snapshot_image.road_4image.url if snapshot_image.road_4image else None,
                }
                for snapshot_image in snapshot_images
            ]

            # Combine all the data
            response_data = {
                "traffic_lights": traffic_light_data,
                "vehicle_counts": vehicle_count_data,
                "masks": mask_data,
                "snapshot_images": snapshot_image_data,
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CreateMaskAPI(APIView):
    # permission_classes = [IsAuthenticated]
    def post(self, request, *args, **kwargs):
        serializer = ManualMaskSerializer(data=request.data)
        if serializer.is_valid():
            road_name = serializer.validated_data['road_name']
            mask_points = serializer.validated_data['mask_points']

            # Save or update the mask for the given road name
            ManualMask.objects.update_or_create(
                road_name=road_name,
                defaults={'mask_points': mask_points}
            )
            return Response({"message": "Mask saved successfully!"}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, *args, **kwargs):
        area_name = request.query_params.get('area_name')
        snapshot_image = SnapshotImage.objects.filter(area_name=area_name).first()

        if not snapshot_image:
            return Response({"message": "No snapshot image found for the given area name."}, status=status.HTTP_404_NOT_FOUND)

        snapshot_serializer = SnapshotImageSerializer(snapshot_image)
        return Response(snapshot_serializer.data, status=status.HTTP_200_OK)


class UploadedImageGroupAPI(APIView):
    def post(self, request, *args, **kwargs):
        """
        Handle POST requests to upload multiple images for a specific area.
        """
        data = request.data
        area_name = data.get('area_name')

        # Check if area_name exists
        if not area_name:
            return Response({"error": "Area name is required."}, status=status.HTTP_400_BAD_REQUEST)

        road_1image = data.get('road_1image')
        road_2image = data.get('road_2image')
        road_3image = data.get('road_3image')
        road_4image = data.get('road_4image')

        # Create or update the record for the given area
        image_group, created = SnapshotImage.objects.update_or_create(
            area_name=area_name,
            defaults={
                'road_1image': road_1image,
                'road_2image': road_2image,
                'road_3image': road_3image,
                'road_4image': road_4image,
            }
        )

        serializer = SnapshotImageSerializer(image_group)
        return Response(
            {
                "message": "Images uploaded successfully!",
                "data": serializer.data
            },
            status=status.HTTP_201_CREATED
        )

    def get(self, request, *args, **kwargs):
        """
        Handle GET requests to retrieve images for a specific area.
        """
        area_name = request.query_params.get('area_name')

        if not area_name:
            return Response({"error": "Area name is required to fetch images."}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch images for the given area
        image_group = SnapshotImage.objects.filter(area_name=area_name).first()

        if not image_group:
            return Response(
                {"error": "No images found for the specified area."},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = SnapshotImageSerializer(image_group)
        return Response(serializer.data, status=status.HTTP_200_OK)

#Test function

# def extract_frame_at_timer(countdown_timer=4, fps=30):
#     """
#     Extract a specific frame from a video when countdown reaches 0.
#
#     Args:
#         video_upload_id (int): ID of the VideoUpload object.
#         countdown_timer (int): Countdown timer value in seconds.
#         fps (int): Frames per second of the video (default=30).
#
#     Returns:
#         SnapshotImage: The saved snapshot object.
#     """
#     # Fetch video upload object
#     # video = VideoUpload.objects.filter(road_name="neelbad").first()
#     # video_upload_id = video.id
#     # video_upload = VideoUpload.objects.get(id=video_upload_id)
#     # video_path = video_upload.video.path
#     video_path = 'http://127.0.0.1:8000/media/videos/videoplayback_sU1dTJi.mp4'
#     # road_name = video_upload.road_name # 'neelbad'
#     road_name = "road_1"
#     # Calculate the target frame index
#     target_frame_index = countdown_timer * fps
#
#     # Initialize VideoCapture
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError("Unable to open video file")
#
#     # Set the frame position
#     cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
#     success, frame = cap.read()
#
#     if not success:
#         raise ValueError(f"Unable to read frame at index {target_frame_index}")
#
#     # Create output directory
#     output_dir = os.path.join(settings.MEDIA_ROOT, 'snapshot_frames')
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Save the extracted frame
#     frame_filename = f"frame_countdown_{road_name}_{countdown_timer}.jpg"
#     frame_path = os.path.join(output_dir, frame_filename)
#     cv2.imwrite(frame_path, frame)
#
#     # Save the frame to the database
#     with open(frame_path, 'rb') as f:
#         snapshot = SnapshotImage(area_name=road_name)
#         snapshot.road_1image.save(frame_filename, ContentFile(f.read()))  # Save to `road_1image` field for now
#         snapshot.save()
#
#     # Release VideoCapture
#     cap.release()
#
#     return snapshot


def extract_frame_at_timer(countdown_timer=4, fps=30):
    video_path = 'http://127.0.0.1:8000/media/videos/videoplayback_sU1dTJi.mp4'
    road_name = "road_1"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_timer = total_frames // fps

    if countdown_timer > max_timer:
        raise ValueError(f"Countdown timer {countdown_timer} exceeds video duration of {max_timer} seconds")

    target_frame_index = countdown_timer * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
    success, frame = cap.read()

    if not success:
        raise ValueError(f"Unable to read frame at index {target_frame_index}")

    output_dir = os.path.join(settings.MEDIA_ROOT, 'snapshot_frames')
    os.makedirs(output_dir, exist_ok=True)

    frame_filename = f"frame_countdown_{road_name}_{countdown_timer}.jpg"
    frame_path = os.path.join(output_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

    with open(frame_path, 'rb') as f:
        snapshot = SnapshotImage(area_name=road_name)
        snapshot.road_1image.save(frame_filename, ContentFile(f.read()))
        snapshot.save()

    cap.release()
    return snapshot



# def process_videooo(areaname = "neelbad",SpacificTime):
#     # Get the uploaded video
#     video = VideoUpload.objects.last()
#
#     # Open video file
#     cap = cv2.VideoCapture(video.video.path)
#
#     # Create output directory if it doesn't exist
#     output_dir = "output_frames"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Get total frames and FPS
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     duration_seconds = total_frames / fps
#
#     # Determine the start time for capturing (last 5 seconds)
#     start_capture_time = duration_seconds - 5
#     start_capture_frame = int(start_capture_time * fps)
#
#     frame_count = 0
#     captured_frames = 0
#
#     # Get mask points from the database
#     # mask_obj = ManualMask.objects.filter(road_name=video.road_name).first()
#     # data = json.loads(mask_obj.mask_points)
#     # mask_points = [(int(item['x']), int(item['y'])) for item in data] if mask_obj else None
#
#     # Create the mask (example image dimensions; adjust as necessary)
#     # success, img = cap.read()
#     # mask = np.zeros(img.shape[:2], dtype=np.uint8)
#
#     # if mask_points:
#     #     # Convert the points to the correct format and type
#     #     contour = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))
#     #     cv2.fillPoly(mask, [contour], 255)
#
#     # Prepare for capturing frames
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         # Capture frames only in the last 5 seconds
#         if frame_count == start_capture_frame:
#             # captured_frames += 1
#             # frame_filename = os.path.join(output_dir, f"frame_{captured_frames}.jpg")
#             # cv2.imwrite(frame_filename, img)
#             # print(f"Saved frame {captured_frames} at time {frame_count // fps + 1} seconds")
#             captured_frames += 1
#
#             # Generate a unique filename for the frame
#             frame_filename = f"frame_{captured_frames}.jpg"
#             frame_path = os.path.join(output_dir, frame_filename)
#
#             # Save the frame to disk
#             cv2.imwrite(frame_path, img)
#             print(f"Saved frame {captured_frames} at time {frame_count // fps + 1} seconds")
#
#             # Read the saved frame to save it in the database
#             with open(frame_path, 'rb') as f:
#                 # Fetch or create the SnapshotImage object for the given area_name
#                 snapshot, created = SnapshotImage.objects.update_or_create(
#                     area_name=areaname,  # Adjust the area_name as needed
#                     defaults={}  # No need to set anything else in defaults for now
#                 )
#
#                 # Overwrite or create the road_1image field
#                 snapshot.road_1image.save(frame_filename, ContentFile(f.read()))
#                 snapshot.save()
#
#         frame_count += 1
#
#     # Release video capture
#     cap.release()
#
#     # Dictionary to store the count of each vehicle type
#     # vehicle_counts = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
#     # total_count = 0
#     # # Process each saved image to count vehicles
#     # for filename in os.listdir(output_dir):
#     #     if filename.endswith(".jpg"):
#     #         img_path = os.path.join(output_dir, filename)
#     #         img = cv2.imread(img_path)
#     #
#     #         # Run YOLO model on the saved image
#     #         results = model(img, stream=True)
#     #
#     #         # Process detection results
#     #         for r in results:
#     #             boxes = r.boxes
#     #             for box in boxes:
#     #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#     #
#     #                 # Check if the center of the bounding box is within the mask
#     #                 if mask is not None:
#     #                     center_x = (x1 + x2) // 2
#     #                     center_y = (y1 + y2) // 2
#     #                     if cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0:
#     #                         idx = box.cls[0]
#     #                         nameOfVehicle = model.names[int(idx)]
#     #
#     #                         if nameOfVehicle in vehicle_counts:
#     #                             vehicle_counts[nameOfVehicle] += 1
#     #                             total_count += 1
#     #
#     # # Save the counts to the database
#     # for vehicle_type, count in vehicle_counts.items():
#     #     VehicleCount.objects.update_or_create(
#     #         # road_name=road_name,
#     #         vehicle_type=vehicle_type,
#     #         defaults={
#     #             'road_name':  video.road_name,
#     #             'count': count,
#     #         }
#     #     )
#     #
#     # # Update or create the TrafficLightState based on the total vehicle count
#     # TrafficLightState.objects.update_or_create(
#     #     road_name=video.road_name,
#     #     defaults={
#     #         'state': 'Green' if total_count > 0 else 'Red',
#     #         'countdown_timer': max(min(45, total_count), 10)  # or whatever logic you want for the countdown timer
#     #     }
#     # )
#     # Return some response or redirect
#     # return render(request, 'dashboard.html')
#     return redirect("dashboard")




def process_videooo(roadname, specific_times, areaname="neelbad"):
    """
    Process video and capture frames at specific timestamps.
    :param areaname: Name of the area to associate with the snapshot.
    :param specific_times: A single timestamp in seconds or a list of timestamps.
    :param roadname: Name of the road to fetch the video for.
    """
    # Get the uploaded video for the specific road
    print(roadname, " Time : ", specific_times)
    try:
        video = VideoUpload.objects.get(road_name=roadname)
    except VideoUpload.DoesNotExist:
        raise ValueError(f"No video found for road name: {roadname}")

    # Open the video file
    cap = cv2.VideoCapture(video.video.path)

    # Create output directory if it doesn't exist
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Get FPS and total duration
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps

    # Ensure specific_times is a list
    if isinstance(specific_times, (int, float)):
        specific_times = [specific_times]

    # Ensure the specified times are within the video duration
    for t in specific_times:
        if t < 0 or t > duration_seconds:
            raise ValueError(f"Invalid time {t} seconds. Video duration is {duration_seconds:.2f} seconds.")

    # Convert times to frame indices
    target_frames = [int(t * fps) for t in specific_times]
    target_frames_set = set(target_frames)  # Use a set for quick lookup

    frame_count = 0
    captured_frames = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Check if the current frame matches any target frame
        if frame_count in target_frames_set:
            captured_frames += 1

            # Generate a unique filename for the frame
            frame_filename = f"frame_{captured_frames}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save the frame to disk
            cv2.imwrite(frame_path, img)
            print(f"Saved frame {captured_frames} at time {frame_count // fps} seconds")

            # Save the frame in the database
            with open(frame_path, 'rb') as f:
                # Fetch or create the SnapshotImage object for the given area_name
                snapshot, created = SnapshotImage.objects.update_or_create(
                    area_name=areaname,
                    defaults={}  # No need to set anything else in defaults for now
                )

                # Overwrite or create the road_1image field
                if roadname == 'road_1':
                    snapshot.road_1image.save(frame_filename, ContentFile(f.read()))
                elif roadname == 'road_2':
                    snapshot.road_2image.save(frame_filename, ContentFile(f.read()))
                elif roadname == 'road_3':
                    snapshot.road_3image.save(frame_filename, ContentFile(f.read()))
                elif roadname == 'road_4':
                    snapshot.road_4image.save(frame_filename, ContentFile(f.read()))
                snapshot.save()

        frame_count += 1

    # Release the video capture
    cap.release()

    return redirect("dashboard")

