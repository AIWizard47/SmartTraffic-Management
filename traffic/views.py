
import time
from io import BytesIO
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect
from .models import VideoUpload, VehicleCount, ManualMask, TrafficLightState, SnapshotImage
from ultralytics import YOLO
import cv2
import os
import numpy as np
from .forms import VideoUploadForm, ManualMaskForm, ImageUploadForm
from django.contrib import messages
import requests
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


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
            return redirect('dashboard')
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
def start_process(request):
    # Get the current traffic light states
    traffic_lights = TrafficLightState.objects.all()
    area_name = request.GET.get('area_name')  # or request.POST.get('area_name')
    snapshot_image = SnapshotImage.objects.filter(area_name=area_name).first()
    # Get the vehicle counts
    vehicle_counts = VehicleCount.objects.all()

    # Get the manual masks
    masks = ManualMask.objects.all()

    # Pair up traffic lights, vehicle counts, and masks
    light_data = zip(traffic_lights, vehicle_counts, masks)

    context = {
        'light_data': light_data,
        'snapshot_image': snapshot_image,  # Include snapshot image in the context
    }
    return render(request, 'start_process.html', context)

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
            captured_frames += 1
            frame_filename = os.path.join(output_dir, f"frame_{captured_frames}.jpg")
            cv2.imwrite(frame_filename, img)
            print(f"Saved frame {captured_frames} at time {frame_count // fps + 1} seconds")

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
        road_name=road_name,
        defaults={
            'state': 'Green' if total_count > 0 else 'Red',
            'countdown_timer': max(min(45, total_count), 10)  # or whatever logic you want for the countdown timer
        }
    )
    # Return some response or redirect
    # return render(request, 'dashboard.html')
    return redirect("dashboard")
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('dashboard')  # Redirect to the dashboard or any other page after upload
    else:
        form = VideoUploadForm()

    return render(request, 'upload_video.html', {'form': form})


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

            return redirect('dashboard')
        else:
            messages.error(request, "There was an error uploading the image. Please try again.")
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
    }
    return render(request, 'upload_image.html', context)

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
    start_countdown(traffic_light)

    return redirect("/")
def dashboard(request):
    # Get the current traffic light states
    traffic_lights = TrafficLightState.objects.all()

    # Get the vehicle counts
    vehicle_counts = VehicleCount.objects.all()

    # Get the manual masks
    masks = ManualMask.objects.all()

    context = {
        'traffic_lights': traffic_lights,
        'vehicle_counts': vehicle_counts,
        'masks': masks,
    }
    return render(request, 'dashboard.html', context)


def process_all_roads(request):
    # Define road names in the order of processing
    road_names = ["road_1", "road_2", "road_3", "road_4"]

    # Load YOLO model
    model = YOLO('Yolo-weight/yolov8l.pt')

    # Get base URL for media files
    base_url = request.build_absolute_uri('/media/')

    # Fetch SnapshotImage object
    snapshot_image = SnapshotImage.objects.filter(area_name='neelbad').first()

    if not snapshot_image:
        return HttpResponseBadRequest("No snapshot image found.")

    # Process each road sequentially
    i = 10
    while i:

        i -= 1
        for road_name in road_names:
            # Inside process_video or process_image
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                "traffic_updates_group",
                {
                    "type": "send_update",
                }
            )
            # Fetch the image URL from the snapshot_image
            image_url = getattr(snapshot_image, f'road_{road_name[-1]}image').url

            # Construct the full URL
            full_image_url = base_url + image_url.split('/media/')[1]

            if not full_image_url:
                return HttpResponseBadRequest(f"No image found for {road_name}.")

            # Get the image and process it
            response = requests.get(full_image_url)
            if response.status_code != 200:
                return HttpResponseBadRequest(f"Failed to fetch image from {full_image_url}")

            img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Get mask points from the database
            mask_obj = ManualMask.objects.filter(road_name=road_name).first()
            if mask_obj:
                data = json.loads(mask_obj.mask_points)
                mask_points = [(int(item['x']), int(item['y'])) for item in data]
            else:
                return HttpResponseBadRequest(f"No mask found for {road_name}.")

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
                if vehicle_type == 'bus':
                    count += 5
                    total_count += 5
                elif vehicle_type == 'truck':
                    count += 3
                    total_count += 3
                elif vehicle_type == 'car':
                    count += 2
                    total_count += 2

                VehicleCount.objects.update_or_create(
                    vehicle_type=vehicle_type,
                    defaults={
                        'road_name': road_name,
                        'count': count,
                    }
                )

            # Update or create the traffic light state for the given road
            countdown_timer = max(min(45, total_count), 10)
            traffic_light, created = TrafficLightState.objects.update_or_create(
                road_name=road_name,
                defaults={
                    'countdown_timer': countdown_timer,
                }
            )

            # Start the countdown for the current road
            start_countdown(traffic_light)

            # If total_count is 0, move to the next road
            if total_count == 0:
                continue  # Move to the next road in the sequence
    return redirect("/")

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

def start_countdown(traffic_light):
    timer = traffic_light.countdown_timer
    channel_layer = get_channel_layer()

    while timer >= 0:
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
        time.sleep(1)
        timer -= 1

    traffic_light.state = 'Red'
    traffic_light.save()
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