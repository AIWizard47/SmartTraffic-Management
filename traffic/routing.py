from django.urls import path
from .consumers import TrafficLightConsumer

websocket_urlpatterns = [
    path('ws/traffic/$', TrafficLightConsumer.as_asgi()),
]