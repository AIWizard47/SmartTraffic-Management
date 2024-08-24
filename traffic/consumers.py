from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer

class TrafficUpdateConsumer(WebsocketConsumer):
    def connect(self):
        async_to_sync(self.channel_layer.group_add)(
            "traffic_updates_group",
            self.channel_name
        )
        self.accept()

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(
            "traffic_updates_group",
            self.channel_name
        )

    def send_update(self, event):
        self.send(text_data=json.dumps(event))