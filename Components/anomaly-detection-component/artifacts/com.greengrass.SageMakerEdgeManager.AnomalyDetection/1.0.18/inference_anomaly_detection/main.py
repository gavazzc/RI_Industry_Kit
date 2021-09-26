import os
from helpers import detect_onnx, degress

import json

import concurrent.futures
import sys
import time
import traceback
import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    SubscribeToTopicRequest,
    SubscriptionResponseMessage,
    UnauthorizedError,
    PublishToTopicRequest,
    PublishMessage,
    JsonMessage,
    BinaryMessage
)


#MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "best.onnx")
print(os.environ.get("SMEM_IC_MODEL_DIR"))
MODEL_PATH = os.path.join(os.path.expandvars(os.environ.get("SMEM_IC_MODEL_DIR")),'best.onnx')

CLASS_NAMES = ['Tracker', 'Cell', 'Diode', 'String', 'Cell Multi', 'Module']

ANCHORS = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]

print("Start Detection...", end='\r')

topic = os.environ.get("IMAGE_PUBLISHER_TOPIC")
detection_topic = os.environ.get("DETECTION_RESULT_TOPIC")
always_send_image = os.environ.get("ALWAYS_SEND_IMAGE").lower()

TIMEOUT = 100

ipc_client = awsiot.greengrasscoreipc.connect()

class StreamHandler(client.SubscribeToTopicStreamHandler):
    def __init__(self):
        super().__init__()
        self.ipc_client = None

    def on_stream_event(self, event: SubscriptionResponseMessage) -> None:
        try:
            print(event)
            message_string = str(event.binary_message.message, "utf-8")
            message = json.loads(message_string)
            #message = event.json_message.message
            print("Received new message")
            print(message)
            tmp = detect_onnx(image_path=message['image_path'],model_path=MODEL_PATH,anchors=ANCHORS,num_classes=len(CLASS_NAMES), get_metadata=False)

            '''lat = degress(tmp["metadata"]["GPS GPSLatitude"])
            lon = degress(tmp["metadata"]["GPS GPSLongitude"])
            lat = -lat if tmp["metadata"]["GPS GPSLatitudeRef"].values[0] != "N" else lat
            lon = -lon if tmp["metadata"]["GPS GPSLongitudeRef"].values[0] != "E" else lon
            '''
            lat = 0
            lon = 0

            res_pred = {
                "Filename": os.path.basename(message['image_path']),
                "Latitude": lat,
                "Longitude": lon,
            }
            tot_det = []
            if len(tmp["detection"]) > 0:
                for values in tmp["detection"]:
                    tmp_det = {}
                    tmp_det["left"] = values[0]
                    tmp_det["right"] = values[2]
                    tmp_det["top"] = values[1]
                    tmp_det["bottom"] = values[3]
                    tmp_det["anomaly_score"] = round(values[4], 3)
                    tmp_det["anomaly"] = CLASS_NAMES[int(values[5])]
                    tot_det.append(tmp_det)
            res_pred["detection"] = tot_det
            if always_send_image=='True' or len(res_pred["detection"])>0:
                self.publish_detection_result(res_pred)


        except:
            traceback.print_exc()

    def publish_detection_result(self, result: dict ):
        if self.ipc_client is None:
            self.ipc_client = awsiot.greengrasscoreipc.connect()

        request = PublishToTopicRequest()
        request.topic = detection_topic
        publish_message = PublishMessage()
        print(f"result : {result}")
        print({"payload":result,"topic":"store/data"})
        payload=json.dumps({"payload":str(result),"topic":"store/data"})
        publish_message.binary_message = BinaryMessage()
        publish_message.binary_message.message = bytes(payload, "utf-8")
        
        request.publish_message = publish_message
        operation = self.ipc_client.new_publish_to_topic()
        operation.activate(request)
        futureResponse = operation.get_response()
        try:
            futureResponse.result(TIMEOUT)
            print('Successfully published result to topic: ' + detection_topic)
            sys.stdout.flush()

        except concurrent.futures.TimeoutError:
            print('Timeout occurred while publishing to topic: ' + topic, file=sys.stderr)
        except UnauthorizedError as e:
            print('Unauthorized error while publishing to topic: ' + topic, file=sys.stderr)
            raise e
        except Exception as e:
            print('Exception while publishing to topic: ' + topic, file=sys.stderr)
            raise e


    def on_stream_error(self, error: Exception) -> bool:
        print("Received a stream error.", file=sys.stderr)
        traceback.print_exc()
        return False  # Return True to close stream, False to keep stream open.

    def on_stream_closed(self) -> None:
        print('Subscribe to topic stream closed.')

class StreamHandlerBasic(client.SubscribeToTopicStreamHandler):
 
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: SubscriptionResponseMessage) -> None:
        try:
            print(event)
            message_string = str(event.binary_message.message, "utf-8")

            # Handle message.
            message_obj = json.loads(message_string)
            
        except:
            traceback.print_exc()

    def on_stream_error(self, error: Exception) -> bool:
        # Handle error.
        return True  # Return True to close stream, False to keep stream open.

    def on_stream_closed(self) -> None:
        # Handle close.
        pass

print("Instantiating ipc client.")
print(f"always_send_image: {always_send_image}")



request = SubscribeToTopicRequest()
request.topic = topic
handler = StreamHandler()
operation = ipc_client.new_subscribe_to_topic(handler)
future = operation.activate(request)
future.result(TIMEOUT)

while True:
    time.sleep(10)

operation.close()
