import os
from helpers import detect_onnx, degress

import json
import random

import cv2

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

def get_tel():
    test_payload = {
        "data" : {
            "battery" : {
                "battery1" : {
                    "batteryCapacityPercent" : 14,
                    "batteryTemperature" : 32,
                    "currentVoltage" : 22
                },
                "battery2" : {
                    "batteryCapacityPercent" : 0,
                    "batteryTemperature" : 0,
                    "currentVoltage" : 0
                }
            },
            "telemetries" : {
                "attitude" : {
                    "q0" : 0.90594452619552612,
                    "q1" : 0.0030836670193821192,
                    "q2" : 0.0035839446354657412,
                    "q3" : -0.42337003350257874
                },
                "flightstatus" : 0,
                "position" : {
                    "altitude" : round(random.uniform(10,60), 6),
                    "latitude" : round(random.uniform(10,60), 6),
                    "longitude" : round(random.uniform(10,60), 6)
                },
                "rc" : {
                    "pitch" : 0,
                    "roll" : 0,
                    "throttle" : 0,
                    "yaw" : 0
                },
                "velocity" : {
                    "vx" : round(random.uniform(10,60), 6),
                    "vy" : round(random.uniform(10,60), 6),
                    "vz" : round(random.uniform(10,60), 6)
                }
            }
        }
    }
    return test_payload

def create_bounding_box(img_path, d, tel, inspection_id, anomalies_number):
    img_name = img_path[img_path.rindex('/')+1:]

    im = cv2.imread(img_path)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        roi=im[y:y+h,x:x+w]
        for i in d:
            image = cv2.rectangle(im,(int(i["left"]),int(i["top"])),(int(i["right"]),int(i["bottom"])),(255, 255, 51), 1)
            cv2.putText(image, str(i['anomaly'])+":"+str(round(i['anomaly_score'], 1)), (int(i["left"]),int(i["top"])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 51), 2)

    d = get_tel()
    internal = {'anomalies_number': anomalies_number}
    d.update(internal)
    tel = json.dumps(d, indent = 4)
    with open("/home/ggc_user/"+str(inspection_id)+"/"+img_name[:-3]+"json", "w") as outfile:
        outfile.write(tel)

    cv2.imwrite("/home/ggc_user/"+str(inspection_id)+"/"+img_name, roi)
    cv2.waitKey(0)

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
            tel = json.dumps(get_tel(), indent = 4)
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

            create_bounding_box(message['image_path'], res_pred["detection"], tel, message['inspection_id'], len(tmp["detection"]))

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
