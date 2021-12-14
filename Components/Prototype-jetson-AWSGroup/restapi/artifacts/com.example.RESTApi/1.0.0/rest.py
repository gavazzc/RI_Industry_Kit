from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import HTMLResponse
from starlette.routing import Route
from starlette.config import Config
import base64
import concurrent.futures
import sys
import time, shutil
import traceback
from os import environ, path
import os
import json
import datetime

import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import (
    PublishToTopicRequest,
    PublishMessage,
    BinaryMessage,
    UnauthorizedError
)

config = Config(".env")

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
            "altitude" : 6.7140679359436035,
            "latitude" : 0,
            "longitude" : 0
         },
         "rc" : {
            "pitch" : 0,
            "roll" : 0,
            "throttle" : 0,
            "yaw" : 0
         },
         "velocity" : {
            "vx" : 0,
            "vy" : 0,
            "vz" : 0.00097626011120155454
         }
      }
   }
}

TIMEOUT = 10

dbp = config('DB_PATH', default="NOTFOUND")

print(f"DB_PATH from .env {dbp}")

DB_PATH = environ.get('DB_PATH') #config('DB_PATH')

ipc_client = awsiot.greengrasscoreipc.connect()

def publish(topic, payload):
    try:
        print("publishing")
        
        request = PublishToTopicRequest()
        request.topic = topic
        publish_message = PublishMessage()
        publish_message.binary_message = BinaryMessage()
        publish_message.binary_message.message = bytes(payload, "utf-8")
        request.publish_message = publish_message
        operation = ipc_client.new_publish_to_topic()
        operation.activate(request)
        futureResponse = operation.get_response()
        print("published")

        try:
            futureResponse.result(TIMEOUT)
            print('Successfully published to topic: ' + topic)
        except concurrent.futures.TimeoutError:
            print('Timeout occurred while publishing to topic: ' + topic, file=sys.stderr)
        except UnauthorizedError as e:
            print('Unauthorized error while publishing to topic: ' + topic, file=sys.stderr)
            raise e
        except Exception as e:
            print('Exception while publishing to topic: ' + topic, file=sys.stderr)
            raise e
    except InterruptedError:
        print('Publisher interrupted.')
    except Exception:
        print('Exception occurred when using IPC.', file=sys.stderr)
        traceback.print_exc()
        exit(1)
    print("returning answer")

def read_folder(inspection_id):
    ret = []
    for root,dirs,files in os.walk(f"{DB_PATH}/{inspection_id}"):
        for f in files:
            with open(f"{root}/{f}") as df:
                data = df.read()
                ret.append(json.loads(data))
    return {"d" : ret, "DB_PATH": DB_PATH, "inspection_id": inspection_id}

async def homepage(request):
    return JSONResponse({'hello': 'world'})

def publish_images(inspection_id):
    print("Scraping directory...")
    for filename in os.listdir("/home/jetson/inspection_images/"):
        publish("drone/image", json.dumps({"image_path":"/home/jetson/inspection_images/"+filename, "inspection_id":inspection_id}))
    print("Published")

# Inspection start/stop function. It handles http://host/inspection/{start/stop} path.
# There must be an inspection_id in the query parameters for start.
async def inspection(request):
    command = request.path_params['command']
    inspection_id = "no_id"
    if "inspection_id" in request.query_params:
        inspection_id = request.query_params["inspection_id"]
        if (inspection_id != None):
            shutil.rmtree("/home/ggc_user/"+inspection_id, ignore_errors=True)
            os.mkdir("/home/ggc_user/"+inspection_id)

    timer_period = -1
    if "timer_period" in request.query_params:
        timer_period = request.query_params["timer_period"]
    topic = f"store/command"
    payload = f'{{"topic":"inspection/command", "inspection_id":"{inspection_id}", "command": "{command}", "timer_period": {timer_period} }}'
    publish(topic, payload)
    publish_images(inspection_id)
    return JSONResponse({'topic': topic, 'payload': payload})

'''async def publish_test_data(request):
    publish("store/command", json.dumps({"topic":"store/data","payload":test_payload}))
    return JSONResponse({'result': 'published'})'''

async def get_all_frames(request):
    content1 = '<!DOCTYPE html> <html> <head> <title>View all frames</title> <style> table { font-family: arial, sans-serif; border-collapse: collapse; width: 80%; } td, th { border: 1px solid #dddddd; text-align: left; padding: 8px; } tr:nth-child(even) { background-color: #dddddd; } </style> </head> <body> <div>'
    content2 = ''
    content3 = ''
    content4 = '</div> </body></html>'
    inc = 0

    if not os.path.exists('/home/ggc_user/aws-demo-1/') or len(os.listdir('/home/ggc_user/aws-demo-1/')) == 0:
        content2 = '<h1 style="margin-left:30px;margin-bottom_20px;font-family: Arial, Helvetica, sans-serif;">Demo - Waiting for an analyzed frame ...</h1>'
    else:    
        content2 = '<h1 style="margin-left:30px;margin-bottom_20px;font-family: Arial, Helvetica, sans-serif;">Demo - View all analyzed frames</h1>'
        for filename in os.listdir("/home/ggc_user/aws-demo-1/"):
            if(filename[-3:] == "jpg"):
                inc+=1
                image = open("/home/ggc_user/aws-demo-1/"+filename, 'rb')
                with open("/home/ggc_user/aws-demo-1/"+filename[:-3]+"json", 'r') as openfile:
                    json_object = json.load(openfile)
                image_read = image.read()
                image_64_encode = base64.encodebytes(image_read)
                content3 += '<div style="float: left;width:40%;padding:10px;"> <h3 style="width:100%;margin-left: 30px;">Frame '+str(inc)+'</h3> <table style="width:100%;margin-left: 30px;"> <tr> <td><b>Anomalies detected</b></td> <td style="color:red"><b>'+str(json_object['anomalies_number'])+'</b></td> </tr> <tr> <td><b>Datetime</b></td> <td>'+str(datetime.datetime.now())+'</td> </tr> <tr> <td><b>Latitude</b></td> <td>'+str(json_object['data']['telemetries']['position']['latitude'])+'</td> </tr> <tr> <td><b>Longitude</b></td> <td>'+str(json_object['data']['telemetries']['position']['longitude'])+'</td> </tr> <tr> <td><b>Altitude</b></td> <td>'+str(json_object['data']['telemetries']['position']['altitude'])+'</td> </tr> </table> <img src="data:image/jpg;base64,'+image_64_encode.decode("utf-8")+'" alt="" style="margin-left:30px;width:100%;margin-top: 10px;"></div>'     
    tot = content1+content2+content3+content4
    return HTMLResponse(tot)

async def get_latest_frame(request):
    content1 = '<!DOCTYPE html> <html> <head> <title>View Latest Frame</title> <style> table { font-family: arial, sans-serif; border-collapse: collapse; width: 80%; } td, th { border: 1px solid #dddddd; text-align: left; padding: 8px; } tr:nth-child(even) { background-color: #dddddd; } </style> </head> <body> <div>'
    content2 = ''
    content3 = ''
    content4 = '</div> </body></html>'
    inc = 0
    if not os.path.exists('/home/ggc_user/aws-demo-1/') or len(os.listdir('/home/ggc_user/aws-demo-1/')) == 0:
        content2 = '<h1 style="margin-left:30px;margin-bottom_20px;font-family: Arial, Helvetica, sans-serif;">Demo - Waiting for an analyzed frame ...</h1>'
    else:
        content2 = '<h1 style="margin-left:30px;margin-bottom_20px;font-family: Arial, Helvetica, sans-serif;">Demo - View the latest analyzed frame</h1>'
        for filename in os.listdir("/home/ggc_user/aws-demo-1/"):
            if(filename[-3:] == "jpg"):
                inc+=1
                image = open("/home/ggc_user/aws-demo-1/"+filename, 'rb')
                with open("/home/ggc_user/aws-demo-1/"+filename[:-3]+"json", 'r') as openfile:
                    json_object = json.load(openfile)
                image_read = image.read()
                image_64_encode = base64.encodebytes(image_read)
                content3 = '<div><table style="width:31%;float:left;margin-left: 30px;"> <tr> <td><b>Anomalies detected</b></td> <td style="color:red"><b>'+str(json_object['anomalies_number'])+'</b></td> </tr><tr> <td><b>Datetime</b></td> <td>'+str(datetime.datetime.now())+'</td> </tr> <tr> <td><b>Latitude</b></td> <td>'+str(json_object['data']['telemetries']['position']['latitude'])+'</td> </tr> <tr> <td><b>Longitude</b></td> <td>'+str(json_object['data']['telemetries']['position']['longitude'])+'</td> </tr> <tr> <td><b>Altitude</b></td> <td>'+str(json_object['data']['telemetries']['position']['altitude'])+'</td> </tr> </table> </div> <div style="text-align:center;"> <img src="data:image/jpg;base64,'+image_64_encode.decode("utf-8")+'" alt="" style="margin-left:20px;"></div>'
    tot =  content1+content2+content3+content4
    return HTMLResponse(tot)

async def get_data(request):
    # read files, merge, return
    if not "inspection_id" in request.query_params:
        return JSONResponse({'error': 'inspection_id parameter must be passed.'})
    inspection_id = request.query_params["inspection_id"]
    ret = read_folder(inspection_id)
    data = json.dumps(json.loads(ret), indent=4)
    return JSONResponse({'data': data})

async def export_to_s3(request):
    if not "inspection_id" in request.query_params:
        return JSONResponse({'error': 'inspection_id parameter must be passed.'})
    inspection_id = request.query_params["inspection_id"]
    topic = f"export/command"
    payload = f'{{"topic":"export/command", "inspection_id":"{inspection_id}" }}'
    publish(topic, payload)
    return JSONResponse({'result': 'exporting'})

routes = [
    Route("/", endpoint=homepage),
    Route("/inspection/{command}", endpoint=inspection),
    Route("/data/get", endpoint=get_data),
    Route("/data/export", endpoint=export_to_s3),
    Route("/getAllFrames", endpoint=get_all_frames),
    Route("/getLatestFrame", endpoint=get_latest_frame)
]

app = Starlette(debug=True, routes=routes)
