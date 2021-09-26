import time
import traceback
import json
import sys

import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    SubscribeToTopicRequest,
    SubscriptionResponseMessage
)

from zipfile import ZipFile
import os
from os.path import basename
import boto3
import threading

TIMEOUT = 10

ipc_client = awsiot.greengrasscoreipc.connect()

def zip_folder(inspection_id):
    folder = f"{db_folder}/{inspection_id}"
    zip_folder = f'{db_folder}/temp_fold'
    os.makedirs(zip_folder, exist_ok=True)
    zip_file = f'{zip_folder}/{inspection_id}.zip'
    # create a ZipFile object : https://thispointer.com/python-how-to-create-a-zip-archive-from-multiple-files-or-directory/
    with ZipFile(zip_file, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))
    return zip_file

def upload_folder_to_s3(inspection_id):
    print(f"Upload started")
    zip_path = zip_folder(inspection_id)
    print(f"zip file : {zip_path}")
    object_name = f'inspections/{inspection_id}.zip'
    print(f"S3 object name : {object_name}")
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(zip_path, s3_bucket, object_name)
        print(f"Upload result : {response}")
    except ClientError as e:
        print(e)
    print(f"Removing zip file : {zip_path}")
    os.remove(zip_path)
    
                    
class StreamHandler(client.SubscribeToTopicStreamHandler):
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: SubscriptionResponseMessage) -> None:
        try:
            print(event)
            message_string = str(event.binary_message.message, "utf-8")

            # Handle message.
            message_obj = json.loads(message_string)
            if "inspection_id" in message_obj and not message_obj["inspection_id"] == None:
                inspection_id = message_obj["inspection_id"]
                x = threading.Thread(target=upload_folder_to_s3, args=(inspection_id,))
                x.start()
            else:
                print(f"No inspection_id field in the export command. message_obj : {message_obj}")
        except:
            traceback.print_exc()

    def on_stream_error(self, error: Exception) -> bool:
        # Handle error.
        return True  # Return True to close stream, False to keep stream open.

    def on_stream_closed(self) -> None:
        # Handle close.
        pass

print("Export starting...")

db_folder = sys.argv[1]

print(db_folder)

s3_bucket = sys.argv[2]

topic = "export/command"


request = SubscribeToTopicRequest()
request.topic = topic
handler = StreamHandler()
operation = ipc_client.new_subscribe_to_topic(handler) 
future = operation.activate(request)
future.result(TIMEOUT)

# Keep the main thread alive, or the process will exit.
while True:
    time.sleep(10)
    
# To stop subscribing, close the operation stream.
operation.close()