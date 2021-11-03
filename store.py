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

from file_ops import FileManager

TIMEOUT = 10

ipc_client = awsiot.greengrasscoreipc.connect()
                    
class StreamHandler(client.SubscribeToTopicStreamHandler):
    file_manager = None
    def __init__(self, file_manager):
        super().__init__()
        self.file_manager = file_manager

    def on_stream_event(self, event: SubscriptionResponseMessage) -> None:
        try:
            print(event)
            message_string = str(event.binary_message.message, "utf-8")

            # Handle message.
            message_obj = json.loads(message_string)
            topic_name = message_obj["topic"]
            if topic_name == "store/data":
                print(f"Data store.")
                self.file_manager.put(message_obj["payload"])
            elif topic_name == "inspection/command":
                print(f"Inspection command.")
                if "command" in message_obj:
                    if message_obj["command"] == "start":
                        print(f"Start command.")
                        if "inspection_id" in message_obj and not message_obj["inspection_id"] == None:
                            print(f"Set inspection id.")
                            self.file_manager.set_inspection_id(message_obj["inspection_id"])
                            if "timer_period" in message_obj and not message_obj["timer_period"] == -1:
                                print(f"Set timer period.")
                                self.file_manager.set_timer_period(message_obj["timer_period"])
                        else:
                            print(f"No inspection_id field in the inspection start command. message_obj : {message_obj}")
                    elif message_obj["command"] == "stop":
                        print(f"Stop command.")
                        self.file_manager.set_inspection_id("no_id")
                    elif message_obj["command"] == "delete":
                        print(f"Delete command.")
                        if "inspection_id" in message_obj and not message_obj["inspection_id"] == None:
                            print(f"Set inspection id.")
                            self.file_manager.delete_inspection(message_obj["inspection_id"])
                        else:
                            print(f"No inspection_id field in the inspection delete command. message_obj : {message_obj}")
                else:
                    print(f"No 'command' field in the payload. {message_obj}")
            else:
                print(f"Unknown topic. {topic_name}")
        except:
            traceback.print_exc()

    def on_stream_error(self, error: Exception) -> bool:
        # Handle error.
        return True  # Return True to close stream, False to keep stream open.

    def on_stream_closed(self) -> None:
        # Handle close.
        pass

print("Store starting...")

db_folder = sys.argv[1]

print(db_folder)

topic = "store/command"

file_manager = FileManager(db_folder, period=60)

request = SubscribeToTopicRequest()
request.topic = topic
handler = StreamHandler(file_manager)
operation = ipc_client.new_subscribe_to_topic(handler) 
future = operation.activate(request)
future.result(TIMEOUT)

# Keep the main thread alive, or the process will exit.
while True:
    time.sleep(10)
    
# To stop subscribing, close the operation stream.
operation.close()