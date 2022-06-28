import json
import os
import datetime
import shutil
from threading import Timer

class FileManager():
    period = 60
    data_obj = {}
    inspection_id = "no_id"
    db_folder = "."

    def __init__(self, db_folder, inspection_id="no_id", period=60):
        super().__init__()
        self.period = period
        self.inspection_id = inspection_id
        self.db_folder = db_folder
        self.data_obj = {"payload" : [], "timestamp": 0}
        self.timer_callback()
        print(f"file ops init {self.period}")

    # Put the payload to the memory
    def put(self, payload):
        print(f"put : {payload}")
        self.data_obj["payload"].append(payload)

    # Save the file to the file system in inspection_id/yyyy/MM/dd partition format
    def save(self):
        self.data_obj["timestamp"] = datetime.datetime.timestamp(datetime.datetime.now())
        json_str = json.dumps(self.data_obj)
        date_partition = datetime.datetime.now().strftime('%G/%m/%d')
        folder = f"{self.db_folder}/{self.inspection_id}/{date_partition}"
        fname_ts = str(datetime.datetime.timestamp(datetime.datetime.now())).split('.')[0]
        os.makedirs(folder, exist_ok=True)
        print(f"save : {folder}/{fname_ts}")
        f = open(f"{folder}/{fname_ts}", "w")
        f.write(json_str)
        f.close()
        self.data_obj["payload"] = []

    # Delete specific inspection from the file system
    def delete_inspection(self, inspection_id):
        folder = f"{self.db_folder}/{inspection_id}"
        shutil.rmtree(folder)

    # Creates a timer to save the file periodically
    def timer_callback(self):
        print(f"tick {self.period}: {self.data_obj}")
        if "payload" in self.data_obj:
            if len(self.data_obj["payload"]) > 0:
                self.save()
        t = Timer(self.period, self.timer_callback)
        t.start()

    def set_inspection_id(self, id):
        self.inspection_id = id

    def set_timer_period(self, period):
        self.period = period

    