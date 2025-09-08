import firebase_admin
from firebase_admin import credentials, firestore
from typing import List, Dict, Any

from google.cloud.firestore_v1 import Client


class TransferDB:

    def __init__(self, transfer_db_service_account_path, recieving_db_service_account_path):
        self.db: Client = None
        self.transfer_path = transfer_db_service_account_path
        self.recieving_path = recieving_db_service_account_path

    def _initialize_transfer(self):
        self.transfer_cred = credentials.Certificate(self.transfer_path)
        firebase_admin.initialize_app(self.transfer_cred, name="undefined")
        self.db = firestore.client(firebase_admin.get_app("undefined"))

    def _initialize_reciveing(self):
        self.recieve_cred = credentials.Certificate(self.recieving_path)
        firebase_admin.initialize_app(self.recieve_cred, name="Engineering Hub")
        self.db = firestore.client(firebase_admin.get_app("Engineering Hub"))

    def get_collection_data(self, collection_path):
        self._initialize_transfer()
        data = self.db.collection(collection_path).get()
        return [i.to_dict() for i in data]

    def write_collection_data(self, collection_path, data: List[Dict[str, Any]]):
        self._initialize_reciveing()
        num = 1
        for i in data:
            print(num)
            i["code"]
            self.db.collection(collection_path).document(i["code"]).set(
                i
            )
            num+=1







from COURSEGEN.config import load_config

config = load_config()

transfer_service_account_path = config.transfer_service_account_path
reciving_service_account_path = config.reciving_service_account_path

transferDB = TransferDB(transfer_db_service_account_path=transfer_service_account_path, recieving_db_service_account_path=reciving_service_account_path)
data = transferDB.get_collection_data("course_data")
print(len(data))

transferDB.write_collection_data("course_data", data)
