from pymongo import MongoClient

class Experiment:
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client['MI_tool'] #DB名を設定
        self.collection = self.db.get_collection('Experiments')
    
    def insert():
        print('test')
        return 'ok'

    def find_one(self, projection=None,filter=None, sort=None):
        return self.collection.find_one(projection=projection,filter=filter,sort=sort)
