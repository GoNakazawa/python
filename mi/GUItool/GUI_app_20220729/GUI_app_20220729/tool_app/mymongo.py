from tool_app import app
from flask_pymongo import PyMongo
from flask_login import UserMixin
from bson.objectid import ObjectId
from collections import defaultdict
import os
import pandas as pd
from tybmilib import myfilename as mfn

###DB作成 (CROUD)
class Experiments():
    def __init__(self):
        self.mongo = PyMongo(app)
        self.collection = self.mongo.db.Experiments
    
    def insert_one(self, document):
        return self.collection.insert_one(document)

    def delete_one(self, filter):
        return self.collection.delete_one(filter)

    def find_one_id(self, id, projection=None, sort=None):
        return self.collection.find_one(projection=projection,filter={'_id':ObjectId(id)},sort=sort)

    def find_all(self, projection=None,filter=None, sort=None):
        return self.collection.find(projection=projection,filter=filter,sort=sort)

    def find_by_user(self, user, projection=None, sort=None):
        return self.collection.find(projection=projection,filter={'userId':user},sort=sort)

    def count_by_user(self, user):
        return self.collection.count_documents(filter={'userId':user})

    def update_data(self, id, data_name, data):
        return self.collection.update_one({'_id':ObjectId(id)}, {'$set':{data_name:data}})

    def update_input_s3_filename(self, id, input_s3_filename):
        return self.update_data(id, "input_s3_filename", input_s3_filename)

    def update_s3uri_original_data(self, id, s3uri_original_data):
        return self.update_data(id, "s3uri_original_data", s3uri_original_data)

    def update_s3uri_master_data(self, id, s3uri_master_data):
        return self.update_data(id, "s3uri_master_data", s3uri_master_data)
    """
    def update_s3_bucket_path(self, id, s3_bucket_path):
        return self.update_data(id, "s3_bucket_path", s3_bucket_path)
    """

    def update_user_s3_bucket(self, id, user_s3_bucket):
        return self.update_data(id, "user_s3_bucket", user_s3_bucket)

    def update_objectives(self, id, objectives):
        return self.update_data(id, "objectives", objectives)

    def update_objectives_message(self, id, objectives_message):
        return self.update_data(id, "objectives_message", objectives_message)

    def update_drop_cols(self, id, drop_cols):
        return self.update_data(id, "drop_cols", drop_cols)

    def update_vis_cols(self, id, vis_cols):
        return self.update_data(id, "vis_cols", vis_cols)

    def update_problemtype(self, id, problemtype):
        return self.update_data(id, "problemtype", problemtype)

    def update_s3_uri_list(self, id, s3_uri_list):
        return self.update_data(id, "s3_uri_list", s3_uri_list)

    def update_model_name(self, id, model_name):
        return self.update_data(id, "model_name", model_name)

    def update_chem_type(self, id, chem_type):
        return self.update_data(id, "chem_type", chem_type)
    
    def update_radius(self, id, radius):
        return self.update_data(id, "radius", radius)

    def update_bit_num(self, id, bit_num):
        return self.update_data(id, "bit_num", bit_num)

    def update_R2(self, id, R2):
        return self.update_data(id, "R2", R2)
    
    def update_MAE(self, id, MAE):
        return self.update_data(id, "MAE", MAE)

    def update_MSE(self, id, MSE):
        return self.update_data(id, "MSE", MSE)
    
    def update_RMSE(self, id, RMSE):
        return self.update_data(id, "RMSE", RMSE)

    def update_Accuracy(self, id, Accuracy):
        return self.update_data(id, "Accuracy", Accuracy)

    def update_Precision(self, id, Precision):
        return self.update_data(id, "Precision", Precision)

    def update_Recall(self, id, Recall):
        return self.update_data(id, "Recall", Recall)

    def update_F_score(self, id, F_score):
        return self.update_data(id, "F_score", F_score)

    def update_vis_method(self, id, vis_method):
        return self.update_data(id, "vis_method", vis_method)

    def update_vis_method_message(self, id, vis_method_message):
        return self.update_data(id, "vis_method_message", vis_method_message)

    def update_search_method(self, id, search_method):
        return self.update_data(id, "search_method", search_method)
    
    def update_status(self, id, system_status):
        return self.update_data(id, "system_status", system_status)
        
    def update_progress_rate_step1(self, id, progress_rate_step1):
        return self.update_data(id, "progress_rate_step1", progress_rate_step1)
    
    def update_progress_rate_step2(self, id, progress_rate_step2):
        return self.update_data(id, "progress_rate_step2", progress_rate_step2)
        
    def update_progress_rate_step3(self, id, progress_rate_step3):
        return self.update_data(id, "progress_rate_step3", progress_rate_step3)
    
    def update_range_condition(self, id, range_condition):
        return self.update_data(id, "range", range_condition)

    def update_fixed_condition(self, id, fixed_condition):
        return self.update_data(id, "fixed", fixed_condition)

    def update_total_condition(self, id, total_condition):
        return self.update_data(id, "total", total_condition)

    def update_combination_condition(self, id, combination_condition):
        return self.update_data(id, "combination", combination_condition)

    def update_ratio_condition(self, id, ratio_condition):
        return self.update_data(id, "ratio", ratio_condition)

    def update_groupsum_condition(self, id, groupsum_condition):
        return self.update_data(id, "groupsum", groupsum_condition)

    def update_groupsum_total_condition(self, id, groupsum_total_condition):
        return self.update_data(id, "groupsum_total", groupsum_total_condition)


    def update_range_message(self, id, range_message):
        return self.update_data(id, "range_message", range_message)

    def update_fixed_message(self, id, fixed_message):
        return self.update_data(id, "fixed_message", fixed_message)

    def update_total_message(self, id, total_message):
        return self.update_data(id, "total_message", total_message)

    def update_combination_message(self, id, combination_message):
        return self.update_data(id, "combination_message", combination_message)

    def update_ratio_message(self, id, ratio_message):
        return self.update_data(id, "ratio_message", ratio_message)

    def update_groupsum_message(self, id, groupsum_message):
        return self.update_data(id, "groupsum_message", groupsum_message)

    def update_target_message(self, id, target_message):
        return self.update_data(id, "target_message", target_message)

    def update_boundary_setting(self, id, boundary_setting):
        return self.update_data(id, "boundary_setting", boundary_setting)

    def update_error_message(self, id, error_message):
        return self.update_data(id, "error_message", error_message)
    def update_error_message_en(self, id, error_message_en):
        return self.update_data(id, "error_message_en", error_message_en)

    def update_check_finished(self, id, finished):
        return self.update_data(id, "finished", finished)

    def update_check_bitnum(self, id, check_bitnum):
        return self.update_data(id, "check_bitnum", check_bitnum)

    def update_status_step1(self, id, status_step1):
        return self.collection.update_one({"_id":ObjectId(id)}, {"$set":{"status_step1":status_step1}})

    def update_status_step2(self, id, status_step2):
        return self.collection.update_one({"_id":ObjectId(id)}, {"$set":{"status_step2":status_step2}})

    def update_status_step3(self, id, status_step3):
        return self.collection.update_one({"_id":ObjectId(id)}, {"$set":{"status_step3":status_step3}})

    def update_chem_list(self, id, chem_list):
        return self.update_data(id, "chem_list", chem_list)

    def update_master_dict(self, id, master_dict):
        return self.update_data(id, "master_dict", master_dict)

    def update_source_names(self, id, source_names):
        return self.update_data(id, "source_names", source_names)

    def update_rank_dict(self, id, rank_dict):
        return self.update_data(id, "rank_dict", rank_dict)

    def update_bit_dict(self, id, bit_dict):
        return self.update_data(id, "bit_dict", bit_dict)

    def update_objective_dict(self, id, objective_dict):
        return self.update_data(id, "objective_dict", objective_dict)

    def update_true_name_dict(self, id, true_name_dict):
        return self.update_data(id, "true_name_dict", true_name_dict)

    def update_vis_count(self, id, vis_count):
        return self.update_data(id, "vis_count", vis_count)

    def update_model_count(self, id, model_count):
        return self.update_data(id, "model_count", model_count)

    def update_infer_count(self, id, infer_count):
        return self.update_data(id, "infer_count", infer_count)

    def update_pow_check_cols(self, id, pow_check_cols):
        return self.update_data(id, "pow_check_cols", pow_check_cols)

    def agg_obj_count(self, user, obj):
        count_obj = "$" + obj
        pipe=[{"$match": {"userId": {"$eq": user}}}, {"$group": {"_id": "$userId", "total": {"$sum": count_obj}}}]
        agg = self.collection.aggregate(pipeline=pipe)
        user_total = 0
        for usr in agg:
            user_total = usr["total"]
        return user_total

class Users():
    def __init__(self):
        self.mongo = PyMongo(app)
        self.collection = self.mongo.db.Users

    def delete_one(self, filter):
        return self.collection.delete_one(filter)

    def insert_one(self, document):
        return self.collection.insert_one(document)

    def get_all_users(self):
        result = self.collection.find()
        user_list = []
        for res in result:
            user_list.append(res["userId"])
        return user_list

    def get_all_departments(self):
        user_list = self.get_all_users()
        dept_list = []
        for user_id in user_list:
            user = self.find_one_userid(user_id)
            dept_list.append(user["department"])
        return sorted(set(dept_list))

    def write_departments(self):
        dept_list = self.get_all_departments()
        dept_df = pd.DataFrame(columns=["department"])
        dept_df["department"] = dept_list
        dept_filename = mfn.get_department_filename()
        dept_df.to_csv(dept_filename, index=False)

    def write_users(self):
        usr_list = self.get_all_users()
        user_df = pd.DataFrame(columns=["userId", "mail_address", "department", "admin", "passWd"])
        user_df["userId"] = usr_list
        for i, user_id in enumerate(usr_list):
            user = self.find_one_userid(user_id)
            user_df["mail_address"][i] = user["mail_address"]
            user_df["department"][i] = user["department"]
            user_df["admin"][i] = user["admin"]
            user_df["passWd"][i] = user["passWd"]
        user_df.to_csv(mfn.get_users_filename(), index=False)

    def find_one_userid(self, userid, projection=None, sort=None):
        return self.collection.find_one(projection=projection,filter={'userId':userid},sort=sort)

    def update_data(self, userid, data_name, data):
        return self.collection.update_one({'userId':userid}, {'$set':{data_name:data}})

    def update_admin(self, userid, admin):
        return self.update_data(userid, "admin", admin)

    def update_dept(self, userid, dept):
        return self.update_data(userid, "department", dept)

    def update_passwd(self, userid, passwd):
        return self.update_data(userid, "passWd", passwd)

    def update_passwd_change(self, userid, passwd_change):
        return self.update_data(userid, "passWd_change", passwd_change)

    def update_page_count(self, userid, page_count):
        return self.update_data(userid, "page_count", page_count)

    def update_vis_count(self, userid, vis_count):
        return self.update_data(userid, "vis_count", vis_count)

    def update_model_count(self, userid, model_count):
        return self.update_data(userid, "model_count", model_count)

    def update_infer_count(self, userid, infer_count):
        return self.update_data(userid, "infer_count", infer_count)

    def update_sagemaker_time(self, userid, sagemaker_time):
        return self.update_data(userid, "sagemaker_time", sagemaker_time)



def mongo_users():
    class User(UserMixin):
        def __init__(self, id, name, password):
            self.id = id
            self.name = name
            self.password = password

    # mongodb
    mongo = PyMongo(app)
    db_user = mongo.db.Users

    # ユーザーリスト作成
    user_table = db_user.find()
    users = {}
    i = 1
    for user in user_table:
        users[i] = User(i, user['userId'], user['passWd'])
        i += 1

    return users

def mongo_users_check(users):
    # ユーザーチェックに使用する辞書作成
    nested_dict = lambda: defaultdict(nested_dict)
    user_check = nested_dict()
    for i in users.values():
        user_check[i.name]["password"] = i.password
        user_check[i.name]["id"] = i.id
    
    return user_check

def init_dict(uid, name, time):
    init_dict = {
        "userId":uid,
        "title":name,
        "update":time,
        "error_message":"",
        "error_message_en":"",
        "input_s3_filename": "",
        "s3uri_original_data":"None",
        "s3uri_master_data":"None",
        #"s3_bucket_path":"なし",
        "user_s3_bucket":"",
        "objectives": "objectives",
        "objectives_message": "なし",
        "drop_cols": "なし",
        "vis_cols": "なし",
        "problemtype": "なし",
        "s3_uri_list":"None",
        "model_name":{},
        "R2":"#",
        "MAE":"#",
        "MSE":"#",
        "RMSE":"#",
        "Accuracy":"#",
        "Precision":"#",
        "Recall":"#",
        "F_score":"#",
        "search_method":[],
        "vis_method":[],
        "vis_method_message":"なし",
        "system_status":"#",
        "range_message": "なし",
        "fixed_message": "なし",
        "total_message": "なし",
        "combination_message": "なし",
        "ratio_message": "なし",
        "groupsum_message": "なし",
        "target_message": "なし",
        "finished": "NOT",
        "chem_type": "not",
        "radius": "0",
        "bit_num": "4096",
        "status_step1": "wait",
        "status_step2": "wait",
        "status_step3": "wait",
        "progress_rate_step1": "0",
        "progress_rate_step2": "0",
        "progress_rate_step3": "0",
        "check_bitnum": "satisfied",
        "master_dict": "",
        "source_names": "",
        "rank_dict": "",
        "bit_dict": "",
        "objective_dict": "",
        "vis_count": 0,
        "model_count:": 0,
        "infer_count:": 0,
        "pow_check_cols": [],
    }
    return init_dict