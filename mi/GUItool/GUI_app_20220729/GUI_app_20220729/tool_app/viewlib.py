from tool_app import app
from flask import Flask, render_template, request, jsonify, make_response, Markup, flash, redirect, url_for, session, send_file, send_from_directory, Response
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin

import pandas as pd
import os
import numpy as np
import matplotlib
import japanize_matplotlib
import datetime
from datetime import datetime
import shutil
import time
import pathlib
from bson.objectid import ObjectId
from datetime import timedelta
import boto3
import sagemaker
from bs4 import BeautifulSoup
import itertools
import threading
import json
#from tool_app.views import experiment
from tybmilib import prep
from tybmilib import vis
from tybmilib import datamgmt
from tybmilib import modeling
from tybmilib import paramsearch
from tybmilib import chemembeding
from tybmilib import logmgmt
from tybmilib import myfilename as mfn
from tybmilib.sampling import Sampling, Inference
from tool_app.mymongo import Experiments, Users

class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするたためのクラス
    """
    pass

# 拡張子の判定
def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(["csv", "xlsx"])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 不要列設定
def set_drop_cols(exp_id, df_s3, drop_cols): 
    try:
        if drop_cols != [""]:
            df_reg = prep.drop_cols(df_s3, drop_cols, exp_id)
        else:
            df_reg = df_s3.copy()
        return df_reg
    except Exception as error_msg:
        raise Lib_ParseError(error_msg)


# "/"の置き換え
def replace_slash_in_target(objectives):
    target_list = []
    for obj in objectives:
        if '/' in obj:
            obj = obj.replace('/','')
        target_list.append(obj)
    return target_list

# エラーメッセージを更新
def update_error(exp_id, error_msg, step):
    ex = Experiments()
    ex.update_error_message(id=exp_id, error_message=error_msg)
    step_all_log = mfn.get_step_all_log_filename(exp_id)
    logmgmt.logError(exp_id, error_msg, step_all_log)

    if step == "step1":
        ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=0)
    if step == "step2":
        ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=0)
    if step == "step3":
        ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=0)


def update_progress_status(exp_id, step1_num=0, step2_num=0, step3_num=0, step1_sts="wait", step2_sts="wait", step3_sts="wait"):
    ex = Experiments()
    ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=step1_num)
    ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=step2_num)
    ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=step3_num)
    ex.update_status_step1(id=exp_id, status_step1=step1_sts)
    ex.update_status_step2(id=exp_id, status_step2=step2_sts)
    ex.update_status_step3(id=exp_id, status_step3=step3_sts)    

def delete_local_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def get_pow_check_cols(df_s3):
    pow_check_cols = []
    for col in df_s3.columns:
        for_check = df_s3[col]
        # 文字列の列を除外して実行
        if pd.to_numeric(for_check, errors="coerce").notna().all():
            for_check_2 = for_check[for_check < pow(10, -5)]
            if len(for_check_2[for_check_2 > 0]) > 0:
                pow_check_cols.append(col)
    return pow_check_cols

def get_model_list(exp_id, objectives):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    try:
        model_name_dict = experiment['model_name']
        model_list = [model_name_dict[obj] for obj in objectives]
    except:
        model_list = []
    return model_list

# step1の結果を削除
def remove_step1_localfiles(exp_id):
    ex = Experiments()
    local_files = [mfn.get_scatter_filename(exp_id, "objectives"), mfn.get_scatter_filename(exp_id, "all"), mfn.get_correlation_filename(exp_id, "objectives"), mfn.get_correlation_filename(exp_id, "all"), mfn.get_profile_filename(exp_id)]
    for f in local_files:
        if os.path.isfile(f):
            os.remove(f)
    ex.update_vis_method(id=exp_id, vis_method=[])
    ex.update_vis_cols(id=exp_id, vis_cols="なし")
    ex.update_vis_method_message(id=exp_id, vis_method_message="なし")

def remove_chemical_files(exp_id):
    list_structure_path = mfn.get_list_structure_path(exp_id)
    if os.path.exists(list_structure_path):
        shutil.rmtree(list_structure_path)
    if not os.path.exists(list_structure_path):
        os.makedirs(list_structure_path)

# step2の結果を削除
def remove_step2_localfiles(exp_id, objectives):
    ex = Experiments()
    target_list = replace_slash_in_target(objectives)

    # remove local files
    delete_local_file(mfn.get_shap_filename(exp_id))
    for target_name in target_list:
        delete_local_file(mfn.get_coefficients_all_filename(exp_id, target_name))
        delete_local_file(mfn.get_coefficients_limit_filename(exp_id, target_name))
        delete_local_file(mfn.get_test_filename(exp_id, target_name))
        delete_local_file(mfn.get_confusion_filename(exp_id, target_name))

    # モデルの削除
    model_list = get_model_list(exp_id, objectives)
    prep.delete_model(model_list)
    model_dict = {}
    for obj in objectives:
        model_dict[obj] = "#"
    ex.update_model_name(id=exp_id,model_name=model_dict)
    ex.update_problemtype(id=exp_id, problemtype="なし")

    ex.update_R2(id=exp_id, R2="#")
    ex.update_MAE(id=exp_id, MAE="#")
    ex.update_MSE(id=exp_id, MSE="#")
    ex.update_RMSE(id=exp_id, RMSE="#")
    ex.update_Accuracy(id=exp_id, Accuracy="#")
    ex.update_Precision(id=exp_id, Precision="#")
    ex.update_Recall(id=exp_id, Recall="#")
    ex.update_F_score(id=exp_id, F_score="#")
    

# step3の結果を削除
def remove_step3_localfiles(exp_id, objectives, method_list):
    if "Simulate" in method_list:
        delete_local_file(mfn.get_simulate_filename(exp_id))
    if "Search_Cluster" in method_list:
        delete_local_file(mfn.get_cluster_img_filename(exp_id))
        delete_local_file(mfn.get_cluster_filename(exp_id))
        delete_local_file(mfn.get_cluster_mean_filename(exp_id))
    if "Search_Pareto" in method_list:
        delete_local_file(mfn.get_pareto_filename(exp_id))        
        target_list = replace_slash_in_target(objectives)
        if len(objectives) >= 3:
            all_trio = itertools.permutations([j for j in range(len(objectives))], 3)
            for j in all_trio:
                delete_local_file(mfn.get_pareto_img_filename(exp_id, str(target_list[j[0]]), str(target_list[j[1]]), str(target_list[j[2]])))
        elif len(objectives) == 2:
            all_pair = itertools.permutations([j for j in range(len(objectives))], 2)
            for j in all_pair:
                delete_local_file(mfn.get_pareto_img_filename(exp_id, str(target_list[j[0]]), str(target_list[j[1]])))
    ex = Experiments()
    ex.update_search_method(id=exp_id, search_method="")

# 化学構造のDataframe作成
def chemical_prepare(df_reg, objectives, exp_id, structure_mode, radius=0, bit_num=4096):
    try:
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)

        master_filename = experiment["s3uri_master_data"]
        df_columns = df_reg.columns
        chem = chemembeding.Features(master_filename, df_columns, exp_id, structure_mode, radius, bit_num)
        mol_list, true_name_dict = chem.get_smiles()
        ex.update_true_name_dict(id=exp_id, true_name_dict=true_name_dict)

        check_duplicate = False
        if structure_mode=="mfp":
            check_duplicate = chem.preview_mfp()
        elif structure_mode=="maccs":
            chem.preview_maccs()
        
        if check_duplicate == True:
            ex.update_check_bitnum(id=exp_id, check_bitnum="short")
            df_chem = df_reg
        else:
            df_chem = chem.generate_fingerprint_dataset(df_reg, objectives)
        return df_chem
    except:
        error_msg = "Error  : 学習データ作成に失敗しました"
        update_error(exp_id, error_msg, "step1")        

# Step1: 入力データとCASコードの読み込み
def read_s3_data(s3_path, exp_id, s3_passwd):
    step_all_log = mfn.get_step_all_log_filename(exp_id)
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    user_id = experiment["userId"]
    usr = Users()
    user = usr.find_one_userid(user_id)
    dept = user["department"]
    s3_path_split = s3_path.split("/")

    if user_id == s3_path_split[3] and mfn.get_user_s3_bucket(dept) == s3_path_split[2]:
        try:
            df_data = prep.read_s3_bucket_data(s3_path, exp_id, s3_passwd)
        except Exception as error_msg:
            raise Lib_ParseError(error_msg)

        if len(df_data.index) == 0 or len(df_data) == 0:
            error_msg = 'Error  : 入力データが空です。入力データを再設定してください'
            logmgmt.raiseError(exp_id, error_msg, step_all_log)

        df_col = df_data.columns.tolist()
        if len([s for s in df_col if 'Unnamed' in s]) != 0:
            error_msg = 'Error  : 入力データ内のカラムに空白が含まれています。入力データを再設定してください'
            logmgmt.raiseError(exp_id, error_msg, step_all_log)

        return df_data
    else:
        error_msg = 'Error  : 入力したS3パスが正確ではありません'
        logmgmt.raiseError(exp_id, error_msg, step_all_log)



# Step1: データチェック
def check_input_data(exp_id, df, objectives):
    step_all_log = mfn.get_step_all_log_filename(exp_id)

    # 文字列行
    target_columns = df.columns.tolist()
    str_columns = []
    for col in target_columns:
        pic = df[[col]][df[col].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
        if len(pic) > 0:
            pic = pd.DataFrame(pic)
            if pic.isnull().values.sum() != pic.size:
                str_columns.append(col)
    if len(str_columns) > 0:
        error_msg = 'Error  : 変数{}に文字列が含まれています。探索実行できないため、対象ファイルのデータを修正してください'.format(", ".join(str_columns))
        logmgmt.raiseError(exp_id, error_msg, step_all_log)
        
    # null行
    null_array = []
    for col_name, col_series in df.iteritems():
        if col_series.isnull().any():
            if not col_name in objectives:
                null_array.append(col_name)
    if len(null_array) > 0:
        error_msg = 'Error  : 変数{}にNUllが含まれています。探索実行できないため、対象ファイルのデータを修正してください'.format(", ".join(null_array))
        logmgmt.raiseError(exp_id, error_msg, step_all_log)

    # 目的変数check
    for j in objectives:
        if not j in df.columns.tolist():
            error_msg = 'Error  : 目的変数{}の名称に誤りがあります'.format(str(j))
            logmgmt.raiseError(exp_id, error_msg, step_all_log)


# Step1: 学習データ作成
def create_train_data(exp_id, df_reg, objectives, bucket_name, radius=0, bit_num=4096):
    try:
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        user_id = experiment["userId"]
        s3_prefix = mfn.get_s3_csv_path(user_id, exp_id)

        # 学習データ作成
        structure_mode = experiment["chem_type"]
        if structure_mode == "maccs":
            df_reg = chemical_prepare(df_reg, objectives, exp_id, structure_mode)
        elif structure_mode == "mfp":
            df_reg = chemical_prepare(df_reg, objectives, exp_id, structure_mode, radius=radius, bit_num=bit_num)
        # 目的変数分の学習データを作成し、S3のSageMaker用ディレクトリに格納
        traindata_path_list = prep.create_multi_traindata(df_reg,exp_id,objectives)
        # upload data
        s3_uri_list = []
        for traindata_path in traindata_path_list:
            s3_uri = datamgmt.upload_file(bucket_name, traindata_path, s3_prefix)
            s3_uri_list.append(s3_uri)
        return s3_uri_list
    except Exception as error_msg:
        raise Lib_ParseError(error_msg)


def get_master_dict(exp_id, s3uri_master_data, master_password):
    try:
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        master_filename = mfn.get_master_filename(exp_id)
        df_master = read_s3_data(s3uri_master_data, exp_id, master_password)
        df_master.to_csv(master_filename, index=False)
    except Exception as e:
        logmgmt.raiseError(exp_id, str(e), step_all_log)

    try:
        master_dict = {}
        for i,sn in enumerate(df_master["Source_Name"]):
            master_dict[sn] = df_master["CAS"][i]

        return master_dict
    except:
        error_msg = "Error : CASデータの読み込みに失敗しました。データ形式を見直してください"
        logmgmt.raiseError(exp_id, error_msg, step_all_log)


def search_source_names(exp_id, source_names, objectives):
    rank_dict = {}
    for source_name in source_names:
        each_folder = mfn.get_each_structure_path(exp_id)
        rank_dict[source_name] = {}
        
        for target in objectives:
            shapresult_filename = mfn.get_shapresult_target_filename(exp_id, target)
            json_open = open(shapresult_filename, 'r')
            json_load = json.load(json_open)
            json_data = json_load["explanations"]["kernel_shap"]["label0"]["global_shap_values"]
            json_sorted = sorted(json_data.items(), key=lambda x:x[1], reverse=True)
            new_json_sorted = []
            for i, tup in enumerate(json_sorted):
                key_num_split = tup[0].split("_")
                if len(key_num_split) > 1 and (key_num_split[0] == "bit" or key_num_split[0] == "MACCS"):
                    try:
                        key_num = int(key_num_split[1])
                        new_json_sorted.append((key_num, i+1))
                    except:
                        continue
            source_img_list = list(pathlib.Path(each_folder).glob('{}*.png'.format(source_name)))

            for source_img in source_img_list:
                key_num = source_img.stem.split("_")[-1]
                for tup in new_json_sorted:
                    if tup[0] == int(key_num):
                        if target == objectives[0]:
                            rank_dict[source_name][str(tup[0])] = []
                        rank_dict[source_name][str(tup[0])].append(str(tup[1]))
    return rank_dict
