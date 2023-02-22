from tool_app import app

from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, session, Response, send_from_directory, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user


import pandas as pd
import os
import numpy as np
import matplotlib
import japanize_matplotlib
import gevent
import datetime
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import shutil
import time
import pathlib
from bson.objectid import ObjectId
import boto3
import sagemaker
from bs4 import BeautifulSoup
import itertools
import json
import secrets
import string
import re
import math
from tool_app.config import SECRET_KEY
from werkzeug.utils import secure_filename
from tybmilib import prep
from tybmilib import vis
from tybmilib import modeling
from tybmilib import datamgmt
from tybmilib import paramsearch
from tybmilib import chemembeding
from tybmilib import myfilename as mfn
from tool_app.mymongo import Experiments, Users, mongo_users, mongo_users_check, init_dict
from tool_app import viewlib as vlib
from tybmilib import logmgmt
from tybmilib.sampling import Sampling, Inference
from tool_app import admin_users as adm
from tool_app import view_thread as vthd
from tool_app.set_sampling import SetCondition


#--------------------------------------------------------------------
#ss = sagemaker.Session(boto3.session.Session(region_name='ap-southeast-2'))
# session time
app.permanent_session_lifetime = timedelta(minutes=60)


def cached(timeout=5 * 60, key='view/{}'):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = key.format(request.path)
            rv = cache.get(cache_key)
            if rv is not None:
                return rv
            rv = f(*args, **kwargs)
            cache.set(cache_key, rv, timeout=timeout)
            return rv
        return decorated_function
    return decorator


# ログイン処理
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    users = mongo_users()
    return users.get(int(user_id))

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=120)
    session.modified = True

# ログイン画面設定
@app.route('/', methods=["GET", "POST"])
def login():

    if(request.method == "POST"):
        # ユーザーチェック
        users = mongo_users()
        user_check = mongo_users_check(users)
        if request.form["username"] in user_check:
            if request.form["password"] == user_check[request.form["username"]]["password"]:
                # ユーザーが存在した場合はログイン
                session_username = request.form.get("username")
                usr = Users()
                user = usr.find_one_userid(session_username)
                change_num = user["passWd_change"]
                if change_num > 0:
                    login_user(users.get(user_check[request.form["username"]]["id"]))
                    session["username"] = session_username
                    return redirect(url_for("index", user_id = request.form["username"]))
                else:
                    # ユーザーがパスワード変更を行っていない場合はログインページに戻される
                    flash("Error: 初期パスワードから変更されていません。メールに記載のリンクからパスワード変更を行ってください", "failed")
                    return render_template("login.html")
            else:
                # パスワードが間違っている場合はログインページに留まる
                flash("Error: パスワードが間違っています", "failed")
                return render_template("login.html")
        else:
            # ユーザーが存在しない場合はログインページに戻される
            flash("Error: ユーザーIDが存在しません", "failed")
            return render_template("login.html")
    else:
        return render_template("login.html")

# Log Out
@app.route('/logout/',methods=['GET', 'POST'])
@login_required
def logout():
    if(request.method == "POST"):
        logout_user()
    return redirect('/')



# 実験一覧表示
@app.route('/index/<user_id>')
@login_required
def index(user_id):
    ex = Experiments()
    cursor = ex.find_by_user(user_id)

    usr = Users()
    user = usr.find_one_userid(user_id)
    user_admin = user["admin"]
    user_department = user["department"]
    input_s3_bucket = mfn.get_user_s3_bucket(user_department)
    datamgmt.create_user_bucket(input_s3_bucket)
    usr.update_page_count(userid=user_id, page_count=ex.count_by_user(user_id))

    users = mongo_users()
    #for i in range(1, 10):
    #    print(users[i].is_authenticated)
    return render_template('index.html', exps=cursor, user_id=user_id, user_admin=user_admin)

# 新規実験ページの作成
@app.route('/add_exp', methods=['GET', 'POST'])
@login_required
def add_exp():
    user_id = request.form['user_id']
    name = request.form['name']
    time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')    
    init_db = init_dict(user_id, name, time)

    ex = Experiments()
    result = ex.insert_one(init_db)
    exp_id = str(result.inserted_id)

    usr = Users()
    user = usr.find_one_userid(user_id)
    user_department = user["department"]
    input_s3_bucket = mfn.get_user_s3_bucket(user_department)
    ex.update_user_s3_bucket(id=exp_id, user_s3_bucket=input_s3_bucket)    

    mfn.prepare_folder(exp_id)
    return redirect(url_for('experiment',exp_id=exp_id))

# 実験の詳細
@app.route('/detail_exp/<exp_id>')
@login_required
def detail_exp(exp_id):
    return redirect(url_for('experiment',exp_id=exp_id))

# 実験の削除
@app.route('/delete_exp/<user_id>/<exp_id>')
@login_required
def delete_exp(user_id, exp_id):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    objectives = experiment['objectives'].split(',')
    
    # delete local files
    local_path = mfn.get_expid_path(exp_id)
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    # delete s3 files
    user_s3_bucket = experiment['user_s3_bucket']
    input_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
    prep.delete_s3_items(user_s3_bucket, input_s3_prefix)
    model_s3_bucket = mfn.get_modeling_s3_bucket()
    prep.delete_s3_items(model_s3_bucket, input_s3_prefix)
    try:
        session = sagemaker.Session(boto3.session.Session())
        bucket = session.default_bucket()
        prefix = experiment["s3_bucket_path"]
        prep.delete_s3_items(bucket, prefix)
    except Exception as e:
        print(str(e))

    # delete model
    model_list = vlib.get_model_list(exp_id, objectives)
    prep.delete_model(model_list)

    # delete db record
    ex.delete_one({"_id": ObjectId(exp_id)})
    flash("Status: 実験ページを削除しました", "success")
    return redirect(url_for('index',user_id=user_id))


# 実験一覧ページ
@app.route('/experiment/<exp_id>', methods=['GET', 'POST'])
@login_required
def experiment(exp_id):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    user_id = experiment['userId']
    name = experiment['title']

    def try_setting(exp_id, str_arg, excp="なし"):
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        try:
            para = experiment[str_arg]
        except:
            para = excp
        return para

    def read_df_csv(filename):
        if os.path.exists(filename):
            df_data = pd.read_csv(filename)
            if len(df_data) > 100:
                df_data = df_data.head(100)
            for i in list(df_data.columns):
                df_data[i].astype(dtype='float64',errors='ignore')
        else:
            default_csv_name = mfn.get_defaultcsv_filename()
            df_data = pd.read_csv(default_csv_name)
        return df_data
    
    def get_df_info(df_data):
        df_data_columns = df_data.columns.tolist()
        df_data_index = df_data.index.tolist()
        df_data_values_num = df_data.values.tolist()
        df_data_values= []
        for i in range(len(df_data_index)):
            df_data_values.append(['{:.2f}'.format(n) if type(n) != str else n  for n in df_data_values_num[i]])
        return df_data_columns, df_data_values
    
    # s3ファイルパス操作
    usr = Users()
    user = usr.find_one_userid(user_id)
    user_department = user["department"]
    input_s3_bucket = mfn.get_user_s3_bucket(user_department)
    ex.update_user_s3_bucket(id=exp_id, user_s3_bucket=input_s3_bucket)
    input_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
    input_s3_folder = "s3://" + input_s3_bucket + "/" + input_s3_prefix
    input_s3_file_list, time_array = datamgmt.get_s3_file_list(user_department, user_id, exp_id)
    input_s3_filename = try_setting(exp_id, "input_s3_filename", excp="")

    drop_cols = str(try_setting(exp_id, "drop_cols"))
    s3uri_original_data = try_setting(exp_id, "s3uri_original_data")
    s3uri_master_data = try_setting(exp_id, "s3uri_master_data")

    # 分析データのプレビュー
    df_s3_file_name = mfn.get_preview_filename(exp_id)
    df_s3 = read_df_csv(df_s3_file_name)
    df_s3_columns, df_s3_values = get_df_info(df_s3)

    # データ可視化結果
    scatter_path = mfn.get_scatter_filename(exp_id, "all", relative=True)   
    correlation_path = mfn.get_correlation_filename(exp_id, "all", relative=True)
    profile_path = mfn.get_profile_filename(exp_id, relative=True)
    vis_cols = try_setting(exp_id, "vis_cols", "なし")

    # モデル名、目的変数名
    objectives = experiment['objectives'].split(',')
    model_list = vlib.get_model_list(exp_id, objectives)
    problemtype = experiment['problemtype']

    # 線形モデルの係数
    coef_all_path_list = []
    coef_limit_path_list = []
    target_list =vlib.replace_slash_in_target(objectives)
    for target_name in target_list:
        coefficients_all_filename = mfn.get_coefficients_all_filename(exp_id, target_name, relative=True)
        coefficients_limit_filename = mfn.get_coefficients_limit_filename(exp_id, target_name, relative=True)
        coef_all_path_list.append(coefficients_all_filename)
        coef_limit_path_list.append(coefficients_limit_filename)

    # テストデータでの評価値
    eval_list= ['','','','']
    if problemtype == 'Regression':
        header_eval_list = ['R2','MAE','MSE','RMSE']
        R2 = experiment['R2'].split(',')
        MAE = experiment['MAE'].split(',')
        MSE = experiment['MSE'].split(',')
        RMSE = experiment['RMSE'].split(',')
        eval_list = [R2,MAE,MSE,RMSE]
    else:
        header_eval_list = ['Accuracy','Precision','Recall','F_score']
        Accuracy = experiment["Accuracy"].split(',')
        Precision = experiment["Precision"].split(',')
        Recall = experiment["Recall"].split(',')
        F_score = experiment["F_score"].split(',')
        eval_list = [Accuracy,Precision,Recall,F_score]

    eval_image_list = []
    for target_name in target_list:    
        if problemtype == 'Regression':
            eval_image_filename = mfn.get_test_filename(exp_id, target_name, relative=True)
        else:
            eval_image_filename = mfn.get_confusion_filename(exp_id, target_name, relative=True)
        eval_image_list.append(eval_image_filename)
    # shap report path
    shapreport_path = mfn.get_shap_filename(exp_id, relative=True)
        
    #制約条件のカラム名
    df_train_file_name = mfn.get_samplingx_filename(exp_id)
    df_train = read_df_csv(df_train_file_name)
    df_train_columns = df_train.columns.tolist()

    # 探索手法
    search_method = try_setting(exp_id, "search_method", [])
    vis_method = try_setting(exp_id, "vis_method", [])
    vis_method_message = try_setting(exp_id, "vis_method_message", "なし")

    # Simulate
    df_samples_Simulate_name = mfn.get_simulate_filename(exp_id)
    df_samples_Simulate = read_df_csv(df_samples_Simulate_name)
    df_Simulate_columns, df_Simulate_values = get_df_info(df_samples_Simulate)
    ### cluster
    cluster_path = mfn.get_cluster_img_filename(exp_id, relative=True)
    df_Clustering_name = mfn.get_cluster_filename(exp_id)
    df_Clustering = read_df_csv(df_Clustering_name)
    df_Clustering_columns, df_Clustering_values = get_df_info(df_Clustering)
    df_Clustering_mean_name = mfn.get_cluster_mean_filename(exp_id)
    df_Clustering_mean = read_df_csv(df_Clustering_mean_name)
    df_Clustering_mean_columns, df_Clustering_mean_values = get_df_info(df_Clustering_mean)
    # pareto
    pareto_path_list = []
    if 'Search_Pareto' in search_method:
        # pareto
        pareto_path_list = []
        if len(objectives) >= 3:
            all_trio = itertools.permutations([j for j in range(len(objectives))], 3)
            for j in all_trio:
                file_name = mfn.get_pareto_img_filename(exp_id, str(target_list[j[0]]), str(target_list[j[1]]), target3=str(target_list[j[2]]), relative=True)
                pareto_path_list.append(file_name)
        elif len(objectives) == 2:
            all_pair = itertools.permutations([j for j in range(len(objectives))], 2)
            for j in all_pair:
                file_name = mfn.get_pareto_img_filename(exp_id, str(target_list[j[0]]), str(target_list[j[1]]), relative=True)
                pareto_path_list.append(file_name)

    ### pareto csv
    df_pareto_name = mfn.get_pareto_filename(exp_id)
    df_pareto = read_df_csv(df_pareto_name)
    df_pareto_columns, df_pareto_values = get_df_info(df_pareto)

    ### 分子構造組み込み
    chem_type = try_setting(exp_id, "chem_type", "not")
    check_bitnum = try_setting(exp_id, "check_bitnum", "satisfied")
    chem_list = try_setting(exp_id, "chem_list", excp=[""]*len(objectives))
    master_dict = try_setting(exp_id, "master_dict", "")
    source_names = try_setting(exp_id, "source_names", "")
    
    bit_dict = try_setting(exp_id, "bit_dict")
    objective_dict = try_setting(exp_id, "objective_dict")
    true_name_dict = try_setting(exp_id, "true_name_dict", excp={})

    list_structure_path = mfn.get_list_structure_path(exp_id)
    each_structure_path = mfn.get_each_structure_path(exp_id)
    bit_structure_path = mfn.get_bit_structure_path(exp_id)

    each_structure_path_relative = mfn.get_each_structure_path(exp_id, relative=True)
    rank_link_dict = {}

    for source_name in source_names:
        rank_link_list = []
        for bit in bit_dict[source_name]:
            img_name = os.path.join(each_structure_path_relative, "{0}_{1:04}.png".format(source_name, int(bit)))
            rank_link_list.append(img_name)
        rank_link_dict[source_name] = rank_link_list

    p_list_structure = list(pathlib.Path(list_structure_path).glob('*.png'))
    p_list_structure.sort()
    
    p_source_names = []
    p_cas_numbers = []
    p_source_nums = []
    true_name_list = []
    p_list_structure_links = []

    if chem_type == "mfp":
        for p_name in p_list_structure:
            source_name = p_name.stem
            p_source_names.append(source_name)
            p_cas_numbers.append(master_dict[source_name])
            try:
                true_name_list.append(true_name_dict[source_name])
            except:
                true_name_list = experiment["true_name_list"]
            p_list_name = list(pathlib.Path(each_structure_path).glob('{}_*.png'.format(source_name)))
            p_source_nums.append(len(p_list_name))
            p_list_structure_links.append(os.path.join(mfn.get_list_structure_path(exp_id, relative=True), source_name+".png"))

    top_bit_structure_names_list = []
    top_bit_display_names_list = []
    top_bit_structure_links_list = []    
    if chem_type=="maccs" or chem_type=="mfp":
        for i, target in enumerate(objectives):
            top_bit_structure = list(pathlib.Path(bit_structure_path).glob('shap_result_{}_*.png'.format(target)))
            top_bit_structure.sort()
            top_bit_structure_names = chem_list[i]
            
            top_bit_display_names = []
            for top_names in top_bit_structure_names:
                top_names_list = top_names.split(",")
                if len(top_names_list) > 5:
                    top_bit_display_names.append(",".join(top_names_list[0:5]) + "...")
                else:
                    top_bit_display_names.append(top_names)
            top_bit_structure_names_list.append(top_bit_structure_names)
            top_bit_display_names_list.append(top_bit_display_names)

            top_bit_structure_links = []
            for top_name in top_bit_structure:
                top_bit_structure_links.append(os.path.join(mfn.get_bit_structure_path(exp_id, relative=True), str(top_name.name)))
            top_bit_structure_links_list.append(top_bit_structure_links)

    #Messages
    objectives_message = try_setting(exp_id, "objectives_message")
    range_message = try_setting(exp_id, "range_message")
    fixed_message = try_setting(exp_id, "fixed_message")
    total_message = try_setting(exp_id, "total_message")
    combination_message = try_setting(exp_id, "combination_message")
    ratio_message = try_setting(exp_id, "ratio_message")
    groupsum_message = try_setting(exp_id, "groupsum_message")
    target_message = try_setting(exp_id, "target_message")
    finished = try_setting(exp_id, "finished", "NOT")    
    error_message = try_setting(exp_id, "error_message", "")
    error_message_en = try_setting(exp_id, "error_message_en", "")
    status_step1 = try_setting(exp_id, "status_step1", "wait")
    status_step2 = try_setting(exp_id, "status_step2", "wait")
    status_step3 = try_setting(exp_id, "status_step3", "wait")
    pow_check_cols = try_setting(exp_id, "pow_check_cols", [])

    browser_filename = 'experiment.html'
    return render_template(browser_filename, exp_id=exp_id, user_id=user_id, name=name, s3_file_list=input_s3_file_list, s3_folder=input_s3_folder, time_array=time_array, \
                            input_s3_filename=input_s3_filename, user_s3_prefix=input_s3_prefix, drop_cols=drop_cols,\
                            s3uri_original_data=s3uri_original_data, s3uri_master_data=s3uri_master_data, vis_cols=vis_cols,\
                            finished=finished, pow_check_cols=pow_check_cols,\
                            error_message=error_message, error_message_en=error_message_en,\
                            df_s3_columns=df_s3_columns, df_s3_values=df_s3_values,\
                            df_train_columns=df_train_columns, objectives=objectives, objectives_message=objectives_message,\
                            vis_method=vis_method, vis_method_message=vis_method_message,\
                            scatter_path=scatter_path, correlation_path=correlation_path, profile_path=profile_path,\
                            problemtype=problemtype, model_list=model_list, shapreport_path=shapreport_path,\
                            coef_all_path_list=coef_all_path_list, coef_limit_path_list=coef_limit_path_list, \
                            header_eval_list=header_eval_list, eval_list=eval_list, eval_image_list=eval_image_list,\
                            range_message=range_message, fixed_message=fixed_message,total_message=total_message,\
                            combination_message=combination_message, ratio_message=ratio_message, target_message=target_message, groupsum_message=groupsum_message,\
                            search_method=search_method,\
                            df_Simulate_values=df_Simulate_values, df_Simulate_columns=df_Simulate_columns,\
                            cluster_path=cluster_path, df_Clustering_values=df_Clustering_values, df_Clustering_columns=df_Clustering_columns,\
                            df_Clustering_mean_values=df_Clustering_mean_values, df_Clustering_mean_columns=df_Clustering_mean_columns,\
                            pareto_path_list=pareto_path_list, df_pareto_values=df_pareto_values, df_pareto_columns=df_pareto_columns,\
                            source_names=source_names, bit_dict=bit_dict, objective_dict=objective_dict, rank_link_dict=rank_link_dict, true_name_list=true_name_list,\
                            p_source_names=p_source_names, p_cas_numbers=p_cas_numbers, p_source_nums=p_source_nums,\
                            p_list_structure_links=p_list_structure_links,\
                            top_bit_structure_links_list=top_bit_structure_links_list, top_bit_structure_names_list=top_bit_structure_names_list, top_bit_display_names_list=top_bit_display_names_list,\
                            chem_type=chem_type, check_bitnum=check_bitnum, status_step1=status_step1, status_step2=status_step2, status_step3=status_step3)


# 実験結果のダウンロード
@app.route('/download_zip/<exp_id>',methods=['GET'])
@login_required
def downloadzip(exp_id):
    file_path = mfn.get_expid_path(exp_id)
    zip_dir = mfn.get_zip_path()

    if not os.path.isdir(zip_dir):
        os.makedirs(zip_dir)

    os.chdir(zip_dir)
    shutil.make_archive(exp_id,'zip',root_dir=file_path)

    # response
    response = make_response()
    response.data = open(zip_dir+'/'+exp_id+'.zip',"rb").read()
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers['Content-Disposition'] = 'attachment;filename=' +exp_id+'.zip'
    return response

# s3にローカルファイルをアップロード
@app.route('/upload/<exp_id>', methods=['POST'])
def upload_file(exp_id):
    # logging
    step_all_log = mfn.get_step_all_log_filename(exp_id)
    # format
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)

    # ファイルがなかった場合の処理
    if 'file' not in request.files:
        return redirect(request.url)
    # データの取り出し
    file = request.files['file']
    # ファイル名がなかった時の処理
    if file.filename == '':
        return redirect(request.url)
    # ファイルのチェック
    if file and vlib.allowed_file(file.filename):
        # 危険な文字を削除（サニタイズ処理）
        filename = secure_filename(file.filename)
        save_path = mfn.get_csv_path(exp_id)
        save_file = os.path.join(save_path, filename)
        file.save(save_file)

        user_id = experiment['userId']
        input_s3_bucket = experiment["user_s3_bucket"]
        input_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
        input_s3_filename = datamgmt.upload_file(input_s3_bucket, save_file, input_s3_prefix)
        if os.path.exists(save_file):
            os.remove(save_file)
        ex.update_input_s3_filename(id=exp_id, input_s3_filename=input_s3_filename)

        ex.update_error_message(id=exp_id, error_message='Status: {} のアップロードが完了しました'.format(filename))
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Compleated to upload input file')
        logmgmt.logInfo(exp_id, "Complete: upload input file: {}".format(input_s3_filename), step_all_log)
    else:
        error_msg = 'Error  : ファイルの拡張子がcsv,xlsxではありません。ファイルアップロードできませんでした'
        ex.update_error_message(id=exp_id, error_message=error_msg)
        logmgmt.logError(exp_id, error_msg, step_all_log)
        return redirect(url_for('experiment',exp_id=exp_id))
    return redirect(url_for('experiment',exp_id=exp_id))

# s3ファイルパスを入力パスにコピー
@app.route("/pathcopy/<exp_id>", methods=['POST'])
def patycopy(exp_id):
    # argument
    submit_data = request.get_json()
    filename = submit_data["filename"]
    path_type = submit_data["pathtype"]

    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    user_id = experiment['userId']
    input_s3_bucket = experiment["user_s3_bucket"]
    input_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
    s3_filename = "s3://" + input_s3_bucket + "/" + input_s3_prefix + filename

    if path_type=="input":
        ex.update_input_s3_filename(id=exp_id, input_s3_filename=s3_filename)
    json_data = {"filepath": s3_filename}
    return jsonify(json.dumps(json_data))

# s3ファイルをローカルにダウンロード
@app.route("/download_file/<exp_id>", methods=["GET"])
def download_file(exp_id):
    # argument
    filename = request.args.get("filename")

    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    user_id = experiment['userId']
    input_s3_bucket = experiment["user_s3_bucket"]
    input_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
    s3_file_path = input_s3_prefix + filename
    
    save_path = mfn.get_csv_path(exp_id)
    datamgmt.download_file(input_s3_bucket, s3_file_path, save_path)

    return send_from_directory(directory=save_path, path=filename, as_attachment=True)

# s3ファイルを削除
@app.route("/delete_file/<exp_id>", methods=["GET"])
def delete_file(exp_id):
    # argument
    filename = request.args.get("filename")

    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    user_id = experiment['userId']
    input_s3_bucket = experiment["user_s3_bucket"]
    input_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
    s3_file_path = input_s3_prefix + filename

    s3_client = boto3.client('s3')
    s3_client.delete_object(Bucket=input_s3_bucket, Key=s3_file_path)
    ex.update_input_s3_filename(id=exp_id, input_s3_filename="")
    ex.update_error_message(id=exp_id, error_message='Status: {} を削除しました'.format(filename))
    ex.update_error_message_en(id=exp_id, error_message_en='Status: Compleated to delete file')

    return redirect(url_for('experiment',exp_id=exp_id))


# Step1:データ前処理(データのプレビュー)
@app.route('/preview_s3_bucket_data/<exp_id>',methods=['POST'])
@login_required
def preview_s3_bucket_data(exp_id):
    try:
        # logging
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: preview_s3_bucket_data", step_all_log)

        # format
        ex = Experiments()
        vlib.update_progress_status(exp_id)

        # ブラウザから情報取得
        submit_data = request.get_json()
        s3uri_original_data = str(submit_data["s3_input"])
        excel_password = str(submit_data["excel_password"])

        # エラー処理
        if str(submit_data["s3_input"])=='':
            error_msg = 'Error  : 分析対象のs3パスを入力してください'
            vlib.update_error(exp_id, error_msg, "step1")
            return redirect(url_for('experiment',exp_id=exp_id), code=200)

        # データ読み込み
        df_s3_filename = mfn.get_preview_filename(exp_id)
        df_s3 = vlib.read_s3_data(s3uri_original_data, exp_id, excel_password)
        df_s3.to_csv(df_s3_filename, index=False)
        ex.update_s3uri_original_data(id=exp_id, s3uri_original_data=s3uri_original_data)
        ex.update_input_s3_filename(id=exp_id, input_s3_filename=s3uri_original_data)
        logmgmt.logInfo(exp_id, "In: {}".format(s3uri_original_data), step_all_log)
        # オーダーの確認
        pow_check_cols = vlib.get_pow_check_cols(df_s3)
        ex.update_pow_check_cols(id=exp_id, pow_check_cols=pow_check_cols)

        # 分子構造分析利用時
        structure_mode = str(submit_data["chem_type"])
        s3uri_master_data = ""
        if structure_mode=="mfp" or structure_mode=="maccs":
            s3uri_master_data = str(submit_data["s3_master"])
            master_password = str(submit_data["master_password"])

        if s3uri_master_data != "":
            master_dict = vlib.get_master_dict(exp_id, s3uri_master_data, master_password)
            ex.update_chem_type(id=exp_id, chem_type=structure_mode)
            ex.update_s3uri_master_data(id=exp_id, s3uri_master_data=s3uri_master_data)
            ex.update_master_dict(id=exp_id, master_dict=master_dict)
            logmgmt.logInfo(exp_id, "In: {}".format(s3uri_master_data), step_all_log)

        ex.update_error_message(id=exp_id, error_message='Status: データ読込が実行されました')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Completed to read input data')
        logmgmt.logInfo(exp_id, "Process End: preview_s3_bucket_data", step_all_log)
        ############正常終了
        return jsonify({"OK": "OK"})

    except Exception as error_msg:
        vlib.update_error(exp_id, str(error_msg), "step1")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)


# Step1 登録時の挙動
@app.route('/read_s3_bucket_data/<exp_id>',methods=['POST'])
@login_required
def read_s3_bucket_data(exp_id):
    try:
        # initialized
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        user_s3_bucket = experiment["user_s3_bucket"]
        latest_objectives = experiment["objectives"].split(",")
        latest_search_method = experiment["search_method"]
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: create train data", step_all_log)

        # format
        vlib.update_progress_status(exp_id)
        vlib.remove_step1_localfiles(exp_id)
        vlib.remove_chemical_files(exp_id)
        vlib.remove_step2_localfiles(exp_id, latest_objectives)
        vlib.remove_step3_localfiles(exp_id, latest_objectives, latest_search_method)
        ex.update_check_finished(id=exp_id, finished="NOT")

        # ブラウザから情報取得
        submit_data = request.get_json()
        # 目的変数設定
        objectives = str(submit_data["objectives"]).split(',')
        if str(submit_data["objectives"])=='':
            error_msg = 'Error  : 目的変数を1つ以上入力してください'
            vlib.update_error(exp_id, error_msg, "step1")
            return redirect(url_for('experiment',exp_id=exp_id), code=200)

        # データ読み込み
        download_path = mfn.get_preview_filename(exp_id)
        df_s3 = prep.read_csv(download_path, exp_id)

        # 不要列設定
        drop_cols = str(submit_data["drop_cols"]).split(",")
        df_reg = vlib.set_drop_cols(exp_id, df_s3, drop_cols)
        # データチェック
        vlib.check_input_data(exp_id, df_reg, objectives)
        # 学習データ作成
        prep.create_sampling_prepare(df_reg, exp_id, objectives)
        s3_uri_list = vlib.create_train_data(exp_id, df_reg, objectives, user_s3_bucket)
        ex.update_s3_uri_list(id=exp_id,s3_uri_list=",".join(s3_uri_list))

        ex.update_objectives(id=exp_id,objectives=str(submit_data["objectives"]))
        ex.update_objectives_message(id=exp_id, objectives_message=str(submit_data["objectives"]))
        ex.update_drop_cols(id=exp_id,drop_cols=str(submit_data["drop_cols"]))
        ex.update_error_message(id=exp_id, error_message='Status: 解析用のデータを作成しました')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Compleated to create training data')
        logmgmt.logInfo(exp_id, "Process End: create train data", step_all_log)
        ############正常終了
        return jsonify({"OK": "OK"})

    except Exception as error_msg:
        ex.update_error_message(id=exp_id, error_message=str(error_msg))
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

# Step1 可視化
@app.route('/visualization_data/<exp_id>',methods=['POST'])
@login_required
def visualization_data(exp_id):
    # format
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    vlib.update_progress_status(exp_id)

    # argument
    submit_data = request.get_json()
    vis_method = submit_data["vis_method"]
    if len(vis_method) == 0:
        error_msg = 'Error  : 可視化手法が選択されていません'
        vlib.update_error(exp_id, error_msg, "step1")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

    # s3アップロード用
    user_s3_bucket = experiment["user_s3_bucket"]

    # 変数名の入力チェック用
    down_load_data = mfn.get_preview_filename(exp_id)
    df_s3 = prep.read_csv(down_load_data, exp_id)
    drop_cols = str(experiment["drop_cols"]).split(",")
    df_reg = vlib.set_drop_cols(exp_id, df_s3, drop_cols)

    # 可視化する変数
    vis_cb = submit_data["vis_cb"]
    vis_list = [elm for elm in vis_cb if elm!= ""]
    if len(vis_list) == 0:
        error_msg = 'Error  : 変数が選択されていません'
        vlib.update_error(exp_id, error_msg, "step1")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)
    elif len(vis_list) > 20:
        error_msg = 'Error  : 変数は20個以下に限定してください'
        vlib.update_error(exp_id, error_msg, "step1")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

    # 可視化用のデータ
    df_vis = df_reg[vis_list]

    # format
    vlib.remove_step1_localfiles(exp_id)
    # データベース更新
    ex.update_vis_cols(id=exp_id, vis_cols=", ".join(vis_list))
    ex.update_vis_method(id=exp_id, vis_method=vis_method)

    # 可視化手法の表示
    vis_method_message_list = []
    if "pairplot" in vis_method:
        vis_method_message_list.append("散布図行列")
    if "correlation_matrix" in vis_method:
        vis_method_message_list.append("相関行列")
    if "profiles" in vis_method:
        vis_method_message_list.append("Pandas-Profiling")
    vis_method_message = ", ".join(vis_method_message_list)
    ex.update_vis_method_message(id=exp_id, vis_method_message=vis_method_message)

    # 可視化実行           
    try:
        t = vthd.subThread_vis(df_vis, user_s3_bucket, vis_method, exp_id)
        t.start()
        ############正常終了
        return jsonify({"OK": "OK"})
    except:
        error_msg = 'Error  : サーバー処理の負荷によりリクエストが実行されませんでした。数十分後に再実行してください'
        vlib.update_error(exp_id, error_msg, "step1")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)


# Step1 mfp用
@app.route('/read_chemical_data/<exp_id>',methods=['POST'])
@login_required
def read_chemical_data(exp_id):
    try:
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        objectives = experiment["objectives"].split(",")
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: read_chemical_data", step_all_log)
        ex.update_error_message(id=exp_id, error_message='Status: 原材料に含まれる分子構造を検索中です')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Searching molecular structure in source')
        
        # ブラウザからデータ取得
        submit_data = request.get_json()
        try:
            radius = int(submit_data["radius"])
            ex.update_radius(id=exp_id, radius=radius)
            if experiment["check_bitnum"] == "short":
                bit_num = int(submit_data["bit_num"])
                ex.update_bit_num(id=exp_id, bit_num=bit_num)
            else:
                bit_num = int(experiment["bit_num"])
        except:
            error_msg = 'Error  : 数値が入力されていません'
            vlib.update_error(exp_id, error_msg, "not")
            return redirect(url_for('experiment',exp_id=exp_id))

        df_s3_file_name = mfn.get_preview_filename(exp_id)
        df_s3 = prep.read_csv(df_s3_file_name, exp_id)
    
        # 不要列設定
        drop_cols = str(experiment["drop_cols"]).split(",")
        df_reg = vlib.set_drop_cols(exp_id, df_s3, drop_cols)
        # upload data
        user_s3_bucket = experiment["user_s3_bucket"]
        s3_uri_list = vlib.create_train_data(exp_id, df_reg, objectives, user_s3_bucket, radius=radius, bit_num=bit_num)
        ex.update_s3_uri_list(id=exp_id,s3_uri_list=",".join(s3_uri_list))
        
        ex.update_error_message(id=exp_id, error_message='Status: 分子構造が描画されました')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Compleated to search molecular structure')
        logmgmt.logInfo(exp_id, "Process End: read_chemical_data", step_all_log)
        ############正常終了
        return jsonify({"OK": "OK"})

    except Exception as error_msg:
        ex.update_error_message(id=exp_id, error_message=str(error_msg))
        return redirect(url_for('experiment',exp_id=exp_id))



# Step2 モデル構築
@app.route('/create_model/<exp_id>',methods=['POST'])
@login_required
def create_model(exp_id):
    # format
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    objectives = experiment['objectives'].split(',')
    vlib.update_progress_status(exp_id)
    #前のデータを削除
    vlib.remove_step2_localfiles(exp_id, objectives)

    # ブラウザからデータを取得
    submit_data = request.get_json()

    # 機械学習モデルタイプの指定
    problemtype = submit_data["ptype"]
    ex.update_problemtype(id=exp_id,problemtype=problemtype)

    # モデル評価指標の設定
    if problemtype == 'Regression':
        eval_metrics = 'MSE'
    elif problemtype == 'BinaryClassification':
        eval_metrics = 'F1'    
    elif problemtype == "MulticlassClassification":
        eval_metrics = 'F1macro'

    # モデル名・評価値表示用
    target_list = vlib.replace_slash_in_target(objectives)
    train_data_list = [mfn.get_trainob_filename(exp_id, target_name) for target_name in target_list]

    # s3パスの取得
    s3_uri_list = experiment['s3_uri_list'].split(',')
    user_s3_bucket = experiment["user_s3_bucket"]
    try:
        t = vthd.subThread_modeling(exp_id, user_s3_bucket, problemtype, objectives, s3_uri_list, train_data_list, eval_metrics)
        t.start()
        ############正常終了
        return jsonify(json.dumps({"OK": problemtype}))        
    except:
        error_msg = 'Error  : サーバー処理の負荷によりリクエストが実行されませんでした。数十分後に再実行してください'
        vlib.update_error(exp_id, error_msg, "step2")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)


# Step2 物質名検索
@app.route('/search_chemical_name/<exp_id>',methods=['POST'])
@login_required
def search_chemical_name(exp_id):
    try:
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        objectives = experiment["objectives"].split(",")
        # logging
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: search_chemical_name", step_all_log)
        

        # ブラウザからデータ取得
        submit_data = request.get_json()
        source_names = str(submit_data["source_names"]).split(",")

        if str(submit_data["source_names"]) == "":
            error_msg = 'Error  : 必要事項を正しく入力してください'
            vlib.update_error(exp_id, error_msg, "not")
            return redirect(url_for('experiment',exp_id=exp_id))

        # shap値上位の物質名を取得
        ex.update_error_message(id=exp_id, error_message='Status: 原材料に含まれる分子構造を検索中です')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Searching molecular structure in source')
        rank_dict = vlib.search_source_names(exp_id, source_names, objectives)

        # 物質名一覧を取得
        master_filename = experiment["s3uri_master_data"]
        df_master = prep.read_s3_bucket_data(master_filename, exp_id)
        source_list = list(df_master['Source_Name'])
        bit_dict={}
        objective_dict={}
        for source_name in source_names:
            if not source_name in source_list:
                error_msg = 'Error  : 原材料名が入力データに存在しません。'
                vlib.update_error(exp_id, error_msg, "not")
                return redirect(url_for('experiment',exp_id=exp_id))

            items_sorted = sorted(rank_dict[source_name].items(), key = lambda x : int(x[0]))
            bit_list = [item[0] for item in items_sorted]
            bit_dict[source_name] = bit_list
            
            objective_dict[source_name]={}
            for i, target in enumerate(objectives):
                rank_list = [rank_dict[source_name][bit][i] for bit in bit_list]
                objective_dict[source_name][target]=rank_list

        ex.update_source_names(id=exp_id, source_names=source_names)
        ex.update_bit_dict(id=exp_id, bit_dict=bit_dict)
        ex.update_objective_dict(id=exp_id, objective_dict=objective_dict)
        ex.update_error_message(id=exp_id, error_message='Status: 原材料に含まれる分子構造が表示されました')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Completed to search molecular structure')
        logmgmt.logInfo(exp_id, "Process End: search_chemical_name", step_all_log)

        ############正常終了
        return jsonify({"OK": "OK"})

    except Exception as error_msg:
        vlib.update_error(exp_id, str(error_msg), "not")
        return redirect(url_for('experiment',exp_id=exp_id))



# Step2 shap値上位の分子構造の描画
@app.route('/draw_top_data/<exp_id>',methods=['POST'])
@login_required
def draw_top_data(exp_id):
    try:
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        objectives = experiment['objectives'].split(',')
        # logging
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: search_chemical_name", step_all_log)

        # ブラウザからデータを取得
        submit_data = request.get_json()
        # s3のインプットパスを読み込み
        try:
            top_num = int(submit_data["top_num"])
        except:
            error_msg = "順位に数値が入力されていません"
            vlib.update_error(exp_id, error_msg, "not")
            return redirect(url_for('experiment',exp_id=exp_id))

        ex.update_error_message(id=exp_id, error_message='Status : 分子構造を検索中です')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Seaching molecular structure')
        
        # 変数名の入力チェック用
        samplingx_filename = mfn.get_samplingx_filename(exp_id)
        df_samplingx_columns = pd.read_csv(samplingx_filename).columns.tolist()
        structure_mode = experiment["chem_type"]
        master_filename = experiment["s3uri_master_data"]
        radius = 0
        bit_num = 4096
        if structure_mode == "mfp":
            radius = int(experiment["radius"])
            if experiment["check_bitnum"] == "short":
                bit_num = int(experiment["bit_num"])
        chem = chemembeding.Features(master_filename, df_samplingx_columns, exp_id, structure_mode, radius, bit_num)
        chem.get_smiles()
        name_list = chem.draw_topnum(top_num, objectives)

        ex.update_chem_list(id=exp_id, chem_list=name_list)
        ex.update_error_message(id=exp_id, error_message='Status : shap値上位の分子構造が描画されました')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Completed to search moleculear structure with top shap value')
        ############正常終了
        return jsonify({"OK": "OK"})

    except Exception as error_msg:
        vlib.update_error(exp_id, str(error_msg), "not")
        return redirect(url_for('experiment',exp_id=exp_id))


# 制約条件の設定
def check_number(num1, num2 = ""):
    if num1 != "" and num2 == "":
        try:
            num1_f = float(num1)
        except:
            return False
    elif num1 == "" and num2 != "":
        try:
            num2 = float(num2)
        except:
            return False
    elif num1 != "" and num2 != "":
        try:
            num1_f = float(num1)
            num2_f = float(num2)
        except:
            return False
    return True


# Step3 サンプリング設定
@app.route('/create_sample/<exp_id>',methods=['POST'])
@login_required
def create_sample(exp_id):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    vlib.update_progress_status(exp_id, step2_num=100, step2_sts="finished")
    # logging
    step_all_log = mfn.get_step_all_log_filename(exp_id)
    logmgmt.logInfo(exp_id, "Process Start: Confirm limit conditions", step_all_log)

    # 変数名の入力チェック用
    df_s3_file_name = mfn.get_samplingx_filename(exp_id)
    df_for_sampling = pd.read_csv(df_s3_file_name)
    df_columns = list(df_for_sampling.columns)
    #Ajax通信でデータ受け取り
    submit_data = request.get_json()

    # 辞書型のデータ定義
    range_mm = {}
    for df_col in df_columns:
        range_mm[df_col] = {}
    # 各説明変数のサンプリング範囲を最小値以上、最大値以下に設定
    for col in df_columns:
        range_mm[col]["mn"] = df_for_sampling[col].min()
        range_mm[col]["mx"] = df_for_sampling[col].max()

    try:
        # 組み合わせを指定
        combination_cb = submit_data["combination_cb"]
        combination_lower = submit_data["combination_lower"]
        combination_upper = submit_data["combination_upper"]
        set_condition_class = SetCondition(exp_id, df_columns, range_mm)
        combination_selects, combination_pairs, combination_dict = set_condition_class.get_combination_condition(combination_cb, combination_lower, combination_upper)
        ex.update_combination_condition(id=exp_id,combination_condition=combination_dict)

        # 値範囲を指定
        range_sel = submit_data["range_sel"]
        range_lower = submit_data["range_lower"]
        range_upper = submit_data["range_upper"]
        range_cols, range_pairs, range_dict = set_condition_class.get_range_condition(range_sel, range_lower, range_upper)
        ex.update_range_condition(id=exp_id,range_condition=range_dict)

        # 固定値を指定
        fixed_sel = submit_data["fixed_sel"]
        fixed_val = submit_data["fixed_val"]
        fixed_cols, fixed_values, fixed_dict = set_condition_class.get_fixed_condition(fixed_sel, fixed_val)
        ex.update_fixed_condition(id=exp_id,fixed_condition=fixed_dict)

        # 比率を指定
        ratio1_sel = submit_data["ratio1_sel"]
        ratio2_sel = submit_data["ratio2_sel"]
        ratio1_val = submit_data["ratio1_val"]
        ratio2_val = submit_data["ratio2_val"]
        ratio_selects, ratio_pairs, ratio_dict = set_condition_class.get_ratio_condition(ratio1_sel, ratio2_sel, ratio1_val, ratio2_val)
        ex.update_ratio_condition(id=exp_id,ratio_condition=ratio_dict)

        # 合計値を指定
        total_cb = submit_data["total_cb"]
        total_val = submit_data["total_val"]
        total_selects, total_values, total_dict = set_condition_class.get_total_condition(total_cb, total_val)
        ex.update_total_condition(id=exp_id,total_condition=total_dict)

        # グループ和を指定
        group_cb_list = submit_data["group_cb_list"]
        group_lower_list = submit_data["group_lower_list"]
        group_upper_list = submit_data["group_upper_list"]
        group_total_vals = submit_data["group_total"]
        group_selects, group_pairs, group_totals, group_dict_list, group_dict_total = set_condition_class.get_groupsum_condition(group_cb_list, group_lower_list, group_upper_list, group_total_vals)
        ex.update_groupsum_condition(id=exp_id,groupsum_condition=group_dict_list)
        ex.update_groupsum_total_condition(id=exp_id, groupsum_total_condition=group_dict_total)
    except Exception as e:
        print(str(e))
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

    
    # サンプリング情報の取得
    nega_flag = submit_data["nega_flag"]
    number_of_sampling = int(submit_data["sampling_num"])
    sampling_width = float(submit_data["sampling_width"])
    if number_of_sampling > 100000 or number_of_sampling < 1:
        error_msg = "Error : サンプリング回数は10万以下で設定してください"
        vlib.update_error(exp_id, error_msg, "step3")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

    # モデル情報の取得
    objectives = experiment['objectives'].split(',')
    model_list = vlib.get_model_list(exp_id, objectives)
    if len(model_list) == 0:
        error_msg = "Error: モデルの読み込みに失敗しました"
        vlib.update_error(exp_id, error_msg, "step3")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

    # インスタンス生成
    sp = Sampling(exp_id, combination_selects, combination_pairs, range_cols, range_pairs, fixed_cols, fixed_values, ratio_selects, ratio_pairs, total_selects, total_values, group_selects, group_pairs, group_totals, number_of_sampling=number_of_sampling, conf_width=sampling_width, nega_flag=nega_flag)
    # サンプリング情報の表示
    combination_message = sp.describe_combination()
    ex.update_combination_message(id=exp_id,combination_message=combination_message)
    range_message = sp.describe_range()
    ex.update_range_message(id=exp_id,range_message=range_message)
    fixed_message = sp.describe_fixed()
    ex.update_fixed_message(id=exp_id,fixed_message=fixed_message)
    ratio_message = sp.describe_ratio()
    ex.update_ratio_message(id=exp_id,ratio_message=ratio_message)
    total_message = sp.describe_total()
    ex.update_total_message(id=exp_id,total_message=total_message)
    groupsum_message = sp.describe_groupsum()
    ex.update_groupsum_message(id=exp_id, groupsum_message=groupsum_message)

    # S3パス設定
    user_s3_bucket = experiment["user_s3_bucket"]
    try:
        t = vthd.subThread_sampling(exp_id, sp, objectives, model_list, user_s3_bucket)
        t.start()
        ############正常終了
        return jsonify(json.dumps({"OK": "OK"}))
    except:
        error_msg = "Error : サーバー処理の負荷によりリクエストが実行されませんでした。数十分後に再実行してください"
        vlib.update_error(exp_id, error_msg, "step3")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

# Step3: パラメータ探索
@app.route('/search_params/<exp_id>',methods=['POST'])
@login_required
def search_params(exp_id):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    vlib.update_progress_status(exp_id, step2_num=100, step3_num=100, step2_sts="finished", step3_sts="finished")
    # logging
    step_all_log = mfn.get_step_all_log_filename(exp_id)
    logmgmt.logInfo(exp_id, "Process Start: Confirm limit conditions", step_all_log)

    # 探索に必要な情報を読み込み
    objectives = experiment['objectives'].split(',')
    problemtype = experiment['problemtype']
    sample_filename = mfn.get_sample_filename(exp_id)
    samples = pd.read_csv(sample_filename)

    #Ajax通信でデータ受け取り
    submit_data = request.get_json()
    cluster_num = int(submit_data["cluster_num"])

    # 探索手法を設定
    search_method = submit_data['search_method']
    if len(search_method) == 0:
        error_msg = "Error: 探索手法が設定されていません"
        vlib.update_error(exp_id, error_msg, "step3")
        return redirect(url_for('experiment',exp_id=exp_id), code=200)
    ex.update_search_method(id=exp_id,search_method=search_method)

    # 辞書型のデータ定義
    range_mm = {}
    for col in objectives:
        range_mm[col] = {}
    # 各説明変数のサンプリング範囲を最小値以上、最大値以下に設定
        range_mm[col]["mn"] = samples[col].min()
        range_mm[col]["mx"] = samples[col].max()

    try:
        # 目的変数の目標値を設定
        target_sel = submit_data["target_sel"]
        target_lower = submit_data["target_lower"]
        target_upper = submit_data["target_upper"]
        set_condition = SetCondition(exp_id, samples.columns, range_mm)
        target_cols, target_pairs, target_message = set_condition.get_target_condition(target_sel, target_lower, target_upper)
        ex.update_target_message(id=exp_id, target_message=target_message)

        # 刻み値
        step_sel = submit_data["step_sel"]
        step_val = submit_data["step_val"]
        step_dict = set_condition.get_step_dict(step_sel, step_val)

        status_msg = 'Status : 条件に合致した実験条件を探索中です'
        ex.update_error_message(id=exp_id, error_message=status_msg)
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Seaching parameters that meet target setting')
    except Exception as e:
        print(str(e))
        return redirect(url_for('experiment',exp_id=exp_id), code=200)

    try:
        # 探索実行
        ps = paramsearch.Search(objectives, problemtype, exp_id, search_method)
        ps.search_samples(samples, target_cols, target_pairs, step_dict, cluster_num=cluster_num)

        ex.update_error_message(id=exp_id, error_message='Status: 探索が実行されました(画像の読み込みに時間がかかることがあります)')
        ex.update_error_message_en(id=exp_id, error_message_en='Status: Completed to search parameters')            
        ############正常終了
        time.sleep(1)
        return jsonify(json.dumps({"OK": "OK"}))
    except:
        error_msg = "Error : サーバー処理の負荷によりリクエストが実行されませんでした。数十分後に再実行してください"
        vlib.update_error(exp_id, error_msg, "step3")
        return redirect(url_for('experiment',exp_id=exp_id))      
   
# プログレスバーストリーム
@app.route("/stream/<name>/<exp_id>")
def stream(exp_id, name):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    if experiment["status_{0}".format(name)] == "finished":
        response = Response(event_finish(exp_id, name), mimetype='text/event-stream')
    else:
        response = Response(event_stream(exp_id, name), mimetype='text/event-stream')
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response

# EventSourceの'progress-item'に出力（100だったら'last-item'イベントに出力）
def event_stream(exp_id, name):
    persent = 0
    while persent!=100:
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        persent = experiment["progress_rate_{0}".format(name)]
        status = experiment["system_status"]
        if persent==0:
            break
        sse_event = "progress-item-"+name
        if persent == 100:
            sse_event = "last-item-"+name
        dataset = json.dumps({"persent": persent, "status": status})
        yield "event:{event}\ndata:{data}\n\n".format(event=sse_event, data=dataset)
        gevent.sleep(2)

# EventSourceの'progress-item'に出力（100だったら'last-item'イベントに出力）
def event_finish(exp_id, name):
    sse_event = "finish-item-"+name
    dataset = json.dumps({"persent": "100", "status": "finish"})
    yield "event:{event}\ndata:{data}\n\n".format(event=sse_event, data=dataset)

#登録済み可視化変数を反映
@app.route("/recover_step1/<exp_id>", methods=['POST'])
@login_required
def recover_step1(exp_id):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    submit_data = request.get_json()

    vis_method = experiment["vis_method"]
    if len(vis_method) == 0:
        vis_method = ["pairplot", "correlation_matrix", "profiles"]
    print(vis_method)
    vis_cols = experiment["vis_cols"].split(", ")
    print(vis_cols)

    json_data = {"vis_method": vis_method, "vis_cols": vis_cols}
    return jsonify(json.dumps(json_data))

#登録済み制約条件を反映
@app.route("/recover_step3/<exp_id>", methods=['POST'])
@login_required
def recover_step3(exp_id):
    ex = Experiments()
    experiment = ex.find_one_id(exp_id)
    submit_data = request.get_json()
    
    range_vals = experiment["range"]
    print(range_vals)
    fixed_vals = experiment["fixed"]
    print(fixed_vals)
    total_vals = experiment["total"]
    print(total_vals)
    combination_vals = experiment["combination"]
    print(combination_vals)
    ratio_vals = experiment["ratio"]
    print(ratio_vals)
    groupsum_vals = experiment["groupsum"]
    groupsum_total = experiment["groupsum_total"]
    print(groupsum_vals)
    print(groupsum_total)

    json_data = {"range": range_vals, "fixed": fixed_vals, "total": total_vals, "combination": combination_vals, "ratio": ratio_vals, "groupsum": groupsum_vals, "groupsum_total": groupsum_total}
    return jsonify(json.dumps(json_data))


# アカウント情報変更画面
@app.route('/account_change/<user_id>',methods=['GET', 'POST'])
@login_required
def account_change(user_id):
    usr=Users()
    user=usr.find_one_userid(user_id)
    cur_dept = str.upper(user["department"])

    return render_template("accountchange.html", user_id=user_id, cur_dept=cur_dept)




# パスワード変更
@app.route('/change_passwd/<user_id>',methods=['POST'])
def change_passwd(user_id):
    # arguments
    old_passwd = str(request.form["old_passwd"])
    new_passwd = str(request.form["new_passwd"])
    re_passwd = str(request.form["re_passwd"])

    # パスワードの変更と、パスワード変更回数のカウント
    usr = Users()
    user = usr.find_one_userid(user_id)
    now_passwd = user["passWd"]
    if now_passwd == old_passwd:
        if now_passwd == new_passwd:
            flash("Error: 古いパスワードと新しいパスワードが同じです", "failed")            
        elif new_passwd == re_passwd:
            if adm.check_passwd(new_passwd):
                usr.update_passwd(userid=user_id, passwd=new_passwd)
                passwd_count = user["passWd_change"]
                usr.update_passwd_change(userid=user_id, passwd_change=passwd_count+1)
                if request.args.get("change_id") == "1":
                    flash("Status: パスワードが変更されました", "success")
                elif request.args.get("change_id") == "2":
                    flash("Status: パスワードが変更されました。GUIアカウントが有効化されました", "success")
                    return redirect(url_for("login"))
            else:
                flash("Error: パスワードルールを満たしていません", "failed")
        else:
            flash("Error: 新しいパスワードが一致していません", "failed")
    else:
        flash("Error: 現在のパスワードが間違っています", "failed")

    # 元の画面に戻る
    if request.args.get("change_id") == "1":
        return redirect(url_for("account_change", user_id=user_id))
    elif request.args.get("change_id") == "2":
        return redirect(request.referrer)
