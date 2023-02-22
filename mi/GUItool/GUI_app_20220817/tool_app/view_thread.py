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
from tool_app import viewlib as vlib


# Step1:データ前処理
class subThread_vis(threading.Thread):
    def __init__(self, df_reg, bucket_name, methods, exp_id):
        super(subThread_vis,self).__init__()
        self.stop_event = threading.Event()
        self._df_reg = df_reg
        self._bucket_name = bucket_name
        self._methods = methods
        self._exp_id = exp_id

    def stop(self):
        self.stop_event.set()
        
    def run(self):
        df_reg = self._df_reg
        bucket_name = self._bucket_name
        methods = self._methods
        exp_id = self._exp_id

        # logging
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: visualize", step_all_log)
        logmgmt.logInfo(exp_id, "Progress detail is written in step1_visualize.log", step_all_log)

        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        user_id = experiment["userId"]
        status_msg = 'Status : データ可視化が実行されています'
        ex.update_error_message(id=exp_id, error_message=status_msg)
        ex.update_error_message_en(id=exp_id, error_message_en="Status: Visualizing input data...")
        ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=5)
        ex.update_status_step1(id=exp_id, status_step1="progress")

        # 可視化
        try:
            # 可視化メソッドの呼び出し        
            for method in methods:
                ex = Experiments()                
                progress_msg = 'データ可視化中({0}/{1})'.format((methods.index(method)), str(len(methods)))
                ex.update_status(id=exp_id, system_status=progress_msg)

                if method=="profiles":
                    s3_output_prefix = mfn.get_s3_html_path(user_id, exp_id)
                elif method=="pairplot" or method=="correlation_matrix":
                    s3_output_prefix = mfn.get_s3_img_path(user_id, exp_id)
                vis.show_method(method, df_reg, bucket_name, s3_output_prefix, exp_id)
                # progress bar
                r = int((methods.index(method)+1)/3*99)
                ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=r)

            ex.update_error_message(id=exp_id, error_message='Status : データ可視化が実行されました')
            ex.update_error_message_en(id=exp_id, error_message_en="Status: Completed to visualize input data")
            vlib.update_progress_status(exp_id, step1_num=100, step1_sts="wait")
            ex.update_status(id=exp_id, system_status="#")

            # count
            usr = Users()
            user=usr.find_one_userid(user_id)
            usr.update_vis_count(userid=user_id, vis_count=user["vis_count"]+1)
            # logging
            logmgmt.logInfo(exp_id, "Process End: visualize", step_all_log)

        except Exception as e:
            vlib.update_progress_status(exp_id)
            vlib.update_error(exp_id, str(e), "step1")
        finally:
            print('Fin\n')


# Step2;回帰モデル構築,評価
class subThread_modeling(threading.Thread):
    def __init__(self, exp_id, bucket_name, problemtype, objectives, s3_uri_list, traindata_list, eval_metrics):
        super(subThread_modeling,self).__init__()
        self.stop_event = threading.Event()
        self._exp_id = exp_id
        self._bucket_name = bucket_name
        self._problemtype = problemtype
        self._objectives = objectives
        self._s3_uri_list = s3_uri_list
        self._traindata_list = traindata_list
        self._eval_metrics = eval_metrics
    
    def stop(self):
        self.stop_event.set()
    
    def run(self):
        exp_id = self._exp_id
        bucket_name = self._bucket_name
        problemtype = self._problemtype
        objectives = self._objectives
        s3_uri_list = self._s3_uri_list
        traindata_list = self._traindata_list
        eval_metrics = self._eval_metrics
        
        def get_eval_list(str_arg, eval_value_list):
            eval_list = []
            for evalue in eval_value_list:
                eval_list.append(str(round(evalue[str_arg], 2)))
            return eval_list

        def calc_evalue(e_list):
            evalue = ""
            for v in e_list:
                evalue = evalue + v + ","
            evalue = evalue[:len(evalue) - 1]
            return evalue

        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        user_id = experiment["userId"]
        usr = Users()
        user=usr.find_one_userid(user_id)
        status_msg = 'Status : モデル構築・Shap値算出が実行中です'
        ex.update_error_message(id=exp_id, error_message=status_msg)
        ex.update_error_message_en(id=exp_id, error_message_en="Status: Building model and evaluate it...")
        ex.update_status_step2(id=exp_id, status_step2="progress")
        ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=5)

        # logging
        #create_logger(exp_id)
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: model building", step_all_log)
        logmgmt.logInfo(exp_id, "Progress detail is written in step2_modelbuilding.log", step_all_log)
        # Modeling
        try:
            # 実行時間計測
            start_time = time.perf_counter()

            # モデルインスタンス設定
            role = mfn.get_aws_role()
            mlmodel = modeling.SagemakerCtrl(bucket_name, role, exp_id, user_id, problemtype, eval_metrics)
            
            # モデル構築 
            model_dict = {}
            for j, obj in enumerate(objectives):
                ex = Experiments()
                progress_msg = "モデル構築中({0}/{1})".format(j, len(objectives))
                ex.update_status(id=exp_id, system_status=progress_msg)
                
                # モデル実行
                model_name = mlmodel.fit(obj, s3_uri_list[j], j)
                model_dict[obj] = model_name
                
                r = int((j+1)/len(objectives)*33)
                ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=r)
            ex.update_model_name(id=exp_id, model_name=model_dict)
            model_list = vlib.get_model_list(exp_id, objectives)

            # テストデータでの評価値
            Eval_value_list = []
            for j, obj in enumerate(objectives):
                ex = Experiments()
                progress_msg = "モデル評価指標算出中({0}/{1})".format(j, len(objectives))
                ex.update_status(id=exp_id, system_status=progress_msg)

                Eval_value = mlmodel.estimate_testdata(obj, model_list[j], traindata_list[j])
                Eval_value_list.append(Eval_value)
                
                # プログレスバー更新
                r = int((j+1)/len(objectives)*33) + 33
                ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=r)
            
            if problemtype == "Regression":
                R2_list = get_eval_list("R2", Eval_value_list)
                MAE_list = get_eval_list("MAE", Eval_value_list)
                MSE_list = get_eval_list("MSE", Eval_value_list)
                RMSE_list = get_eval_list("RMSE", Eval_value_list)
                ex.update_R2(id=exp_id, R2=calc_evalue(R2_list))
                ex.update_MAE(id=exp_id, MAE=calc_evalue(MAE_list))
                ex.update_MSE(id=exp_id, MSE=calc_evalue(MSE_list))
                ex.update_RMSE(id=exp_id, RMSE=calc_evalue(RMSE_list))
            else:
                Accuracy_list = get_eval_list("Accuracy", Eval_value_list)
                Precision_list = get_eval_list("Precision", Eval_value_list)
                Recall_list = get_eval_list("Recall", Eval_value_list)
                F_score_list = get_eval_list("F_score", Eval_value_list)
                ex.update_Accuracy(id=exp_id, Accuracy=calc_evalue(Accuracy_list))
                ex.update_Precision(id=exp_id, Precision=calc_evalue(Precision_list))
                ex.update_Recall(id=exp_id, Recall=calc_evalue(Recall_list))
                ex.update_F_score(id=exp_id, F_score=calc_evalue(F_score_list))                                
            
            # 線形モデルでの係数
            if problemtype == 'Regression':
                mlmodel.estimate_multi_coefficients(objectives, traindata_list)
            
            # shap値作成
            html_path_list = []
            for j, obj in enumerate(objectives):
                ex = Experiments()
                progress_msg = "Shap値算出中({0}/{1})".format(j, len(objectives))
                ex.update_status(id=exp_id, system_status=progress_msg)

                path = mlmodel.analyze(obj, s3_uri_list[j], traindata_list[j], model_list[j])
                html_path_list.append(path)

                r = int((j+1)/len(objectives)*33) + 67
                ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=r)

            # 実行時間計測
            end_time = time.perf_counter()
            run_minites = (end_time-start_time)/60.0
            print('time = {} Minutes'.format(run_minites))
            usr.update_sagemaker_time(userid=user_id, sagemaker_time=user["sagemaker_time"]+run_minites)

            # shapレポートの統合
            soup_list = []
            for html_path in html_path_list:
                with open(html_path) as f:
                    soup_list.append(BeautifulSoup(f.read(), 'lxml'))
            pure_bound_html = ''.join([soup.prettify() for soup in soup_list])
            bound_html = pure_bound_html.replace('</html>\n<html>\n', '')
            # 保存処理
            save_path = mfn.get_shap_filename(exp_id)
            with open(save_path, mode='w') as f:
                f.write(bound_html)

            ex.update_error_message(id=exp_id, error_message='Status : モデル構築・評価指標算出が実行されました')
            ex.update_error_message_en(id=exp_id, error_message_en="Status: Completed to build and evaluate model")
            vlib.update_progress_status(exp_id, step2_num = 100, step2_sts="finished")
            ex.update_check_finished(id=exp_id, finished="DONE")
            ex.update_status(id=exp_id, system_status="#")
            
            # count
            usr.update_model_count(userid=user_id, model_count=user["model_count"]+1)
            # logging
            logmgmt.logInfo(exp_id, "Process End: model building", step_all_log)

        except Exception as e:
            vlib.update_progress_status(exp_id)
            vlib.update_error(exp_id, str(e), "step2")
        finally:
            print('Fin\n')



# Step3;サンプル生成
class subThread_sampling(threading.Thread):
    def __init__(self, exp_id, sample_class, objectives, model_list, bucket_name):
        super(subThread_sampling,self).__init__()
        self.stop_event = threading.Event()
        self._exp_id = exp_id
        self._sample_class = sample_class
        self._objectives = objectives
        self._model_list = model_list
        self._bucket_name = bucket_name
    
    def stop(self):
        self.stop_event.set()
        
    def run(self):
        exp_id = self._exp_id
        sample_class = self._sample_class
        objectives = self._objectives
        model_list = self._model_list
        bucket_name = self._bucket_name
        
        # logging
        step_all_log = mfn.get_step_all_log_filename(exp_id)
        logmgmt.logInfo(exp_id, "Process Start: set condition and create sample", step_all_log)
        logmgmt.logInfo(exp_id, "Progress detail is written in step3_sampling.log", step_all_log)
        
        # initialize
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        user_id = experiment["userId"]
        usr = Users()
        user=usr.find_one_userid(user_id)
        status_msg = 'Status : サンプル生成、及びモデル推論が実行中です'
        ex.update_error_message_en(id=exp_id, error_message_en="Status: Sampling and infering...")
        ex.update_error_message(id=exp_id, error_message=status_msg)
        ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=5)
        ex.update_status_step3(id=exp_id, status_step3="progress")
        
        # sampling
        try:
            progress_msg = "サンプル生成中"
            ex.update_status(id=exp_id, system_status=progress_msg)
            # 列名チェック
            samplingx_filename = mfn.get_samplingx_filename(exp_id)
            df_samplingx_columns = pd.read_csv(samplingx_filename).columns.tolist()

            # 分子構造組込の情報取得
            structure_mode = experiment["chem_type"]
            if structure_mode == "maccs" or structure_mode == "mfp":
                master_filename = experiment["s3uri_master_data"]
                radius = 0
                bit_num = 4096
                if structure_mode == "mfp":
                    radius = int(experiment["radius"])
                    if experiment["check_bitnum"] == "short":
                        bit_num = int(experiment["bit_num"])
                chem = chemembeding.Features(master_filename, df_samplingx_columns, exp_id, structure_mode, radius, bit_num)
                chem.get_smiles()
            else:
                chem = ""
            # サンプル生成
            samples = sample_class.create_samples()

            # サンプル生成結果が0行より多い場合に、モデル推論
            if len(samples > 0):
                ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=25)
                
                ifr = Inference(exp_id, user_id, bucket_name)
                samples_converted = ifr.convert_chemical_structure(chem, samples)

                # logging
                step3_log = mfn.get_step3_createsample_log_filename(exp_id)
                logmgmt.logInfo(exp_id, "Process Start: predict sample", step3_log)
                
                # 実行時間計測
                start_time = time.perf_counter()

                # モデル推論
                pre_list = []
                for j in range(len(objectives)):
                    ex = Experiments()

                    # ステータス情報更新
                    progress_msg = 'モデル推論中({0}/{1})'.format(j, str(len(objectives)))
                    ex.update_status(id=exp_id, system_status=progress_msg)                
                        
                    # 実行関数
                    predicted = ifr.model_inference(model_list[j], samples_converted, objectives[j])
                    pre_list.append(predicted)

                    # プログレスバー更新
                    r = int((j+1)/len(objectives)*75) + 25
                    ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=r)

                # 実行時間計測
                end_time = time.perf_counter()
                run_minites = (end_time-start_time)/60.0
                print('time = {} Minutes'.format(run_minites))
                usr.update_sagemaker_time(userid=user_id, sagemaker_time=user["sagemaker_time"]+run_minites)

                pre_array = np.vstack(pre_list).T
                ys = pd.DataFrame(pre_array,index=samples.index,columns=objectives)
                samples_predicted = pd.concat([samples,ys], axis=1)

                # データの保存
                sample_filename = mfn.get_sample_filename(exp_id)
                samples_predicted.to_csv(sample_filename, index=False, sep=',')
                #logging
                logmgmt.logDebug(exp_id, "Out: {}".format(sample_filename), step3_log)
                logmgmt.logInfo(exp_id, "Process End: predict sample", step3_log)

                # message
                ex.update_error_message(id=exp_id, error_message='Status : サンプル生成、及びモデル推論が実行されました')
                ex.update_error_message_en(id=exp_id, error_message_en="Status: Completed sampling and model inference")
                vlib.update_progress_status(exp_id, step2_num = 100, step3_num = 100, step2_sts="finished", step3_sts="finished")
                ex.update_check_finished(id=exp_id, finished="DONE")
                ex.update_status(id=exp_id, system_status="#")

                # count
                usr.update_infer_count(userid=user_id, infer_count=user["infer_count"]+1)

                # logging
                logmgmt.logInfo(exp_id, "Process End: set condition and create sample", step_all_log)
            else:
                ex.update_error_message(id=exp_id, error_message='Error  : サンプル生成されませんでした。制約条件を見直してください')
                ex.update_error_message_en(id=exp_id, error_message_en="Status: Failed sampling. Please reconsider the conditions")
                vlib.update_progress_status(exp_id, step2_num = 100, step2_sts="finished")
                ex.update_status(id=exp_id, system_status="#")
                # logging
                logmgmt.logInfo(exp_id, "Process Failed: sampling", step_all_log)

        except Exception as e:
            vlib.update_progress_status(exp_id, step2_num=100, step2_sts="finished")
            vlib.update_error(exp_id, str(e), "step3")
        finally:
            print('Fin\n')

