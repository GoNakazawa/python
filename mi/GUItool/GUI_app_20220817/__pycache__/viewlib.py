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
from tybmilib import prep
from tybmilib import vis
from tybmilib import datamgmt
from tybmilib import modeling
from tybmilib import paramsearch
from tybmilib import chemembeding
from tybmilib.logmgmt import create_logger, stop_watch, logger
from tybmilib import myfilename as mfn
from tool_app.mymongo import Experiments, Users, mongo_users, mongo_users_check, init_dict

role = "arn:aws:iam::375869297825:role/for_app_dev2"

# Step1:データ前処理
class subThread_vis(threading.Thread):
    def __init__(self, df_reg, objectives, s3_bucket_path, exp_id):
        super(subThread_vis,self).__init__()
        self.stop_event = threading.Event()
        self._df_reg = df_reg
        self._objectives = objectives
        self._s3_bucket_path = s3_bucket_path
        self._exp_id = exp_id
    
    def stop(self):
        self.stop_event.set()
        
    def run(self):
        exp_id = self._exp_id
        df_reg = self._df_reg
        objectives = self._objectives
        s3_bucket_path = self._s3_bucket_path
        
        # logging

        create_logger(exp_id)
        print("test1")
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        # 可視化
        try:
            local_files = [mfn.get_scatter_filename(exp_id, "objectives"), mfn.get_scatter_filename(exp_id, "all"), mfn.get_correlation_filename(exp_id, "objectives"), mfn.get_correlation_filename(exp_id, "all"), mfn.get_profile_filename(exp_id)]
            for f in local_files:
                print(f)
                if os.path.isfile(f):
                    os.remove(f)
            print("test1")

            method=['profiles','pairplot','correlation_matrix']
            print(method)
            ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=5)

            # 可視化メソッドの呼び出し        
            for i in method:
                ex = Experiments()
                experiment = ex.find_one_id(exp_id)
                
                s = 'データ可視化中({}/3)'.format((method.index(i)))
                ex.update_status(id=exp_id, system_status=s)

                print(s)                
                vis.show_plot(df_reg, objectives, s3_bucket_path, exp_id, method=[i])

                # progress bar
                r = int((method.index(i)+1)/3*100)
                ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=r)

            # logging
            ex.update_error_message(id=exp_id, error_message='Status : データ可視化が実行されました')
        except Exception as e:
            ex.update_progress_rate_step1(id=exp_id, progress_rate_step1=0)
            ex.update_error_message(id=exp_id, error_message='Error  : '+str(e))
        finally:
            print('Fin\n')

# Step2;回帰モデル構築,評価
class subThread_modeling(threading.Thread):
    def __init__(self, exp_id, s3_bucket_path, problemtype, objectives, s3_uri_list, traindata_list, metrics):
        super(subThread_modeling,self).__init__()
        self.stop_event = threading.Event()
        self._exp_id = exp_id
        self._s3_bucket_path = s3_bucket_path
        self._problemtype = problemtype
        self._objectives = objectives
        self._s3_uri_list = s3_uri_list
        self._traindata_list = traindata_list
        self._metrics = metrics
    
    def stop(self):
        self.stop_event.set()
    
    def run(self):
        exp_id = self._exp_id
        s3_bucket_path = self._s3_bucket_path
        problemtype = self._problemtype
        objectives = self._objectives
        s3_uri_list = self._s3_uri_list
        traindata_list = self._traindata_list
        metrics = self._metrics
        
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
    
    # logging
        create_logger(exp_id)        
        # Modeling
        try:
            ex = Experiments()
            experiment = ex.find_one_id(exp_id)
            name = experiment['title']
            mlmodel = modeling.SagemakerCtrl(s3_bucket_path,role,exp_id,problemtype,metrics)
            ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=5)

            # モデル構築
            logger(exp_id).info('Process Start  : {}'.format('fit_multi_model'))
            logger(exp_id).info('In  : {}'.format([objectives,s3_uri_list]))
            print("---------------------")
            print(traindata_list)
            # 複数モデル実行
            model_dict = {}
            for j in range(len(objectives)):                
                experiment = ex.find_one_id(exp_id)
                print('目的変数: '+ str(objectives[j]))
                progress_msg = "モデル構築中({0}/{1})".format(j, str(len(objectives)))
                ex.update_status(id=exp_id, system_status=progress_msg)
                
                # モデル実行
                model_name = mlmodel.fit(objectives[j],s3_uri_list[j],s3_bucket_path,role,problemtype,j)
                model_dict[objectives[j]] = model_name
                
                r = int((j+1)/len(objectives)*33)
                ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=r)

            #setting
            ex.update_model_name(id=exp_id,model_name=model_dict)
            model_list = []
            for j in range(len(objectives)):
                model_n = model_dict[objectives[j]]
                model_list.append(model_n)

            # ロギング関連
            logger(exp_id).info('Out  : {}'.format(model_list))
            logger(exp_id).info('Process End  : {}'.format('fit_multi_model'))
            logger(exp_id).info('Process Start  : {}'.format('analyze_multi_model'))

            html_path_list = []
            for j in range(len(objectives)):                
                experiment = ex.find_one_id(exp_id)
                print('目的変数: '+ str(objectives[j]))
                progress_msg = "Shap値算出中({0}/{1})".format(j, str(len(objectives)))
                ex.update_status(id=exp_id, system_status=progress_msg)
                    
                path = mlmodel.analyze(objectives[j],s3_uri_list[j],traindata_list[j],model_list[j],s3_bucket_path,role,problemtype)
                html_path_list.append(path)
                r = int((j+1)/len(objectives)*33) + 33
                ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=r)

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
            
            # logging
            logger(exp_id).info('Process End  : {}'.format('analyze_multi_model'))
            logger(exp_id).info('Process Start  : {}'.format('estimate_multi_testdata'))

            # テストデータでの評価値
            Eval_value_list = []
            for j in range(len(objectives)):
                experiment = ex.find_one_id(exp_id)
                # ステータス情報更新
                progress_msg = "モデル評価指標算出中({0}/{1})".format(j, str(len(objectives)))
                ex.update_status(id=exp_id, system_status=progress_msg)

                region = boto3.Session().region_name
                Eval_value = mlmodel.estimate_testdata(objectives[j],model_list[j],traindata_list[j],s3_bucket_path,region,problemtype)
                Eval_value_list.append(Eval_value)
                
                # プログレスバー更新
                r = int((j+1)/len(objectives)*33) + 67
                ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=r)
            # logging
            logger(exp_id).info('Process End  : {}'.format('estimate_multi_testdata'))
            
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

            ex.update_error_message(id=exp_id, error_message='Status : モデル構築・評価指標算出が実行されました')
            ex.update_check_finished(id=exp_id, finished="DONE")

        except (modeling.Lib_ParseError,datamgmt.Lib_ParseError) as el:
            ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=0)
            ex.update_error_message(id=exp_id, error_message=str(el))
        except Exception as e:
            ex.update_progress_rate_step2(id=exp_id, progress_rate_step2=0)
            ex.update_error_message(id=exp_id, error_message='Error  : '+str(e))
        finally:
            print('Fin\n')



# Step3;サンプル生成
class subThread_sampling(threading.Thread):
    def __init__(self, exp_id, boundary_setting, objectives, model_list, number_of_samples, s3_bucket_path):
        super(subThread_sampling,self).__init__()
        self.stop_event = threading.Event()
        self._exp_id = exp_id
        self._boundary_setting = boundary_setting
        self._objectives = objectives
        self._number_of_samples = number_of_samples
        self._model_list = model_list
        self._s3_bucket_path = s3_bucket_path
    
    def stop(self):
        self.stop_event.set()
        
    def run(self):
        exp_id = self._exp_id
        boundary_setting = self._boundary_setting
        objectives = self._objectives
        number_of_samples = self._number_of_samples
        model_list = self._model_list
        s3_bucket_path = self._s3_bucket_path
        
        # logging
        create_logger(exp_id)
        
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)

        structure_mode = experiment["chem_type"]
        if structure_mode == "maccs" or structure_mode == "mfp":
            master_filename = mfn.get_master_filename(exp_id)
            df_master = prep.read_csv(master_filename, exp_id)

            radius = 0
            bit_num = 4096
            if structure_mode == "mfp":
                radius = int(experiment["radius"])
                if experiment["check_bitnum"] == "short":
                    bit_num = int(experiment["bit_num"])

            chem = chemembeding.Features(df_master, exp_id, structure_mode, radius, bit_num)
            mol_list = chem.get_smiles()
        else:
            chem = ""

        # sampling
        try:
            sb = paramsearch.Search_Boundary(experiment['problemtype'],exp_id,s3_bucket_path)
            samples, samples_predicted = sb.create_samples(boundary_setting,objectives,model_list,chemical_feature=chem,number_of_samples=number_of_samples)
            ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=25)

            # モデル呼び出し
            pre_list = []
            for j in range(len(objectives)):
                ex = Experiments()
                experiment = ex.find_one_id(exp_id)
                # ステータス情報更新
                progress_msg = 'サンプル生成中({0}/{1})'.format(j, str(len(objectives)))
                ex.update_status(id=exp_id, system_status=progress_msg)                
                    
                # 実行関数
                btr = modeling._AutopilotBatchjobRegressor(model_list[j],exp_id)
                pre_list.append(btr.predict(samples_predicted,s3_bucket_path))

                # プログレスバー更新
                r = int((j+1)/len(objectives)*75) + 25
                ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=r)
                
            pre_array = np.vstack(pre_list).T
            ys = pd.DataFrame(pre_array,index=samples.index,columns=objectives)
            samples = pd.concat([samples,ys], axis=1)

            # データの保存
            sample_filename = mfn.get_sample_filename(exp_id)
            samples.to_csv(sample_filename, index=False, sep=',')
            ex.update_error_message(id=exp_id, error_message='Status : サンプル生成が実行されました')
            ex.update_check_finished(id=exp_id, finished="DONE")

        except (paramsearch.Lib_ParseError,modeling.Lib_ParseError) as el:
            ex.update_error_message(id=exp_id, error_message=str(el))
            ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=0)
            logger(exp_id).info('Process Stop : {}'.format(el))
        except Exception as e:
            ex.update_error_message(id=exp_id, error_message='Error  : '+str(e))
            ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=0)
            logger(exp_id).info('Process Stop : {}'.format(e))
        finally:
            print('Fin\n')


# Step3;探索開始
class subThread_search(threading.Thread):
    def __init__(self, exp_id, model_list, objectives, problemtype, samples, objectives_target, method_list):
        super(subThread_search,self).__init__()
        self.stop_event = threading.Event()
        self._exp_id = exp_id
        self._model_list = model_list
        self._objectives = objectives
        self._problemtype = problemtype
        self._samples = samples
        self._objectives_target = objectives_target
        self._method_list = method_list
    
    def stop(self):
        self.stop_event.set()
        
    def run(self):
        exp_id = self._exp_id
        objectives = self._objectives
        problemtype = self._problemtype
        samples = self._samples
        objectives_target = self._objectives_target
        method_list = self._method_list
        model_list = self._model_list
        
        # logging
        create_logger(exp_id)
    
        ex = Experiments()
        experiment = ex.find_one_id(exp_id)
        
        try:
            # 探索実行            
            ps = paramsearch.Search(model_list,objectives,problemtype,exp_id)
            ps.search_samples(samples,objectives_target,method_list)

            ex.update_error_message(id=exp_id, error_message='Status : 探索が実行されました(画像の読み込みに時間がかかることがあります。少し待って更新して下さい。)')
        except paramsearch.Lib_ParseError as el:
            ex.update_error_message(id=exp_id, error_message=str(el))
            ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=0)
        except Exception as e:
            ex.update_error_message(id=exp_id, error_message='Error  : '+str(e))
            ex.update_progress_rate_step3(id=exp_id, progress_rate_step3=0)
        finally:
            print('Fin\n')
