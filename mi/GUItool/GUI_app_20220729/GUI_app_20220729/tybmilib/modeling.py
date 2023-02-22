# -*- coding: utf-8 -*-
"""SageMakerの呼び出しのmodule
@author: TOYOBO CO., LTD.

【説明】
SageMakerにアクセスし、モデル作成、作成後のモデルデプロイ、endpointの呼び出しに対応したモジュール

"""

# Import functions
import numpy as np
import pandas as pd
import multiprocessing
import os
import boto3
import sagemaker
import shutil
import configparser
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from sagemaker import clarify
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from time import gmtime, strftime, sleep
from contextlib import redirect_stdout
from bs4 import BeautifulSoup
from tybmilib import prep
from tybmilib import datamgmt
from tybmilib import logmgmt
from tybmilib import myfilename as mfn
import random
from botocore.config import Config

#------------------------------------------------------------
# Read local file `config.ini`.

Local_mode = mfn.get_localmode()

if Local_mode:
    from tqdm import tqdm # 通常
else:
    from tqdm import tqdm_notebook as tqdm # notebook用



class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするためのクラス
    """
    pass
    
class SagemakerCtrl:
    """SageMkaerへのアクセス管理を行うクラス
    
    モデル構築、評価値出力、Shap値算出他機能が実装
    """
    def __init__(self, bucket_name, role, experiment_ID, user_ID, problemtype, eval_metrics):
        """コンストラクタ

        Args:
            bucket_name(str): 1st argument
            role (str): 2nd argument
            experiment_ID (str): 3rd argument
            user_ID (str): 4th argument
            problemtype (str): 5th argument
            eval_metrics (str): 6th argument

        Returns:
            None
            
        """
        
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session(boto3.session.Session())
        self.bucket = bucket_name
        self.role = role
        self.experiment_ID = experiment_ID
        self.user_ID = user_ID
        self.problemtype = problemtype
        self.eval_metrics = eval_metrics
        

    def present_info(self,objectives, model_list):
        """Notebook用のkernel再起動時のグローバル関数リストを出力

        Args:
            objectives (list): 1st argument
            model_list (list): 2nd argument

        Returns:
            None
            
        """
        
        print('------------------------------')
        print('#=========【途中再起動した場合、別セルに貼り付け、実行】以下の情報は、次セクションでも利用します。=========')
        print('objectives = {}'.format(objectives))
        print('model_list = {}'.format(model_list))
        print(f'problemtype = {self.problemtype!r}')
        print(f'user_name = {self.user_ID!r}')
        print(f'experiment_ID = {self.experiment_ID!r}')


    def fit_multi_model(self, objectives, s3_uri_list):
        """モデル呼び出しを複数実行するためのラッパー関数

        Args:
            objectives (list): 1st argument
            s3_uri_list (list): 2nd argument

        Returns:
            list: model_list
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: fit_multi_model", step2_log)
        
        # 複数モデル実行用リスト
        model_list = []
        for j, obj in enumerate(objectives):
            print('目的変数：'+ str(obj))
            model_name = self.fit(obj, s3_uri_list[j], j)
            model_list.append(model_name)
            print('------------------------------')
            
        # ロギング関連
        logmgmt.logInfo(self.experiment_ID, "Process End: fit_multi_model", step2_log)
        return model_list


    def analyze_multi_model(self, objectives, s3_uri_list, traindata_path_list, model_list):
        """複数モデルでshap値出力のためのラッパー関数

        Args:
            objectives (list): 1st argument
            s3_uri_list (list): 2nd argument
            traindata_path_list (list): 3rd argument
            model_list (list): 4th argument

        Returns:
            None
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: analyze_multi_model", step2_log)
        
        print('------------------------------')
        process = tqdm(range(len(objectives)))
        
        html_path_list = []
        for j in process:
            process.set_description("Shap processing")
            path = self.analyze(objectives[j], s3_uri_list[j], traindata_path_list[j], model_list[j])
            html_path_list.append(path)

        # shapレポートの統合
        soup_list = []
        for html_path in html_path_list:
            with open(html_path) as f:
                soup_list.append(BeautifulSoup(f.read(), 'lxml'))
        pure_bound_html = ''.join([soup.prettify() for soup in soup_list])
        bound_html = pure_bound_html.replace('</html>\n<html>\n', '')

        save_path = mfn.get_shap_filename(self.experiment_ID, Local_mode=Local_mode)
        # 保存処理
        with open(save_path, mode='w') as f:
            f.write(bound_html)
                            
        # logging
        logmgmt.logInfo(self.experiment_ID, "Process End: analyze_multi_model", step2_log)


    def estimate_multi_testdata(self, objectives, model_list, traindata_path_list):
        """複数モデルで予測・実績値プロットを出力するためのラッパー関数

        Args:
            objectives (list): 1st argument
            model_list (list): 2nd argument
            traindata_path_list (list): 3rd argument

        Returns:
            list: Eval_value_list
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: estimate_multi_model", step2_log)
        
        Eval_value_list = []
        process = tqdm(range(len(objectives)))
        for j in process:
            process.set_description("Testdata processing")
            Eval_value = self.estimate_testdata(objectives[j], model_list[j], traindata_path_list[j])
            Eval_value_list.append(Eval_value)
        # 格納データ明示
        print('=========outputフォルダへの格納データ=========')
        for obj in objectives:
            if self.problemtype == 'Regression':
                print('テストデータとの比較結果: test_{}.png'.format(obj))
            elif self.problemtype == 'BinaryClassification':
                print('混合行列: Confusion_matrix_{}.png'.format(obj))
            elif self.problemtype == 'MulticlassClassification':
                print('混合行列: Confusion_matrix_{}.png'.format(obj))
        
        # logging
        logmgmt.logInfo(self.experiment_ID, "Process End: estimate_multi_model", step2_log)
        
        if Local_mode: 
            return Eval_value_list
        else:
            return None


    def estimate_multi_coefficients(self, objectives, traindata_path_list):
        """複数変数に対して、線形モデルでの回帰係数を出力するためのラッパー関数

        Args:
            objectives (list): 1st argument
            traindata_path_list (list): 2nd argument

        Returns:
            None
            
        """
        
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: estimate_multi_coefficients", step2_log)

        for j, obj in enumerate(objectives):
            self.estimate_coefficients(obj, traindata_path_list[j])
  
        # logging
        logmgmt.logInfo(self.experiment_ID, "Process End: estimate_multi_coefficients", step2_log)

        
    def fit(self, target, s3_traindata, objectives_number):
        """SageMakerでのモデル構築呼び出し

        Args:
            target (str): 1st argument
            s3_traindata (str): 2nd argument
            objectives_number (int): 3rd argument

        Returns:
            str: model_name
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Progress: create model on {}".format(target), step2_log)

        # session
        client = boto3.Session().client(service_name='sagemaker',config=Config(connect_timeout=60, read_timeout=60, retries={'max_attempts': 20}),region_name=self.region)

        # 設定
        s3_prefix = mfn.get_s3_modeling_path(self.user_ID, self.experiment_ID)
        bucket_name = mfn.get_modeling_s3_bucket()
        model_output_path = "s3://" + bucket_name + "/" + s3_prefix
        username_for_model = str(self.user_ID).replace("_", "-").replace(".", "-")

        # Model name
        model_name = 'ml-' + username_for_model +  '-' + str(objectives_number) +  '-' + self.experiment_ID
        # Job name
        auto_ml_job_name = 'job-' + username_for_model + '-' + datetime.now().strftime("%m%d%H%M%S")
        
        """
        # Endpoint name
        ep_name = 'ep-' + prefix_list[1] +  '-' + str(objectives_number) +  '-' + self.experiment_ID
        # Endpoint Config name
        ep_config_name = "ep-config-" + datetime.now().strftime("%m%d%H%M%S")
        """
        # 過去モデル削除
        try:
            client.delete_model(ModelName=model_name)
            logmgmt.logInfo(self.experiment_ID, "model名 {} の削除".format(model_name), step2_log)
        except Exception as e:
            pass
        
        # client
        input_data_config = [{
            'DataSource': {
               'S3DataSource': {
                   'S3DataType': 'S3Prefix',
                   'S3Uri': s3_traindata
               }
           },
           'CompressionType': 'None',
           'TargetAttributeName': target
        }]
        output_data_config = {'S3OutputPath': model_output_path}
        
        try:
            client.create_auto_ml_job(
                AutoMLJobName = auto_ml_job_name,
                InputDataConfig=input_data_config,
                OutputDataConfig=output_data_config,
                ProblemType=self.problemtype,
                AutoMLJobObjective={'MetricName':self.eval_metrics},
                AutoMLJobConfig={'CompletionCriteria': {'MaxCandidates': 30}},
                RoleArn=self.role)
        except Exception as e:
            print(str(e))
            error_msg = "Error: Model作成時にエラー発生したため、既存モデル数、SageMakerの設定を確認して下さい。"
            logmgmt.raiseError(self.experiment_ID, error_msg, step2_log)

        describe_response = client.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
        job_run_status = describe_response['AutoMLJobStatus']
        
        pbar = tqdm(total=5)
        status = 'Starting'
        while job_run_status not in ('Completed'): #'Failed', 'Completed', 'Stopped'
            describe_response = client.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
            job_run_status = describe_response['AutoMLJobStatus']
            job_secondary_status = describe_response['AutoMLJobSecondaryStatus']
            if job_secondary_status != 'Starting':
                if job_secondary_status != status:
                    pbar.update(1)
                status = describe_response['AutoMLJobSecondaryStatus']
                logmgmt.logDebug(self.experiment_ID, "Progress Status: {}".format(status), step2_log)
            sleep(60)
        pbar.close()
        
        best_candidate = client.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['BestCandidate']
        print("Metric: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
        print("Value: " + str(best_candidate['FinalAutoMLJobObjectiveMetric']['Value']))        
        
        # create sagemaker model
        client.create_model(ModelName=model_name, Containers=best_candidate['InferenceContainers'], ExecutionRoleArn=self.role)
        
        print('=========正常終了=========')
        print('モデル名: ' + model_name)
        
        # logging
        logmgmt.logDebug(self.experiment_ID, "Out: {}".format(model_name), step2_log)
        logmgmt.logInfo(self.experiment_ID, "Complete: create model on {}".format(target), step2_log)
        
        return model_name


    def analyze(self, target, s3_traindata, local_traindata, model_name, baseline_setting='mean'):
        """SageMakerでの対象モデルでのShap value算出機能呼び出し

        Args:
            target (str): 1st argument
            s3_traindata (str): 2nd argument
            local_traindata (str): 3rd argument
            model_name (str): 4th argument
            baseline_setting (str): 5th argument
            
        Returns:
            str: local_shaptg_filename
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Progress: analyze model on {}".format(target), step2_log)
        
        # S3関連
        s3_prefix = os.path.join(mfn.get_s3_modeling_path(self.user_ID, self.experiment_ID), "clarify-explainability/{}".format(target))
        explainability_output_path = "s3://" + self.bucket + "/" + s3_prefix
        instance_type = 'ml.m5.2xlarge'
        logmgmt.logDebug(self.experiment_ID, "bucket path: {}".format(explainability_output_path), step2_log)

        # Shapベースラインの設定
        df = prep.read_csv(local_traindata, self.experiment_ID)
        df_without_target = df.drop(columns=target)
        headers = df.columns.to_list()
        
        if baseline_setting == 'mean':
            l = []
            for col in df_without_target.columns:
                try:
                    c = df_without_target[col].mean()
                except Exception as e:
                    c = df_without_target[col].mode()[0]
                l.append(c)
        else:
            # autopilot再現
            l = ['' for col in df_without_target.columns]
        shap_baseline = [l]

        # 対象となるモデルの設定
        clarify_processor = clarify.SageMakerClarifyProcessor(role=self.role, instance_count=1, instance_type=instance_type, sagemaker_session=self.session)
        # Shap値算出用の設定
        model_config = clarify.ModelConfig(model_name=model_name, instance_type=instance_type, instance_count=1, accept_type='text/csv')
        shap_config = clarify.SHAPConfig(baseline=shap_baseline, num_samples=2048, agg_method='mean_abs', save_local_shap_values=True)
        explainability_data_config = clarify.DataConfig(s3_data_input_path=s3_traindata, s3_output_path=explainability_output_path, label=target, headers=headers, dataset_type='text/csv')
        
        try:
            with redirect_stdout(open(os.devnull, 'w')):
                # 説明性の計算
                clarify_processor.run_explainability(data_config=explainability_data_config, model_config=model_config, explainability_config=shap_config)
        except Exception as e:
            error_msg = "Error: Shap Value算出時にエラー発生したため、model格納先の指定Bucketを確認して下さい。"
            logmgmt.raiseError(self.experiment_ID, str(error_msg), step2_log)
                    
        # 文字列操作
        if '/' in target:
            target = target.replace('/', '')
        
        try:
            # レポートデータのダウンロード
            local_folder = mfn.get_csv_data_path(self.experiment_ID, Local_mode=Local_mode)
            s3_report_filename = mfn.get_s3_report_filename_prefix(s3_prefix)
            local_report_filename = datamgmt.download_file(self.bucket, s3_report_filename, local_folder)
            local_shaptg_filename = mfn.get_shaptg_filename(self.experiment_ID, target, Local_mode=Local_mode)
            os.rename(local_report_filename, local_shaptg_filename)

            s3_json_filename = mfn.get_s3_analysis_filename_prefix(s3_prefix)
            local_analysis_filename = datamgmt.download_file(self.bucket, s3_json_filename, local_folder)
            local_jsontg_filename = mfn.get_shapresult_target_filename(self.experiment_ID, target, Local_mode=Local_mode)
            os.rename(local_analysis_filename, local_jsontg_filename)

            # htmlの編集
            soup = BeautifulSoup(open(local_shaptg_filename), "lxml")
            pre_title = soup.h2.text
            object_title = pre_title.replace('label0', target)
            for i in soup.findAll('h2'):
                if i.text == pre_title:
                    i.string = object_title
            with open(local_shaptg_filename, mode = 'w', encoding = 'utf-8') as fw:
                fw.write(soup.prettify())

            print('=====目的変数:{} Shap値レポート====='.format(target))
            print(local_shaptg_filename)
        except Exception as e:
            logmgmt.raiseError(self.experiment_ID, 'Error : {}'.format(e), step2_log)
        
        # logging
        logmgmt.logDebug(self.experiment_ID, "Out: {}".format(s3_report_filename), step2_log)
        logmgmt.logInfo(self.experiment_ID, "Progress: analyze model on {}".format(target), step2_log)
        
        return local_shaptg_filename


    def estimate_testdata(self, target, model_name, local_traindata):
        """Autopilotで構築したモデルの予測結果と評価指標を表示

        Args:
            target (str): 1st argument
            model_name (str): 2nd argument
            local_traindata (str): 3rd argument
            
        Returns:
            dict: Eval_value
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Progress: estimate model on {}".format(target), step2_log)
        
        # データ設定
        df_test = pd.read_csv(local_traindata, sep=',')
        X_test = df_test.drop(columns=target)
        y_test = df_test[target]

        try:
            predictor = _AutopilotBatchjobRegressor(model_name=model_name, experiment_ID=self.experiment_ID, user_ID=self.user_ID, bucket_name=self.bucket, region_name=self.region)
            y_pre = predictor.predict(X_test, target)
            
            # 文字列操作
            if '/' in target:
                target = target.replace('/', '')    
            
            Eval_value = {} #追加
            # problemtypeに応じた評価指標の設定
            if self.problemtype == 'Regression':
                # 評価指標の計算
                R2 = r2_score(y_test.values, y_pre)
                MAE = mean_absolute_error(y_test.values, y_pre)
                MSE = mean_squared_error(y_test.values, y_pre)
                RMSE = np.sqrt(MSE)
                print('=====目的変数:{} デプロイモデルの性能評価====='.format(target))
                print('決定係数R2:','{:.2f}'.format(R2),
                    'MAE:','{:.2f}'.format(MAE),
                    'MSE:','{:.2f}'.format(MSE),
                    'RMSE:','{:.2f}'.format(RMSE)
                    )
                # 追加
                Eval_value['R2'] = R2
                Eval_value['MAE'] = MAE
                Eval_value['MSE'] = MSE
                Eval_value['RMSE'] = RMSE
            
            else:
                Accuracy = accuracy_score(y_test.values, y_pre)
                if self.problemtype == 'MulticlassClassification':
                    Precision = precision_score(y_test.values, y_pre, average='macro')
                    Recall = recall_score(y_test.values, y_pre, average='macro')
                    F1 = f1_score(y_test.values, y_pre, average='macro')
                else:
                    Precision = precision_score(y_test.values, y_pre)
                    Recall = recall_score(y_test.values, y_pre)
                    F1 = f1_score(y_test.values, y_pre)
                print('=====目的変数:{} デプロイモデルの性能評価====='.format(target))
                print('Accuracy:','{:.2f}'.format(Accuracy),
                    'Precision:','{:.2f}'.format(Precision),
                    'Recall:','{:.2f}'.format(Recall),
                    'F_score:','{:.2f}'.format(F1),
                    )
                # 追加
                Eval_value['Accuracy'] = Accuracy
                Eval_value['Precision'] = Precision
                Eval_value['Recall'] = Recall
                Eval_value['F_score'] = F1            

            # 実値と予測結果の差
            if self.problemtype == 'Regression':
                #print('=====実際の値と予測結果の比較=====')
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.scatter(y_test.values, y_pre,alpha=0.6)
                data_min = min(y_test.min(),y_pre.min())
                data_max = max(y_test.max(),y_pre.max())
                data_range = data_max-data_min
                plot_min = data_min - data_range*0.05
                plot_max = data_max + data_range*0.05
                ax.set_xlim(plot_min,plot_max)
                ax.set_ylim(plot_min,plot_max)
                ax.set_xlabel('(Actual)',size=18)
                ax.set_ylabel('(Prediction)',size=18)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=16)
                
                test_filename = mfn.get_test_filename(self.experiment_ID, target, Local_mode=Local_mode)
                if os.path.exists(test_filename):
                    os.remove(test_filename)
                plt.savefig(test_filename)
                plt.close()
                if Local_mode == True:
                    s3_prefix = mfn.get_s3_img_path(self.user_ID, self.experiment_ID)
                    s3_uri = datamgmt.upload_file(self.bucket, test_filename, s3_prefix)
                    logmgmt.logDebug(self.experiment_ID, "Out: {}".format(s3_uri), step2_log)
            # コンフュージョン行列                
            else:
                lst = y_test.values.tolist()
                df_class = list(set(lst))
                cm = confusion_matrix(y_test.values, y_pre)
                cm = pd.DataFrame(data=cm, index=df_class, columns=df_class)
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.set_ylim(len(cm), 0)
                sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
                plt.yticks(rotation=0)
                plt.xlabel("(Prediction)", fontsize=13, rotation=0)
                plt.ylabel("(Actual)", fontsize=13)
                
                confusion_filename = mfn.get_confusion_filename(self.experiment_ID, target, Local_mode=Local_mode)
                if os.path.exists(confusion_filename):
                    os.remove(confusion_filename)
                plt.savefig(confusion_filename)
                plt.close()
                if Local_mode == True:
                    # upload
                    s3_prefix = mfn.get_s3_img_path(self.user_ID, self.experiment_ID)
                    s3_uri = datamgmt.upload_file(self.bucket, confusion_filename, s3_prefix)
                    logmgmt.logDebug(self.experiment_ID, "Out: {}".format(s3_uri), step2_log)
            # logging
            logmgmt.logInfo(self.experiment_ID, "Complete: estimate model on {}".format(target), step2_log)

            return Eval_value

        except Exception as e:
            logmgmt.raiseError(self.experiment_ID, "Error: Process Stop", step2_log)

        
    def estimate_coefficients(self, target, local_traindata, var_lim=8):
        """線形モデルを採用した場合での回帰係数を出力

        Args:
            target (str): 1st argument
            local_traindata (str): 2nd argument
            var_lim (int): 3rd argument
            
        Returns:
            None
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Progress: estimate coefficients on {}".format(target), step2_log)
        
        # data設定
        df = prep.read_csv(local_traindata, self.experiment_ID)
        df = df.dropna(how='any')
        drop_col = df.select_dtypes(include=['object']).columns.to_list()
        df = df.drop(drop_col, axis=1)
        x = prep.drop_cols(df,[target],self.experiment_ID)
        y = df[target]
        
        # 線形モデルの構築
        lr = LinearRegression(normalize=True) # 線形モデルの定義 
        lr.fit(x.values,y.values)# 線形モデルの予測実行
        
        # 係数の取得
        Coef = pd.DataFrame({'Features':x.columns.to_list(),'Coefficients':lr.coef_.tolist()}).sort_values(by='Coefficients',ascending=True)
        Coef_pos = Coef[Coef['Coefficients']>=0] # 係数が正である説明変数を取得
        Coef_neg = Coef[Coef['Coefficients']<0]  # 係数が負である説明変数を取得
        nrm = max(abs(min(Coef["Coefficients"])), max(Coef["Coefficients"]))
        Coef_pos["Coefficients"] /= nrm
        Coef_neg["Coefficients"] /= nrm

        # (参考)線形モデルの性能確認    
        y_pred = lr.predict(x)
        R2 = r2_score(y, y_pred)
        MAE = mean_absolute_error(y, y_pred)
        MSE = mean_squared_error(y, y_pred)
        RMSE = np.sqrt(MSE)
        
        # (参考)線形モデルの性能確認
        print('====(参考)目的変数:{} 線形モデルの性能評価===='.format(target))
        print('決定係数R2: ','{:.2f}'.format(R2),'MAE: ','{:.2f}'.format(MAE),'MSE: ','{:.2f}'.format(MSE),'RMSE: ','{:.2f}'.format(RMSE))

        file_mode = ['all','limit']
        # 可視化
        for j in range(len(file_mode)):
            if j == 0:
                fig1, axes = plt.subplots(1, 1, figsize=(6,8))
                axes.barh(Coef_neg['Features'],Coef_neg['Coefficients'],color='steelblue')
                axes.barh(Coef_pos['Features'],Coef_pos['Coefficients'],color='lightcoral') 
            else:
                fig2, axes = plt.subplots(1, 1, figsize=(6,8))
                axes.barh(Coef_neg.iloc[:var_lim,:]['Features'],Coef_neg.iloc[:var_lim,:]['Coefficients'],color='steelblue')
                axes.barh(Coef_pos.iloc[len(Coef_pos)-var_lim:,:]['Features'],Coef_pos.iloc[len(Coef_pos)-var_lim:,:]['Coefficients'],color='lightcoral')
            axes.axvline(0, ls='--',color='black')
            if len(Coef_neg) == 0:
                axes.set_xlim(-abs(Coef_pos['Coefficients'].max()),Coef_pos['Coefficients'].max())                    
            elif len(Coef_pos) == 0:
                axes.set_xlim(-abs(Coef_neg['Coefficients'].min()),Coef_neg['Coefficients'].max())
            else:
                axes.set_xlim(-max(abs(Coef_neg['Coefficients'].min()),Coef_pos['Coefficients'].max()), max(abs(Coef_neg['Coefficients'].min()),Coef_pos['Coefficients'].max()))

            axes.set_xlabel('Coefficients',size=16)
            axes.set_ylabel('Features',size=16)
            title = "Coefficients({0})_{1}".format(file_mode[j], target)
            axes.set_title(title, size=16)
            axes.tick_params(labelsize=16)
            plt.subplots_adjust(left=0.4)
            plt.tick_params(labelsize=10)
            
            # 文字列操作
            if '/' in target:
                target = target.replace('/', '')
            
            # file
            coefficients_filename = mfn.get_coefficients_filename(self.experiment_ID, file_mode[j], target, Local_mode=Local_mode)
            if os.path.exists(coefficients_filename):
                os.remove(coefficients_filename)
            plt.savefig(coefficients_filename)
            plt.clf()
            plt.close()

            if Local_mode == True:
                # upload
                s3_prefix = mfn.get_s3_img_path(self.user_ID, self.experiment_ID)
                s3_uri = datamgmt.upload_file(self.bucket, coefficients_filename, s3_prefix)
                logmgmt.logDebug(self.experiment_ID, "Out: {}".format(s3_uri), step2_log)
        
        # データ格納
        print('=========outputフォルダへの格納データ=========')
        print('coef値グラフ(全変数/重要変数): visulize_linear_(coef_all/coef_importance)_' + str(target) + '.png')
            
        # logging
        logmgmt.logInfo(self.experiment_ID, "Complete: estimate coefficients on {}".format(target), step2_log)    


# Autopilot用バッチ推論
class _AutopilotBatchjobRegressor:
    """バッチ推論呼び出しクラス
    
    Sagemakerバッチ変換機能を利用するためのクラス
    """
    
    def __init__(self, model_name, experiment_ID, user_ID, bucket_name, region_name):
        """コンストラクタ

        Args:
            model_name (str): 1st argument
            experiment_ID (str): 2nd argument
            user_ID (str): 3rd argument
            bucket_name (str): 4th argument
            region_name (str): 5th argument
            
        Returns:
            None
            
        """
        self.model_name = model_name
        self.experiment_ID = experiment_ID
        self.user_ID = user_ID
        self.bucket = bucket_name
        self.region = region_name


    def predict(self, test_data, target, sampling=False):
        """モデル推論実行

        Args:
            test_data (pandas.DataFrame): 1st argument
            target (str): 2nd argument
            
        Returns:
            numpy.array: np.float(pre_array)
            
        """
        # logging
        step2_log = mfn.get_step2_modelbuilding_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: predict", step2_log)
        
        # 設定ファイル
        client = boto3.Session().client(service_name='sagemaker',config=Config(connect_timeout=60, read_timeout=60, retries={'max_attempts': 20}),region_name=self.region)
        timestamp_suffix = strftime("%Y%m%d%H%M%S", gmtime())
        s3_prefix = os.path.join(mfn.get_s3_modeling_path(self.user_ID, self.experiment_ID), "inference-results/{}".format(target))        
        s3_output_path = "s3://" + self.bucket + "/" + s3_prefix


        suf = str(random.randint(0, 100000))
        file_suffix = timestamp_suffix + suf
        if sampling == True:
            target = target + "sampling"
        output_data = "temp_{}_{}.csv".format(target, file_suffix)
        output_temp_filename = mfn.define_csv_data_path(self.experiment_ID, output_data, Local_mode=Local_mode)
        test_data.to_csv(output_temp_filename, header=None, index=False, sep=',')
        data_s3_path = datamgmt.upload_file(self.bucket, output_temp_filename, s3_prefix)
        os.remove(output_temp_filename)

        transform_job_name = 'automl-transform-' + file_suffix
        transform_input = {
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': data_s3_path
                }
            },
            'ContentType': 'text/csv',
            'CompressionType': 'None',
            'SplitType': 'Line'
        }
        
        transform_output = {
            'S3OutputPath': s3_output_path,
        }
        
        transform_resources = {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1
        }
        
        client.create_transform_job(TransformJobName = transform_job_name,
                                    ModelName = self.model_name,
                                    TransformInput = transform_input,
                                    TransformOutput = transform_output,
                                    TransformResources = transform_resources
                                   )

        # 完了するまで待つ
        describe_response = client.describe_transform_job(TransformJobName = transform_job_name)
        job_run_status = describe_response['TransformJobStatus']
        while job_run_status not in ('Failed', 'Completed', 'Stopped'):
            describe_response = client.describe_transform_job(TransformJobName = transform_job_name)
            job_run_status = describe_response['TransformJobStatus']
            logmgmt.logDebug(self.experiment_ID, "Process Status: {}".format(job_run_status), step2_log)
            sleep(60)
       
        if job_run_status == 'Completed':
            temp_prefix = os.path.join(s3_prefix, output_data+'.out') 
            # メモリ上にデータダウンロード
            logmgmt.logDebug(self.experiment_ID, "Out: {}".format(s3_output_path), step2_log)

            s3 = boto3.client('s3')
            body = s3.get_object(Bucket=self.bucket, Key=temp_prefix)['Body'].read()
            predictions = ','.join(body.decode('utf-8').splitlines())
            pre = predictions.split(',')
            pre_float = list(map(float, pre))
            
            # logging
            logmgmt.logInfo(self.experiment_ID, "Process End: predict", step2_log)
            return np.array(pre_float)


# Autopilot用API
class _AutopilotRegressor:
    """endpoint呼び出しクラス
    
    scikit-learnライクにendpoint呼び出しを行うための機能
    """
    
    def __init__(self,ep_name,region_name=None,progress_bar=False):
        """コンストラクタ

        Args:
            ep_name (str): 1st argument
            region_name (str): 2nd argument
            progress_bar (boolean): 3rd argument
            
        Returns:
            None
            
        """
        
        self.ep_name = ep_name
        if region_name is None:
            self.sm_rt = boto3.Session().client('runtime.sagemaker')
        else:
            self.sm_rt = boto3.Session().client('runtime.sagemaker',region_name=region_name)
        self.progress_bar = progress_bar
            
    def predict(self,X):
        """入力データに対して回帰モデルでの推論結果を出力

        Args:
            X (pd.DataFrame or np.array (2D) or list (2D)): 1st argument
            
        Returns:
            np.array: pre_float
            
        """
        '''
        # logging
        logger().info('Process Start  : {}'.format('predict'))
        logger().debug('In  : {}'.format([X]))
        '''
        # 推論用のBodyを作る
        dfX = pd.DataFrame(X) # convert in case of np.array or list
                 
        lines = [','.join(list(map(str, dfX.iloc[idx,:].values))) for idx in dfX.index]
        
        # エンドポイントで推論
        if self.progress_bar is True:
            iterator = tqdm(lines)
        else:
            iterator = lines
        
        pre = []
        for x_str in iterator:
            response = self.sm_rt.invoke_endpoint(
                EndpointName=self.ep_name, 
                ContentType='text/csv', 
                Accept='text/csv', 
                Body=x_str
            )
            pre.append(response['Body'].read().decode("utf-8"))

        pre_float = list(map(float, pre))
        
        '''
        # logging
        logger().debug('Out  : {}'.format([np.array(pre_float)]))
        logger().info('Process End  : {}'.format('predict'))
        '''
        return np.array(pre_float)

    def predict_proba(self,X):
        """入力データに対して分類モデルでの推論結果を出力

        Args:
            X (pd.DataFrame or np.array (2D) or list (2D)): 1st argument
            
        Returns:
            np.array: pre_float
            
        """
        '''
        # logging
        logger().info('Process Start  : {}'.format('predict_proba'))
        logger().debug('In  : {}'.format([X]))
        '''
        # 推論用のBodyを作る
        dfX = pd.DataFrame(X) # convert in case of np.array or list
          
        lines = [','.join(list(map(str, dfX.iloc[idx,:].values))) for idx in dfX.index]
        
        # エンドポイントで推論
        if self.progress_bar is True:
            iterator = tqdm(lines)
        else:
            iterator = lines
        
        pre = []
        for x_str in iterator:
            response = self.sm_rt.invoke_endpoint(
                EndpointName=self.ep_name, 
                ContentType='text/csv', 
                Accept='text/csv', 
                Body=x_str
            )
            pre.append(response['Body'].read().decode("utf-8"))        

        pre_float = list(map(float, pre))
        
        '''
        # logging
        logger().debug('Out  : {}'.format([np.array(pre_float)]))
        logger().info('Process End  : {}'.format('predict_proba'))
        '''
        return np.array(pre_float)

# 内部関数
class _AutopilotMultiprocessRegressor:
    """endpoint呼び出しクラス
    
    scikit-learnライクにendpoint呼び出しを行うための機能
    (APIコールを並列処理にて実行)
    
    """
    
    def __init__(self,ep_name,region_name=None,progress_bar=False):
        """コンストラクタ

        Args:
            ep_name (str): 1st argument
            region_name (str): 2nd argument
            progress_bar (boolean): 3rd argument
            
        Returns:
            None
            
        """
        
        self.ep_name = ep_name
        self.region_name = region_name
        self.progress_bar = progress_bar
            
    def predict(self,X):
        """入力データに対して分類モデルでの推論結果を出力

        Args:
            X (pd.DataFrame or np.array (2D) or list (2D)): 1st argument
            
        Returns:
            np.array: pre_float
            
        """
        
        # 推論用のBodyを作る
        dfX = pd.DataFrame(X) # convert in case of np.array or list
        
        # CSVファイルを経由してBodyを作る場合
        dfX.to_csv("temp_for_prediction.csv",index=False)
        with open("temp_for_prediction.csv") as f:
            lines = f.readlines(delimiter=',')[1:]

        args_list = []
        for line in lines:
            args_list.append((self.ep_name,self.region_name, line))
            
        with multiprocessing.Pool(None) as pool:
            if self.progress_bar is True:
                result = [float(var) for var in tqdm(pool.imap(self.call_endpoint_for_multiprocess, args_list),total=len(args_list))]
            else:
                result = [float(var) for var in pool.imap(self.call_endpoint_for_multiprocess, args_list)]
            
        os.remove("temp_for_prediction.csv")
        return np.array(result)

    def predict_proba(self,X):
        """入力データに対して分類モデルでの推論結果を出力

        Args:
            X (pd.DataFrame or np.array (2D) or list (2D)): 1st argument
            
        Returns:
            np.array: pre_float
            
        """
        
        # 推論用のBodyを作る
        dfX = pd.DataFrame(X) # convert in case of np.array or list
        
        # method A: CSVファイルを経由してBodyを作る場合
        dfX.to_csv("temp_for_prediction.csv",index=False)
        with open("temp_for_prediction.csv") as f:
            lines = f.readlines(delimiter=',')[1:]
        
        args_list = []
        for line in lines:
            args_list.append((self.ep_name,self.region_name, line))
            
        with multiprocessing.Pool(None) as pool:
            if self.progress_bar is True:
                result = [float(var) for var in tqdm(pool.imap(self.call_endpoint_for_multiprocess, args_list),total=len(args_list))]
            else:
                result = [float(var) for var in pool.imap(self.call_endpoint_for_multiprocess, args_list)]
            
        os.remove("temp_for_prediction.csv")
        return np.array(result)
        
    @staticmethod
    def call_endpoint_for_multiprocess(args):
        """並列処理実行のためのラッパー関数

        Args:
            args: 1st argument
            
        Returns:
            str: response['Body'].read().decode("utf-8")
            
        """
        
        # エンドポイント呼び出し関数
        ep_name, region_name, x_str = args

        if region_name is None:
            response = boto3.Session().client('runtime.sagemaker').invoke_endpoint(
                EndpointName=ep_name, 
                ContentType='text/csv', 
                Accept='text/csv', 
                Body=x_str
            )
        else:
            response = boto3.Session().client('runtime.sagemaker',region_name=region_name).invoke_endpoint(
                EndpointName=ep_name, 
                ContentType='text/csv', 
                Accept='text/csv', 
                Body=x_str
            )
        return response['Body'].read().decode("utf-8")

#------------------------------------------------------------
if __name__ == '__main__':
    None
