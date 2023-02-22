# -*- coding: utf-8 -*-
"""
@author: TOYOBO CO., LTD.
"""
# Import functions
import numpy as np
import pandas as pd
import importlib
import multiprocessing
import os
import sys
import boto3
import botocore
import sagemaker
import shutil
import seaborn as sns
import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sagemaker import AutoML
from datetime import datetime
from scipy.special import expit
from matplotlib import cm
from sagemaker import clarify
from sagemaker.local import LocalSession
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from time import gmtime, strftime, sleep
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from contextlib import redirect_stdout
from tybmilib import prep
from tybmilib import datamgmt
from tybmilib import modeling
from tybmilib.logmgmt import logger, stop_watch

#------------------------------------------------------------
class SagemakerCtrl:
    def __init__(self,prefix,role,experiment_ID,problemtype,metrics):
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
        self.prefix = prefix
        self.role = role
        self.problemtype = problemtype
        self.metrics = metrics
        self.experiment_ID = experiment_ID
        
    def present_info(self,objectives,ep_list,model_list,s3_bucket_path):
        print('------------------------------')
        print('#=========【途中再起動した場合、別セルに貼り付け、実行】以下の情報は、次セクションでも利用します。=========')
        print('objectives = {}'.format(objectives))
        print('ep_list = {}'.format(ep_list))
        print('model_list = {}'.format(model_list))
        print(f's3_bucket_path = {s3_bucket_path!r}')
        print(f'problemtype = {self.problemtype!r}')
        
    def fit_multi_model(self,objectives,s3_uri_list):
        logger().info('Process Start  : {}'.format('fit_multi_model'))
        logger().debug('In  : {}'.format([objectives,s3_uri_list]))
        
        # 複数モデル実行用リスト
        model_list = []
        endpoint_list = []
        for j in range(len(objectives)):
            print('目的変数：'+ str(objectives[j]))
            model_name,ep_name = self.fit(objectives[j],s3_uri_list[j],self.prefix,self.role,self.problemtype,j)
            model_list.append(model_name)
            endpoint_list.append(ep_name)
            print('------------------------------')
            
        # ロギング関連
        logger().debug('Out  : {}'.format([model_list,endpoint_list]))
        logger().info('Process End  : {}'.format('fit_multi_model'))
        return model_list,endpoint_list

    def analyze_multi_model(self,objectives,s3_uri_list,traindata_path_list,model_list):
        logger().info('Process Start  : {}'.format('analyze_multi_model'))
        logger().debug('In  : {}'.format([objectives,s3_uri_list,traindata_path_list,model_list]))
        
        print('------------------------------')
        process = tqdm(range(len(objectives)))
        for j in process:
            process.set_description("Shap processing")
            self.analyze(objectives[j],s3_uri_list[j],traindata_path_list[j],model_list[j],self.prefix,self.role,self.problemtype)
        
        # logging
        logger().info('Process End  : {}'.format('analyze_multi_model'))
        return None

    def estimate_multi_testdata(self,objectives,ep_list,traindata_path_list):
        # logging
        logger().info('Process Start  : {}'.format('estimate_multi_testdata'))
        logger().debug('In  : {}'.format([objectives,ep_list,traindata_path_list]))
        
        print('------------------------------')
        for j in range(len(objectives)):
            self.estimate_testdata(objectives[j],ep_list[j],traindata_path_list[j],self.region,self.problemtype)
        
        # 格納データ明示
        print('=========outputフォルダへの格納データ=========')
        for j in range(len(objectives)):
            if self.problemtype == 'Regression':
                print('テストデータとの比較結果：test_' + str(objectives[j]) + '.png')
            elif self.problemtype == 'BinaryClassification':
                print('混合行列：Confusion_matrix_' + str(objectives[j]) + '.png')
                #print('ROC曲線：ROC_curve_' + str(objectives[j]) + '.png')
            elif self.problemtype == 'MulticlassClassification':
                print('混合行列：Confusion_matrix_' + str(objectives[j]) + '.png')
        
        # logging
        logger().info('Process End  : {}'.format('estimate_multi_testdata'))
        return None
                
    def estimate_multi_coefficients(self,objectives,traindata_path_list):
        # logging
        logger().info('Process Start  : {}'.format('estimate_multi_coefficients'))
        logger().debug('In  : {}'.format([objectives,traindata_path_list]))
        
        print('------------------------------')
        for j in range(len(objectives)):
            self.estimate_coefficients(objectives[j],traindata_path_list[j],self.problemtype)

        # logging
        logger().info('Process End  : {}'.format('estimate_multi_coefficients'))
        return None
        
    def fit(self,target:'str',traindata:'str',prefix:'str',role:'str',problemtype:'str',objectives_number:'int'):
        # logging関連
        logger().info('Process Start  : {}'.format('fit'))
        logger().debug('In  : {}'.format([target,traindata,prefix,role,problemtype,objectives_number]))
        
        # 設定
        bucket = self.session.default_bucket()
        prefix_list = prefix.split('/')
        
        # Output path
        output_path = 's3://{}/{}/output/'.format(bucket, prefix)
        
        # Model name
        model_name = 'ml-' + prefix_list[1] +  '-' + str(objectives_number) +  '-' + self.experiment_ID
        
        # Job name
        auto_ml_job_name = 'job-' + prefix_list[1] + '-' + datetime.now().strftime("%m%d%H%M%S")
        
        # Endpoint name
        ep_name = 'ep-' + prefix_list[1] +  '-' + str(objectives_number) +  '-' + self.experiment_ID
        
        # Endpoint Config name
        ep_config_name = "ep-config-" + datetime.now().strftime("%m%d%H%M%S")
                
        # session
        client = boto3.Session().client(service_name='sagemaker',region_name=self.region)
        InstanceType = 'ml.m5.large'
        #,ml.t2.medium

        # 過去モデル削除
        try:
            logger().info('Try  : endpoint名{}の削除'.format(ep_name))
            response = client.delete_endpoint(EndpointName=ep_name)
        except Exception as e:
            logger().info('Result  : endpoint名{}の削除失敗'.format(ep_name))
            None
        
        try:
            logger().info('Try  : model名{}の削除'.format(model_name))
            response = client.delete_model(ModelName=model_name)
        except Exception as e:
            logger().info('Result  : model名{}の削除失敗'.format(model_name))
            None
            
        # client
        input_data_config = [{
            'DataSource': {
               'S3DataSource': {
                   'S3DataType': 'S3Prefix',
                   'S3Uri': traindata
               }
           },
           'CompressionType': 'None',
           'TargetAttributeName': target
        }]
        
        output_data_config = {'S3OutputPath':output_path}
        
        client.create_auto_ml_job(
            AutoMLJobName = auto_ml_job_name,
            InputDataConfig=input_data_config,
            OutputDataConfig=output_data_config,
            ProblemType=problemtype,
            AutoMLJobObjective={'MetricName':self.metrics},
            AutoMLJobConfig={'CompletionCriteria': {'MaxCandidates': 30}},
            RoleArn=role)
        
        #print ('JobStatus - Secondary Status')
        #print('------------------------------')
        
        describe_response = client.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)
        #print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
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
                    #print (describe_response['AutoMLJobStatus'] + " - " + describe_response['AutoMLJobSecondaryStatus'])
                status = describe_response['AutoMLJobSecondaryStatus']
                logger().info('Process Status  : {}'.format(status))
            sleep(60)
        pbar.close()
        
        best_candidate = client.describe_auto_ml_job(AutoMLJobName=auto_ml_job_name)['BestCandidate']
        best_candidate_name = best_candidate['CandidateName']
        #print("CandidateName: " + best_candidate_name)
        print("Metric: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
        print("Value: " + str(best_candidate['FinalAutoMLJobObjectiveMetric']['Value']))
        
        # create sagemaker model
        create_model_api_response = client.create_model(
            ModelName=model_name,
            Containers=best_candidate['InferenceContainers'],
            ExecutionRoleArn=role
        )
        #print ("create_model API response", create_model_api_response)
        
        # create sagemaker endpoint config
        create_endpoint_config_api_response = client.create_endpoint_config(
            EndpointConfigName=ep_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'tybmilib',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': InstanceType
                },
            ]
        )
        #print ("create_endpoint_config API response", create_endpoint_config_api_response)
        
        # create sagemaker endpoint
        create_endpoint_api_response = client.create_endpoint(
            EndpointName=ep_name,
            EndpointConfigName=ep_config_name,
        )
        #print ("create_endpoint API response", create_endpoint_api_response)
        
        print('=========正常終了=========')
        print('モデル名：' + model_name)
        print('エンドポイント名：' + ep_name)
        
        # logging
        logger().debug('Out  : {}'.format([model_name,ep_name]))
        logger().info('Process End  : {}'.format('fit'))
        
        return model_name,ep_name
    
    def analyze(self,target:'str',traindata:'str',local_traindata:'str',model_name:'str',prefix:'str',role:'str',problemtype:'str',baseline_setting='mean'):
        # logging
        logger().info('Process Start  : {}'.format('analyze'))
        logger().debug('In  : {}'.format([target,traindata,local_traindata,model_name,prefix,role,problemtype,baseline_setting]))
        
        # S3関連
        bucket = self.session.default_bucket()
        explainability_output_path = 's3://{}/{}/output/clarify-explainability'.format(bucket, prefix)
        instance_type = 'ml.m5.2xlarge'
        Filename = 'report.html'
        
        # Shapベースラインの設定
        df = prep.read_csv(local_traindata.replace('.csv', ''))
        df_without_target = df.drop(columns=target)
        headers = df.columns.to_list()
        
        if baseline_setting == 'mean':
            l = []
            for j in df_without_target.columns.tolist():
                try:
                    c = df_without_target[j].mean()
                except Exception as e:
                    c = df_without_target[j].mode()[0]
                l.append(c)
        else:
            # autopilot再現
            l = ['' for j in df_without_target.columns.tolist()]
        shap_baseline = [l]
        
        # 対象となるモデルの設定
        clarify_processor = clarify.SageMakerClarifyProcessor(role=role,
                                                               instance_count=1,
                                                               instance_type=instance_type,
                                                               sagemaker_session=self.session
                                                              )
        
        model_config = clarify.ModelConfig(model_name=model_name,
                                            instance_type=instance_type,
                                            instance_count=1,
                                            accept_type='text/csv'
                                          )
        
        if problemtype == 'Regression':
            predictions_config = clarify.ModelPredictedLabelConfig()
        else:
            predictions_config = clarify.ModelPredictedLabelConfig(probability_threshold=0.8)

        shap_config = clarify.SHAPConfig(baseline=shap_baseline,
                                          num_samples=2048,
                                          agg_method='mean_abs',
                                          save_local_shap_values=True
                                        )
        
        '''
        # autopilot再現
        shap_config = clarify.SHAPConfig(baseline=shap_baseline,
                                          num_samples=50,
                                          agg_method='mean_abs',
                                          save_local_shap_values=True
                                        )
        '''
        
        # 説明に利用するデータ
        explainability_data_config = clarify.DataConfig(
               s3_data_input_path=traindata,
               s3_output_path=explainability_output_path,
               label=target,
               headers=headers,
               dataset_type='text/csv'
            )
        
        with redirect_stdout(open(os.devnull, 'w')):
            # 説明性の計算
            clarify_processor.run_explainability(
               data_config=explainability_data_config,
               model_config=model_config,
               explainability_config=shap_config
            )
            
        # レポートデータのダウンロード
        report_key = explainability_output_path + '/' + Filename
        report_path_list = datamgmt.S3Dao().download_data([report_key])
        
        # shapレポートのダウンロード
        report_local_path = report_path_list[0]
        idx = report_local_path.find('.html')
        
        # output用ディレクトリの追加
        path = os.getcwd()
        new_path = 'output' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        
        # 文字列操作
        if '/' in target:
            target = target.replace('/', '')
        
        rename = report_local_path[:idx] + '_' + target + '.html'
        os.rename(report_local_path,rename)
        pre_name = path+'/output/report_'+target+'.html'
        try:
            os.remove(pre_name)
        except Exception as e:
            None
        shutil.move(rename,path+'/output')
        
        print('=====目的変数:{} Shap値レポート====='.format(target))
        print(path+'/output/report_'+target+'.html')
        
        # モデル削除
        #self.session.delete_model(model_name)
        
        # logging
        logger().info('Process End  : {}'.format('analyze'))
        
        return None
    
    def estimate_testdata(self,target:'str',ep_name:'str',local_traindata:'str',region:'str',problemtype:'str'):
        """
        新規テストデータの実験結果に基づき、Autopilotで構築したモデルの予測結果と評価指標を表示する関数
        """
        # logging
        logger().info('Process Start  : {}'.format('estimate_testdata'))
        logger().debug('In  : {}'.format([target,ep_name,local_traindata,region,problemtype]))
        
        # データ設定
        df_test = pd.read_csv(local_traindata, sep=',')
        X_test = df_test.drop(columns=target)
        y_test = df_test[target]
        
        # problemtypeに応じたmodel設定
        if problemtype == 'Regression':
            regressor = modeling._AutopilotRegressor(ep_name=ep_name, region_name=region, progress_bar=False)
            y_pre = regressor.predict(X_test)
            #pd.DataFrame(y_pre, columns=['predictions']).to_csv('prediction.csv')
        else:
            classifier = modeling._AutopilotRegressor(ep_name=ep_name, region_name=region, progress_bar=False)
            y_pre = classifier.predict_proba(X_test)
            
        # 文字列操作
        if '/' in target:
            target = target.replace('/', '')    
        
        # 保存先指定
        new_path = 'output' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)

        # problemtypeに応じた評価指標の設定
        if problemtype == 'Regression':
            # 評価指標の計算
            R2 = r2_score(y_test.values, y_pre)
            MAE = mean_absolute_error(y_test.values, y_pre)
            MSE = mean_squared_error(y_test.values, y_pre)
            RMSE = np.sqrt(MSE)
            print('=====目的変数:{} デプロイモデルの性能評価====='.format(target))
            print('決定係数R2：','{:.2f}'.format(R2),
                  'MAE：','{:.2f}'.format(MAE),
                  'MSE：','{:.2f}'.format(MSE),
                  'RMSE：','{:.2f}'.format(RMSE)
                 )
        elif problemtype == 'MulticlassClassification':
            Accuracy = accuracy_score(y_test.values, y_pre)
            Precision = precision_score(y_test.values, y_pre, average='macro')
            Recall = recall_score(y_test.values, y_pre, average='macro')
            F1 = f1_score(y_test.values, y_pre, average='macro')
            print('=====目的変数:{} デプロイモデルの性能評価====='.format(target))
            print('Accuracy：','{:.2f}'.format(Accuracy),
                  'Precision：','{:.2f}'.format(Precision),
                  'Recall：','{:.2f}'.format(Recall),
                  'F-score：','{:.2f}'.format(F1),
                 )
        else:
            Accuracy = accuracy_score(y_test.values, y_pre)
            Precision = precision_score(y_test.values, y_pre)
            Recall = recall_score(y_test.values, y_pre)
            F1 = f1_score(y_test.values, y_pre)
            #fpr, tpr, _ = roc_curve(y_test.values,classifier.predict_proba(X_test)[:, 1])
            #AUC = auc(fpr, tpr)
            print('=====目的変数:{} デプロイモデルの性能評価====='.format(target))
            print('Accuracy：','{:.2f}'.format(Accuracy),
                  'Precision：','{:.2f}'.format(Precision),
                  'Recall：','{:.2f}'.format(Recall),
                  'F-score：','{:.2f}'.format(F1),
                  #'AUC：','{:.2f}'.format(AUC)
                 )
        # 可視化
        if problemtype == 'Regression':
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
            plt.savefig('test_'+str(target)+'.png')
            plt.close()
        else:
            lst = y_test.values.tolist()
            df_class = list(set(lst))
            classes = [i for i in range(len(df_class))]
            cm = confusion_matrix(y_test.values, y_pre)
            cm = pd.DataFrame(data=cm, index=df_class, columns=df_class)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.set_ylim(len(cm), 0)
            sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
            plt.yticks(rotation=0)
            plt.xlabel("(Prediction)", fontsize=13, rotation=0)
            plt.ylabel("(Actual)", fontsize=13)
            plt.savefig('Confusion_matrix_'+str(target)+'.png')
            plt.close()
        '''
        # ROC曲線作成
        if problemtype == 'BinaryClassification':
            plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
            plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, 1.0])
            plt.savefig('ROC_curve_'+str(target)+'.png')
            plt.close()
        '''
        os.chdir(path)
        
        # logging
        logger().info('Process End  : {}'.format('estimate_testdata'))
        
        return None
        
    def estimate_coefficients(self,target:'str',local_traindata:'str',problemtype:'str',var_lim=8):
        # logging
        logger().info('Process Start  : {}'.format('estimate_coefficients'))
        logger().debug('In  : {}'.format([target,local_traindata,problemtype,var_lim]))
        
        # data設定
        df = prep.read_csv(local_traindata.replace('.csv', ''))
        df = df.dropna(how='any')
        drop_col = df.select_dtypes(include=['object']).columns.to_list()
        df = df.drop(drop_col, axis=1)
        x = prep.drop_cols(df,[target])
        y = df[target]
        
        # 回帰係数算出
        if problemtype == 'Regression':
            # 線形モデルの構築
            lr = LinearRegression(normalize=True) # 線形モデルの定義 
            lr.fit(x.values,y.values)# 線形モデルの予測実行
            
            # 係数の取得
            Coef = pd.DataFrame({'Features':x.columns.to_list(),'Coefficients':lr.coef_.tolist()}).sort_values(by='Coefficients',ascending=True)
            Coef_pos = Coef[Coef['Coefficients']>=0] # 係数が正である説明変数を取得
            Coef_neg = Coef[Coef['Coefficients']<0]  # 係数が負である説明変数を取得
            
            # (参考)線形モデルの性能確認    
            y_pred = lr.predict(x)
            R2 = r2_score(y, y_pred)
            MAE = mean_absolute_error(y, y_pred)
            MSE = mean_squared_error(y, y_pred)
            RMSE = np.sqrt(MSE)
            
            # (参考)線形モデルの性能確認
            print('====(参考)目的変数:{} 線形モデルの性能評価===='.format(target))
            print('決定係数R2：','{:.2f}'.format(R2),'MAE：','{:.2f}'.format(MAE),'MSE：','{:.2f}'.format(MSE),'RMSE：','{:.2f}'.format(RMSE))
    
            file_name_list = ['Cofficients(All)_{}','Cofficients(Limit Features)_{}']
            # 可視化
            for j in range(len(file_name_list)):
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
                    axes.set_xlim(-abs(Coef_pos['Coefficients'].max())-5,Coef_pos['Coefficients'].max()+5)                    
                elif len(Coef_pos) == 0:
                    axes.set_xlim(-abs(Coef_neg['Coefficients'].min())-5,Coef_neg['Coefficients'].max()+5)
                else:
                    axes.set_xlim(-max(abs(Coef_neg['Coefficients'].min()),Coef_pos['Coefficients'].max())-5, max(abs(Coef_neg['Coefficients'].min()),Coef_pos['Coefficients'].max())+5)
                axes.set_xlabel('Coefficients',size=16)
                axes.set_ylabel('Features',size=16)
                axes.set_title(file_name_list[j].format(target),size=16)
                plt.subplots_adjust(left=0.4)
                
                # 文字列操作
                if '/' in target:
                    target = target.replace('/', '')

                # 保存先指定
                new_path = 'output' #フォルダ名
                if not os.path.exists(new_path):#ディレクトリがなかったら
                    os.mkdir(new_path)#作成したいフォルダ名を作成
                path = os.getcwd()
                os.chdir(path + '/' + new_path)
                
                plt.savefig(file_name_list[j].format(target))
                plt.close()
                
                os.chdir(path)
                #Coef_pos.iloc[len(Coef_pos)-var_lim:,:]
                
            # データ格納
            print('=========outputフォルダへの格納データ=========')
            print('coef値グラフ(全変数/重要変数)：visulize_linear_(coef_all/coef_importance)_' + str(target) + '.png')
        else:
            print('設定されたproblemtypeでは、回帰係数を求めることが出来ません。')
            
        # logging
        logger().info('Process End  : {}'.format('estimate_coefficients'))
        
        return None
        
# Autopilot用API
class _AutopilotRegressor:
    def __init__(self,ep_name,region_name=None,progress_bar=False):
        self.ep_name = ep_name
        if region_name is None:
            self.sm_rt = boto3.Session().client('runtime.sagemaker')
        else:
            self.sm_rt = boto3.Session().client('runtime.sagemaker',region_name=region_name)
        self.progress_bar = progress_bar
            
    def predict(self,X):
        """
        X: pd.DataFrame or np.array (2D) or list (2D)
        """
        # logging
        logger().info('Process Start  : {}'.format('predict'))
        logger().debug('In  : {}'.format([X]))
        
        # 推論用のBodyを作る
        dfX = pd.DataFrame(X) # convert in case of np.array or list
        
        """
        # method A: CSVファイルを経由してBodyを作る場合
        dfX.to_csv("temp_for_prediction.csv",index=False)
        with open("temp_for_prediction.csv") as f:
            lines = f.readlines()[1:]
            
        # method B: DataFrameから直接Bodyを作る場合
        lines = []
        for idx in dfX.index:
            x_array = dfX.iloc[idx,:].values # np.array without target
            x_list = list(map(str, x_array)) # list
            x_str = ','.join(x_list)
            lines.append(x_str)
            
        """            
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
        
        """
        # method A: CSVファイルを経由してBodyを作る場合
        
        # tempファイルの削除
        path = os.getcwd()
        rem_file = path + "/temp_for_prediction.csv"
        os.remove(rem_file)
        """
        # logging
        logger().debug('Out  : {}'.format([np.array(pre_float)]))
        logger().info('Process End  : {}'.format('predict'))

        return np.array(pre_float)

    def predict_proba(self,X):
        """
        X: pd.DataFrame or np.array (2D) or list (2D)
        """        
        # logging
        logger().info('Process Start  : {}'.format('predict_proba'))
        logger().debug('In  : {}'.format([X]))
        
        # 推論用のBodyを作る
        dfX = pd.DataFrame(X) # convert in case of np.array or list
        
        """
        # method A: CSVファイルを経由してBodyを作る場合
        dfX.to_csv("temp_for_prediction.csv",index=False)
        with open("temp_for_prediction.csv") as f:
            lines = f.readlines()[1:]
            
        # method B: DataFrameから直接Bodyを作る場合
        lines = []
        for idx in dfX.index:
            x_array = dfX.iloc[idx,:].values # np.array without target
            x_list = list(map(str, x_array)) # list
            x_str = ','.join(x_list)
            lines.append(x_str)
            
        """            
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
        
        """
        # method A: CSVファイルを経由してBodyを作る場合
        
        # tempファイルの削除
        path = os.getcwd()
        rem_file = path + "/temp_for_prediction.csv"
        os.remove(rem_file)
        """
        # logging
        logger().debug('Out  : {}'.format([np.array(pre_float)]))
        logger().info('Process End  : {}'.format('predict_proba'))
        
        return np.array(pre_float)

# 内部関数
class _AutopilotMultiprocessRegressor:
    def __init__(self,ep_name,region_name=None,progress_bar=False):
        self.ep_name = ep_name
        self.region_name = region_name
        self.progress_bar = progress_bar
            
    def predict(self,X):
        """
        X: pd.DataFrame or np.array (2D) or list (2D)
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
        """
        X: pd.DataFrame or np.array (2D) or list (2D)
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