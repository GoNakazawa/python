#-*- coding: utf-8 -*-
"""パラメータ探索、サンプル生成の実行module
@author: TOYOBO CO., LTD.

【説明】
パラメータ探索用のサンプル生成、作成サンプル上にて探索を実行するモジュール

"""

# Import functions
import numpy as np
import pandas as pd
import boto3
import os
import sagemaker
import random
import pprint
import configparser
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import japanize_matplotlib
import itertools
from tybmilib import modeling
from tybmilib import prep
from tybmilib import logmgmt
from tybmilib import myfilename as mfn
import multiprocessing

#------------------------------------------------------------
# Read local file `config.ini`.

Local_mode = mfn.get_localmode()
localfile_path = mfn.get_static_path()

if Local_mode:
    from tqdm import tqdm # 通常
else:
    from tqdm import tqdm_notebook as tqdm # notebook用

class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするためのクラス
    """
    pass


def get_step_dict(samples, default_step_value=0.01):
    """刻み値のデフォルト設定
    
    Args:
        samples (pandas.DataFrame): 1st argument
    Returns:
        dict: step_dict
        
    """
    step_dict = {}
    for col in samples.columns:
        step_dict[col] = default_step_value
    return step_dict


def limit_samples_by_target(df, target_cols, target_pairs):
    """モデル推論結果に対して、目的変数の設定に応じてフィルタリング
    
    Args:
        df (pandas.DataFrame): 1st argument
        target_cols (list): 2nd argument
        target_pairs (list): 3rd argument
    Returns:
        pandas.DataFrame: temp_sample[df_cond]
        
    """
    temp_sample = df.copy()
    df_cond = True
    for i, col in enumerate(target_cols):
        df_cond = df_cond & (target_pairs[i][0] < temp_sample[col]) & (temp_sample[col] < target_pairs[i][1])
    return temp_sample[df_cond].reset_index(drop=True)


class Search:
    """各探索手法をコントロールするためのクラス
    
    指定された探索手法を別途呼び出しし、実行
    """
    
    def __init__(self,objectives,problemtype,experiment_ID, search_method):
        """コンストラクタ

        Args:
            objectives (list): 1st argument
            problemtype (str): 2nd argument
            experiment_ID (str): 3rd argument
            search_method (list): 4th argument
            
        Returns:
            None
            
        """
        
        self.objectives = objectives
        self.problemtype = problemtype
        self.experiment_ID = experiment_ID
        self.search_method = search_method


    def search_samples(self, samples, target_cols, target_pairs, step_dict, cluster_num=3, k_in_knn=1, rate_of_training_samples_inside_ad=1.0, explore_outside=True, rate_of_explore_outside=1.5):
        """設定された制約条件に基づくサンプル生成を行う機能

        Args:
            samples (pandas.DataFrame): 1st argument
            target_cols (list): 2nd argument
            target_pairs (list): 3rd argument
            step_dict (dict): 4th argument
            k_in_knn (int): 5th argument
            rate_of_training_samples_inside_ad (int): 6th argument
            explore_outside (boolian): 7th argument
            rate_of_explore_outside (int): 8th argument
            
        Returns:
            None

        """

        for method_name in self.search_method:
            if method_name == 'Simulate':
                se = _Simulate(self.objectives, self.experiment_ID, target_cols, target_pairs, step_dict, k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)
                se.search(samples)
                print("simulate")
            elif method_name == 'Search_Cluster':
                se = _Search_Cluster(self.objectives,self.experiment_ID,target_cols,target_pairs, step_dict, k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)
                se.search(samples, N_clusters=cluster_num)
                print("cluster")
            elif method_name == 'Search_Pareto':
                if self.problemtype == 'Regression':
                    se = _Search_Pareto(self.objectives,self.experiment_ID,target_cols,target_pairs, step_dict, k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)
                    se.search(samples)
                    print("pareto")                

        
class _Simulate:
    """探索手法「シミュレーション」をコントロールするためのクラス
    
    指定された探索手法を別途呼び出しし、実行
    """
    
    def __init__(self,objectives,experiment_ID,target_cols,target_pairs,step_dict,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside):
        """コンストラクタ

        Args:
            objectives (list): 1st argument
            experiment_ID (str): 2nd argument
            target_cols (list): 3rd argument
            target_pairs (list): 4th argument
            step_dict (dict): 5th argument
            k_in_knn (int): 6th argument
            rate_of_training_samples_inside_ad (int): 7th argument
            explore_outside (boolian): 8th argument
            rate_of_explore_outside (int): 9th argument
                        
        Returns:
            None
            
        """
        
        self.objectives = objectives
        self.target_cols = target_cols
        self.target_pairs = target_pairs
        self.step_dict = step_dict
        self.k_in_knn = k_in_knn
        self.rate_of_training_samples_inside_ad = rate_of_training_samples_inside_ad
        self.explore_outside = explore_outside
        self.rate_of_explore_outside = rate_of_explore_outside
        self.experiment_ID = experiment_ID
    

    def search(self, samples):
        """生成サンプルに対して探索を実行する機能

        Args:
            samples (pandas.DataFrame): 1st argument
            
        Returns:
            None

        """
        # logging
        step3_log = mfn.get_step3_createsample_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: Simulate", step3_log)
        
        target_list = self.objectives
        logmgmt.logDebug(self.experiment_ID, "target: {}".format(target_list), step3_log)

        # AD範囲判定
        k_in_knn = self.k_in_knn
        rate_of_training_samples_inside_ad = self.rate_of_training_samples_inside_ad
        explore_outside = self.explore_outside
        rate_of_explore_outside = self.rate_of_explore_outside
        
        data_path = mfn.get_samplingx_filename(self.experiment_ID, Local_mode=Local_mode)
        x_train = prep.read_csv(data_path, self.experiment_ID)

        # target条件の反映
        limited_samples = limit_samples_by_target(samples, self.target_cols, self.target_pairs)

        # AD範囲判定
        if len(limited_samples)>0:
            # 探索結果
            x_predict = limited_samples.drop(columns=self.objectives)

            # 欠損値／カテゴリカル変数の除去
            data_list = [x_train,x_predict]
            dropped_list = []
            for i in range(len(data_list)):
                x = data_list[i].copy()
                # 欠損値を含む列の除外
                x_columns = x.columns.tolist()
                null_columns = x.columns[x.isnull().any()].tolist()
                
                # 欠損値を含む行の削除
                x.dropna(how='all')
                
                # 文字列行を含むカラムの削除
                target_columns = [i for i in x_columns if i not in null_columns]
                str_columns = []
                for j in range(len(target_columns)):
                    pic = x[[target_columns[j]]][x[target_columns[j]].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
                    if len(pic) > 0:
                        str_columns.append(target_columns[j])
                dropped_x = prep.drop_cols(x,str_columns, self.experiment_ID)                          
                dropped_list.append(dropped_x)
                        
            d_cols = list(set(dropped_list[0].columns.tolist()) & set(dropped_list[1].columns.tolist()))
            x_train = dropped_list[0][d_cols]
            x_predict = dropped_list[1][d_cols]
            
            # 標準化
            ss = StandardScaler(copy=True, with_mean=True, with_std=True)
            ss.fit(x_train)
            autoscaled_x = pd.DataFrame(ss.transform(x_train), columns = d_cols)
            autoscaled_x_pre = pd.DataFrame(ss.transform(x_predict), columns = d_cols)
            # AD by k-NN for trainning
            ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
            ad_model.fit(autoscaled_x)
            knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)
            knn_distance_train = pd.DataFrame(knn_distance_train)
            knn_distance_train.index = x_train.index
            mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1))
            mean_of_knn_distance_train.columns = ['mean_of_knn_distance']
            sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)
            ad_threshold = sorted_mean_of_knn_distance_train.iloc[
                round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]
            if explore_outside == True:
                ad_threshold = ad_threshold * rate_of_explore_outside

            # AD by k-NN for prediction
            knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_pre, n_neighbors=k_in_knn)
            knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_predict.index)
            knn_distance_prediction.index = x_predict.index
            mean_of_knn_distance_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1))
            inside_ad_flag_prediction = mean_of_knn_distance_prediction <= ad_threshold
            inside_ad_flag_prediction.columns = ['inside_ad_flag']
            
            # merge
            limited_samples = pd.concat([limited_samples, inside_ad_flag_prediction], axis=1)
        if len(limited_samples)>0:
            limited_samples_steped = limited_samples.copy()
            for col in samples.columns:
                temp_sp = limited_samples[col] * (1/self.step_dict[col])
                limited_samples_steped[col] = temp_sp.round() * self.step_dict[col]
            
            limited_samples_steped = limited_samples_steped.drop_duplicates()
            simulate_filename = mfn.get_simulate_filename(self.experiment_ID, Local_mode=Local_mode)
            limited_samples_steped.round(5).to_csv(simulate_filename, index=False, sep=',')
            print('=========outputフォルダへの格納データ=========')
            print('=====【simulate】探索結果:条件を満たす実験サンプル: {}====='.format(os.path.basename(simulate_filename)))

            # logging
            logmgmt.logInfo(self.experiment_ID, "Process End: Simulate", step3_log)
        else:
            error_msg = 'Error: 指定された物性を達成できる実験条件が得られませんでした。'
            logmgmt.raiseError(self.experiment_ID, error_msg, step3_log)


class _Search_Cluster:
    """探索手法「クラスタリング」をコントロールするためのクラス
    
    指定された探索手法を別途呼び出しし、実行
    """
    
    def __init__(self,objectives,experiment_ID,target_cols,target_pairs,step_dict,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside):
        """コンストラクタ

        Args:
            objectives (list): 1st argument
            experiment_ID (str): 2nd argument
            target_cols (list): 3rd argument
            target_pairs (list): 4th argument
            step_dict (dict): 5th argument
            k_in_knn (int): 6th argument
            rate_of_training_samples_inside_ad (int): 7th argument
            explore_outside (boolian): 8th argument
            rate_of_explore_outside (int): 9th argument
                        
        Returns:
            None
            
        """
        
        self.objectives = objectives
        self.target_cols = target_cols
        self.target_pairs = target_pairs
        self.step_dict = step_dict
        self.k_in_knn = k_in_knn
        self.rate_of_training_samples_inside_ad = rate_of_training_samples_inside_ad
        self.explore_outside = explore_outside
        self.rate_of_explore_outside = rate_of_explore_outside
        self.experiment_ID = experiment_ID
    
 
    def search(self, samples, clustering_method='Kmeans', N_clusters=3, eps=50, min_samples=5):
        """生成サンプルに対して探索を実行する機能

        Args:
            samples (pandas.DataFrame): 1st argument
            clustering_method (str): 2nd argument
            N_clusters (int): 3rd argument
            eps (int): 4th argument
            min_samples (int): 5th argument
            
        Returns:
            None

        """
        
        # logging
        step3_log = mfn.get_step3_createsample_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: Search_Cluster", step3_log)
        
        target_list = self.objectives
        logmgmt.logDebug(self.experiment_ID, "Target: {}".format(target_list), step3_log)
        
        # AD範囲判定
        k_in_knn = self.k_in_knn
        rate_of_training_samples_inside_ad = self.rate_of_training_samples_inside_ad
        explore_outside = self.explore_outside
        rate_of_explore_outside = self.rate_of_explore_outside
        
        data_path = mfn.get_samplingx_filename(self.experiment_ID, Local_mode=Local_mode)
        x_train = prep.read_csv(data_path, self.experiment_ID)

        # target 条件反映
        limited_samples = limit_samples_by_target(samples, self.target_cols, self.target_pairs)

        if len(limited_samples)>0:            
            # クラスタリング
            x = limited_samples.drop(columns=target_list)
            y = limited_samples[target_list]
    
            # クラスタリングモデルの設定
            km = KMeans(n_clusters=N_clusters, n_init=30, max_iter=1000)
            gmm = GMM(n_components=N_clusters)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            models = [km,gmm,db]
            models_str =['Kmeans','GMM','DBSCAN']
    
            # 選択モデルでのクラスタリング実行
            model_number = models_str.index(clustering_method)

            # 欠損値／カテゴリカル変数の除去
            data_list = [x]
            dropped_list = []
            for i in range(len(data_list)):
                x = data_list[i].copy()
                # 欠損値を含む列の除外
                x_columns = x.columns.tolist()
                null_columns = x.columns[x.isnull().any()].tolist()
                
                # 文字列行を含むカラムの削除
                target_columns = [i for i in x_columns if i not in null_columns]
                str_columns = []
                for j in range(len(target_columns)):
                    pic = x[[target_columns[j]]][x[target_columns[j]].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
                    if len(pic) > 0:
                        str_columns.append(target_columns[j])
                dropped_x = prep.drop_cols(x,str_columns, self.experiment_ID)
                dropped_list.append(dropped_x)
            
            x = dropped_list[0]
            limited_samples = pd.concat([x, y], axis=1)
            
            # クラスタリングの実施
            cluster_labels = pd.DataFrame(models[model_number].fit_predict(x),columns=['cluster_labels'])
            
            # クラスタリング結果の付与
            result = pd.concat([limited_samples,cluster_labels], axis=1)
           
        # AD範囲判定
        if len(limited_samples)>0:
            # 探索結果
            x_predict = limited_samples.drop(columns=self.objectives)

            # 欠損値／カテゴリカル変数の除去
            data_list = [x_train,x_predict]
            dropped_list = []
            for i in range(len(data_list)):
                x = data_list[i].copy()
                # 欠損値を含む列の除外
                x_columns = x.columns.tolist()
                null_columns = x.columns[x.isnull().any()].tolist()
                
                # 文字列行を含むカラムの削除
                target_columns = [i for i in x_columns if i not in null_columns]
                str_columns = []
                for j in range(len(target_columns)):
                    pic = x[[target_columns[j]]][x[target_columns[j]].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
                    if len(pic) > 0:
                        str_columns.append(target_columns[j])
                dropped_x = prep.drop_cols(x,str_columns, self.experiment_ID)                       
                dropped_list.append(dropped_x)
                        
            d_cols = list(set(dropped_list[0].columns.tolist()) & set(dropped_list[1].columns.tolist()))
            x_train = dropped_list[0][d_cols]
            x_predict = dropped_list[1][d_cols]

            # 標準化
            ss = StandardScaler(copy=True, with_mean=True, with_std=True)
            ss.fit(x_train)
            autoscaled_x = pd.DataFrame(ss.transform(x_train), columns = d_cols)
            autoscaled_x_pre = pd.DataFrame(ss.transform(x_predict), columns = d_cols)
                
            # AD by k-NN for trainning
            ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
            ad_model.fit(autoscaled_x)
            knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)
            knn_distance_train = pd.DataFrame(knn_distance_train)
            knn_distance_train.index = x_train.index
            mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1))
            mean_of_knn_distance_train.columns = ['mean_of_knn_distance']
            sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)
            ad_threshold = sorted_mean_of_knn_distance_train.iloc[
                round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]
            if explore_outside == True:
                ad_threshold = ad_threshold * rate_of_explore_outside

            # AD by k-NN for prediction
            knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_pre, n_neighbors=k_in_knn)
            knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_predict.index)
            knn_distance_prediction.index = x_predict.index
            mean_of_knn_distance_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1))
            inside_ad_flag_prediction = mean_of_knn_distance_prediction <= ad_threshold
            inside_ad_flag_prediction.columns = ['inside_ad_flag']
            
            # merge
            limited_samples = pd.concat([limited_samples, inside_ad_flag_prediction], axis=1)
            
            # クラスタリング結果の付与
            result_ad = pd.concat([limited_samples,cluster_labels], axis=1)

        # 結果格納
        if len(limited_samples)>0:                        
            # 各クラスタラベル代表値の算出
            mean = result.groupby('cluster_labels', as_index=False).mean()
            x_mean = mean.drop(columns=target_list + ['cluster_labels'])

            #可視化
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            
            try:
                # 主成分分析
                pca = PCA(n_components=2)
                pca.fit(x_mean)
                x_pca = pca.transform(x_mean)
                pca_df = pd.DataFrame(x_pca)
                pca_df['cluster_labels'] = mean['cluster_labels'] 
                
                for i in pca_df['cluster_labels'].unique():
                    tmp = pca_df.loc[pca_df['cluster_labels'] == i]
                    ax.scatter(tmp[0], tmp[1], label='cluster_labels_' + str(i))
                plt.title('Clustering_by_{}'.format(models_str[model_number]),size=16)
                plt.tick_params(labelsize=16)
                plt.legend(fontsize=16)
            except Exception as e:
                logmgmt.raiseError(self.experiment_ID, 'Error: 制約条件が少ないため、固有値分解ができませんでした', step3_log)
            
            #result_ad = result_ad.sort_values(by=target_list[0], ascending=False).reset_index(drop=True)
            result_ad_steped = result_ad.copy()
            for col in samples.columns:
                temp_sp = result_ad[col] * (1/self.step_dict[col])
                result_ad_steped[col] = temp_sp.round() * self.step_dict[col]

            result_ad_steped = result_ad_steped.drop_duplicates()
            cluster_filename = mfn.get_cluster_filename(self.experiment_ID, Local_mode=Local_mode)
            result_ad_steped.round(5).to_csv(cluster_filename, index=False, sep=',')
            cluster_mean_filename = mfn.get_cluster_mean_filename(self.experiment_ID, Local_mode=Local_mode)
            mean.to_csv(cluster_mean_filename, index=False, sep=',')
            cluster_img_filename = mfn.get_cluster_img_filename(self.experiment_ID, Local_mode=Local_mode)
            plt.savefig(cluster_img_filename)
            plt.close()
 
            # 出力表示
            print('=========outputフォルダへの格納データ=========')
            print('=====【Search_Cluster】探索結果:条件を満たす実験サンプル: {}====='.format(os.path.basename(cluster_filename)))
            print('クラスタリング結果(クラスター毎の平均値): {}'.format(os.path.basename(cluster_mean_filename)))
            print('各特徴量を2次元に次元圧縮した場合でのクラスタリング状況の描画: {}'.format(os.path.basename(cluster_img_filename)))

            # logging
            logmgmt.logInfo(self.experiment_ID, "Process End: Search_Cluster", step3_log)
        else:
            logmgmt.raiseError(self.experiment_ID, 'Error: 指定された物性を達成できる実験条件が得られませんでした。', step3_log)
        
        
class _Search_Pareto:
    """探索手法「パレート解」をコントロールするためのクラス
    
    指定された探索手法を別途呼び出しし、実行
    """
    
    def __init__(self,objectives,experiment_ID,target_cols,target_pairs,step_dict,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside):
        """コンストラクタ

        Args:
            objectives (list): 1st argument
            experiment_ID (str): 2nd argument
            target_cols (list): 3rd argument
            target_pairs (list): 4th argument
            step_dict (dict): 5th argument
            k_in_knn (int): 6th argument
            rate_of_training_samples_inside_ad (int): 7th argument
            explore_outside (boolian): 8th argument
            rate_of_explore_outside (int): 9th argument
                        
        Returns:
            None
            
        """
        
        self.objectives = objectives
        self.target_cols = target_cols
        self.target_pairs = target_pairs
        self.step_dict = step_dict
        self.k_in_knn = k_in_knn
        self.rate_of_training_samples_inside_ad = rate_of_training_samples_inside_ad
        self.explore_outside = explore_outside
        self.rate_of_explore_outside = rate_of_explore_outside
        self.experiment_ID = experiment_ID


    def search(self, samples):
        """生成サンプルに対して探索を実行する機能

        Args:
            samples (pandas.DataFrame): 1st argument
            
        Returns:
            None

        """
        
        # logging
        step3_log = mfn.get_step3_createsample_log_filename(self.experiment_ID, Local_mode=Local_mode)
        logmgmt.logInfo(self.experiment_ID, "Process Start: Search_Pareto", step3_log)
                        
        # AD範囲判定
        k_in_knn = self.k_in_knn
        rate_of_training_samples_inside_ad = self.rate_of_training_samples_inside_ad
        explore_outside = self.explore_outside
        rate_of_explore_outside = self.rate_of_explore_outside
        
        # originalデータ範囲特定
        number_of_y = len(self.objectives)
        target_list = self.objectives
        logmgmt.logDebug(self.experiment_ID, "Target: {}".format(target_list), step3_log)

        x_data_path = mfn.get_samplingx_filename(self.experiment_ID, Local_mode=Local_mode)
        x_train = prep.read_csv(x_data_path, self.experiment_ID)
        y_data_path = mfn.get_trainy_filename(self.experiment_ID, Local_mode=Local_mode)
        y_train = prep.read_csv(y_data_path, self.experiment_ID)
        
        # target条件反映
        limited_samples = limit_samples_by_target(samples, self.target_cols, self.target_pairs)
        col_list = limited_samples.columns.to_list()
        
        # AD範囲判定
        if len(limited_samples)>0:
            # 探索結果
            x_predict = limited_samples.drop(columns=self.objectives)
            y_predict = limited_samples[self.objectives]
            y_train = y_train[self.objectives]

            # 欠損値／カテゴリカル変数の除去
            data_list = [x_train,x_predict]
            dropped_list = []
            for i in range(len(data_list)):
                x = data_list[i].copy()
                # 欠損値を含む列の除外
                x_columns = x.columns.tolist()
                null_columns = x.columns[x.isnull().any()].tolist()
                
                # 文字列行を含む列の除外
                target_columns = [i for i in x_columns if i not in null_columns]
                str_columns = []
                for j in range(len(target_columns)):
                    pic = x[[target_columns[j]]][x[target_columns[j]].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
                    if len(pic) > 0:
                        str_columns.append(target_columns[j])
                dropped_x = prep.drop_cols(x,str_columns, self.experiment_ID)
                dropped_list.append(dropped_x)
            
            d_cols = list(set(dropped_list[0].columns.tolist()) & set(dropped_list[1].columns.tolist()))
            x_train_d = dropped_list[0][d_cols]
            x_predict_d = dropped_list[1][d_cols]

            # 標準化
            ss = StandardScaler(copy=True, with_mean=True, with_std=True)
            ss.fit(x_train_d)
            autoscaled_x = pd.DataFrame(ss.transform(x_train_d), columns = d_cols)
            autoscaled_x_pre = pd.DataFrame(ss.transform(x_predict_d), columns = d_cols)
            
            # AD by k-NN for trainning
            ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
            ad_model.fit(autoscaled_x)
            knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)
            knn_distance_train = pd.DataFrame(knn_distance_train)
            knn_distance_train.index = x_train.index
            mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1))
            mean_of_knn_distance_train.columns = ['mean_of_knn_distance']
            sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)
            ad_threshold = sorted_mean_of_knn_distance_train.iloc[
                round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]
            if explore_outside == True:
                ad_threshold = ad_threshold * rate_of_explore_outside

            # AD by k-NN for prediction
            knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_pre, n_neighbors=k_in_knn)
            knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_predict.index)
            knn_distance_prediction.index = x_predict.index
            mean_of_knn_distance_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1))
            inside_ad_flag_prediction = mean_of_knn_distance_prediction <= ad_threshold
            inside_ad_flag_prediction.columns = ['inside_ad_flag']

            # Pareto-optimal solutions
            y_predict_inside_ad = y_predict.iloc[inside_ad_flag_prediction.values[:, 0], :]
            dataset_prediction_inside_ad = x_predict.iloc[inside_ad_flag_prediction.values[:, 0], :]
            pareto_optimal_index = []
            for sample_number in range(y_predict_inside_ad.shape[0]):
                flag = y_predict_inside_ad <= y_predict_inside_ad.iloc[sample_number, :]
                if flag.any(axis=1).all():
                    pareto_optimal_index.append(sample_number)
            samples_inside_ad = pd.concat([y_predict_inside_ad, dataset_prediction_inside_ad], axis=1)
            pareto_optimal_samples = samples_inside_ad.iloc[pareto_optimal_index, :]
            
            # index
            pareto_optimal_samples = pareto_optimal_samples.reindex(columns=col_list)
            pareto_optimal_samples = pareto_optimal_samples.sort_values(by=target_list[0], ascending=False).reset_index(drop=True)
            pareto_optimal_steped = pareto_optimal_samples.copy()
            for col in samples.columns:
                temp_sp = pareto_optimal_samples[col] * (1/self.step_dict[col])
                pareto_optimal_steped[col] = temp_sp.round() * self.step_dict[col]
            pareto_optimal_steped = pareto_optimal_steped.drop_duplicates()

            # 文字列操作
            for i in range(len(target_list)):
                if '/' in target_list[i]:
                    target_list[i] = target_list[i].replace('/', '')
            
            # 可視化
            if number_of_y > 2:
                # 3D plot for understanding
                all_trio = itertools.permutations([j for j in range(number_of_y)], 3)
                for j in all_trio:
                    t_list = [target_list[i] for i in list(j)]
                    fig = plt.figure(figsize=(25,20))
                    ax = fig.add_subplot(projection='3d')
                    ax.plot(y_train.iloc[:, j[0]], y_train.iloc[:, j[1]], y_train.iloc[:, j[2]], "o", color='red', label='training samples', ms=5, mew=0.5)
                    ax.plot(y_predict.iloc[:, j[0]], y_predict.iloc[:, j[1]], y_predict.iloc[:, j[2]], "o", color='grey', label='samples outside AD in prediction', ms=5, mew=0.5)
                    ax.plot(y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[0]], y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[1]], y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[2]], "o", color='black', label='samples inside AD in prediction', ms=2, mew=0.5)
                    ax.plot(y_predict_inside_ad.iloc[pareto_optimal_index, j[0]], y_predict_inside_ad.iloc[pareto_optimal_index, j[1]], y_predict_inside_ad.iloc[pareto_optimal_index, j[2]], "o", color='blue', label='Pareto optimum samples in prediction', ms=7, mew=0.5)
                    ax.set_xlabel(y_train.columns[j[0]])
                    ax.set_ylabel(y_train.columns[j[1]])
                    ax.set_zlabel(y_train.columns[j[2]])
                    ax.view_init(elev=25, azim=75)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
                    plt.title('Pareto optimal samples {}'.format('・'.join(t_list)),size=18)
                    plt.tick_params(labelsize=16)

                    # データ保存
                    pareto_filename = mfn.get_pareto_filename(self.experiment_ID, Local_mode=Local_mode)
                    pareto_optimal_steped.round(5).to_csv(pareto_filename, index=False)
                    pareto_img_filename = mfn.get_pareto_img_filename(self.experiment_ID, t_list[0], t_list[1], target3=t_list[2], Local_mode=Local_mode)
                    plt.savefig(pareto_img_filename, bbox_inches='tight')
                    plt.close()

            elif number_of_y == 2:
                all_pair = itertools.permutations([j for j in range(number_of_y)], 2)
                for j in all_pair:
                    t_list = [target_list[i] for i in list(j)]
                    #plt.rcParams['font.size'] = 12
                    plt.figure(figsize=(25,20))
                    plt.scatter(y_train.iloc[:, j[0]], y_train.iloc[:, j[1]], color='red', label='training samples')
                    plt.scatter(y_predict.iloc[:, j[0]], y_predict.iloc[:, j[1]], color='grey', label='samples outside AD in prediction')
                    plt.scatter(y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[0]],y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[1]], color='black',label='samples inside AD in prediction')
                    plt.scatter(y_predict_inside_ad.iloc[pareto_optimal_index, j[0]],y_predict_inside_ad.iloc[pareto_optimal_index, j[1]], color='blue',label='Pareto optimum samples in prediction')
                    plt.xlabel(y_train.columns[j[0]])
                    plt.ylabel(y_train.columns[j[1]])
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
                    plt.tick_params(labelsize=16)
                    xmin = min(y_train.iloc[:, j[0]].min(), y_predict.iloc[:, j[0]].min())
                    xmax = max(y_train.iloc[:, j[0]].max(), y_predict.iloc[:, j[0]].max())
                    ymin = min(y_train.iloc[:, j[1]].min(), y_predict.iloc[:, j[1]].min())
                    ymax = max(y_train.iloc[:, j[1]].max(), y_predict.iloc[:, j[1]].max())
                    plt.xlim([xmin - (xmax - xmin) * 0.1, xmax + (xmax - xmin) * 0.1])
                    plt.ylim([ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1])
                    plt.title('Pareto optimal samples {}'.format('・'.join(t_list)),size=18)
                                            
                    # データ保存
                    pareto_filename = mfn.get_pareto_filename(self.experiment_ID, Local_mode=Local_mode)
                    pareto_optimal_steped.round(5).to_csv(pareto_filename, index=False)
                    pareto_img_filename = mfn.get_pareto_img_filename(self.experiment_ID, t_list[0], t_list[1], Local_mode=Local_mode)
                    plt.savefig(pareto_img_filename, bbox_inches='tight')
                    plt.close()

            else:
                logmgmt.raiseError(self.experiment_ID, 'Error: 目的変数が単一のため、パレート解が発見できませんでした。他の探索手法をご利用ください。', step3_log)

            # Check the pareto-optimal solutions
            print('=========outputフォルダへの格納データ=========')
            print('パレート解となるサンプル群: {}'.format(os.path.basename(pareto_filename)))
            if number_of_y > 2:
                print('パレート解の描画(使用2変数ごとに2次元で描画): {}'.format(os.path.basename(pareto_img_filename)))
            # logging
            logmgmt.logInfo(self.experiment_ID, "Process End: Search_Pareto", step3_log)
        else:
            logmgmt.raiseError(self.experiment_ID, 'Error: 指定された物性を達成できる実験条件が得られませんでした。', step3_log)
