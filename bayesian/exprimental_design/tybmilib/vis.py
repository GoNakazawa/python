#-*- coding: utf-8 -*-
"""
@author: TOYOBO CO., LTD.
"""
# Import functions
import os
import glob
import matplotlib
import japanize_matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
from datetime import datetime
from pathlib import Path
from tybmilib import prep
from tybmilib import datamgmt
from tybmilib import myfilename as mfn
from tybmilib import logmgmt

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
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするたためのクラス
    """
    pass


def show_method(method, vis_df, bucket_name, s3_output, experiment_ID):
    """各種可視化関数を実行するためのラッパー関数

    Args:
        method (str): 1st argument
        vis_df (pandas.DataFrame): 2nd argument
        bucket_name (str): 3rd argument
        s3_output (str): 4th argument
        experiment_ID (list): 5th argument

    Returns:
        None
            
    """    
    if method == 'profiles':
        show_profiles(vis_df, bucket_name, s3_output, experiment_ID)
    elif method == 'pairplot':        
        show_scatter(vis_df, bucket_name, s3_output, experiment_ID) 
    elif method == 'correlation_matrix':
        show_correlation_matrix(vis_df, bucket_name, s3_output, experiment_ID)
    else:
        print("no method")

# 可視化
def show_plot(df,bucket_name,s3_output,experiment_ID,methods):
    """各種可視化関数を実行するためのラッパー関数（Notebook用）

    Args:
        df (pandas.DataFrame): 1st argument
        bucket_name (str): 2nd argument
        s3_output (str): 3rd argument
        experiment_ID (str): 4th argument
        methods (list): 5th argument

    Returns:
        None
            
    """
        
    # 対象データフレーム定義
    x = df.copy()
    x_columns = x.columns.tolist()
    
    # Nullカラムの削除
    null_columns = x.columns[x.isnull().any()].tolist()    
    target_columns = [i for i in x_columns if i not in null_columns]
    
    # 文字列行を含むカラムの削除
    str_columns = []
    for j in range(len(target_columns)):
        pic = x[[target_columns[j]]][x[target_columns[j]].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
        if len(pic) > 0:
            str_columns.append(target_columns[j])
    if len(str_columns) > 0:
        dropped_x = prep.drop_cols(x,str_columns)
    else:
        dropped_x = x.copy()
    
    for method in methods:
        show_method(method, dropped_x, bucket_name, s3_output,experiment_ID)


def show_profiles(df, bucket_name, s3_output, experiment_ID):
    """pandas-profilingの実行関数

    Args:
        df (pandas.DataFrame): 1st argument
        bucket_name (str): 2nd argument
        s3_output (str): 3rd argument
        experiment_ID (str): 4th argument

    Returns:
        None
            
    """
    # logging
    step1_log = mfn.get_step1_visualize_log_filename(experiment_ID, Local_mode=Local_mode)
    logmgmt.logInfo(experiment_ID, "Process Start: show_profiles", step1_log)
    logmgmt.logDebug(experiment_ID, "In: {}".format(df.columns.tolist()), step1_log)

    # データフレーム定義
    x = df.copy()
    
    # Change the config when creating the report
    try:
        profile = pdp.ProfileReport(x, 
                                    title="Pandas Profiling Report", 
                                    explorative=True,
                                    vars=dict(num={"low_categorical_threshold": 0}),
                                   )
    except Exception:
        error_msg = "Error: ProfileReport作成時にエラー発生しました。入力データ形式、データ型について確認して下さい。"
        logmgmt.raiseError(experiment_ID, error_msg, step1_log)

    # file
    profile_filename = mfn.get_profile_filename(experiment_ID, Local_mode=Local_mode) 
    if os.path.exists(profile_filename):
        os.remove(profile_filename)
    profile.to_file(profile_filename)
    
    if Local_mode == True:
        # upload
        uri = datamgmt.upload_file(bucket_name, profile_filename, s3_output)
        logmgmt.logDebug(experiment_ID, 'Out  : {}'.format(uri), step1_log)

    # logging
    logmgmt.logInfo(experiment_ID, "Process End: show_plot", step1_log)


def show_scatter(df, bucket_name, s3_output, experiment_ID):
    """散布図行列の実行関数

    Args:
        df (pandas.DataFrame): 1st argument
        bucket_name (str): 2nd argument
        s3_output (str): 3rd argument
        experiment_ID (str): 4th argument

    Returns:
        None
            
    """
    # logging    
    step1_log = mfn.get_step1_visualize_log_filename(experiment_ID, Local_mode=Local_mode)
    logmgmt.logInfo(experiment_ID, "Process Start: show_profiles", step1_log)
    logmgmt.logDebug(experiment_ID, "In: {}".format(df.columns.tolist()), step1_log)
    
    # データフレーム定義
    x = df.copy()
    # 数値データ以外の削除
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how='all')

    process = tqdm(range(0, 1))
    for i in process:
        process.set_description("Scatter processing")

        fs = 1.0
        max_len_cols = max([len(x) for x in x.columns.tolist()])
        if max_len_cols > 20:
            fs = fs - 0.01 * max_len_cols
        sns.set(style="ticks", font_scale=fs, color_codes=True, font="IPAexGothic")

        # pairplot図を出力
        try:
            sns.pairplot(x, diag_kind='kde')
        except Exception:
            error_msg = "Error: 散布図作成時にエラーが発生したため、入力データ形式、データ型について確認して下さい。"
            logmgmt.raiseError(experiment_ID, error_msg, step1_log)

        # file
        scatter_filename = mfn.get_scatter_filename(experiment_ID, "all", Local_mode=Local_mode)
        if os.path.exists(scatter_filename):
            os.remove(scatter_filename)
        plt.savefig(scatter_filename)
        plt.clf()
        plt.close()

        if Local_mode == True:
            # upload
            uri = datamgmt.upload_file(bucket_name, scatter_filename, s3_output)

            logmgmt.logDebug(experiment_ID, "Out: {}".format(uri), step1_log)

    # logging
    logmgmt.logInfo(experiment_ID, "Process End: show_scatter", step1_log)


def show_correlation_matrix(df, bucket_name, s3_output, experiment_ID):
    """相関行列の実行関数

    Args:
        df (pandas.DataFrame): 1st argument
        bucket_name (str): 2nd argument
        s3_output (str): 3rd argument
        experiment_ID (str): 4th argument

    Returns:
        None
            
    """
    # logging
    step1_log = mfn.get_step1_visualize_log_filename(experiment_ID, Local_mode=Local_mode)
    logmgmt.logInfo(experiment_ID, "Process Start: show_profiles", step1_log)
    logmgmt.logDebug(experiment_ID, "In: {}".format(df.columns.tolist()), step1_log)
    
    # データフレーム定義
    x = df.copy()
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how='all')
    
    process = tqdm(range(0, 1))
    for i in process:
        process.set_description("Correlation matrix processing")
        
        # データ指定
        columns = len(df.columns)
        
        # 相関行列
        try:
            corr_mat = x.corr(method='pearson')
        except Exception:
            error_msg = "Error: 相関行列作成時にエラーが発生したため、入力データ形式、データ型について確認して下さい。"
            logmgmt.raiseError(experiment_ID, error_msg, step1_log)

        # 描画条件
        fig, axes = plt.subplots(figsize=(15, 15))
        sns.set(style="ticks", font_scale=0.8, color_codes=True, font="IPAexGothic")
        annot_size = 20.0 - 0.8*columns # 中身
        label_size = 10.0 - 0.3*columns # 軸
        if annot_size < 1.0:
            annot_size = 1.0
        if label_size < 1.0:
            label_size = 1.0

        # ヒートマップ
        b = sns.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.1f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values,
            annot_kws = {'size':annot_size},
            cbar=True
            )
        b.set_xticklabels(b.get_xmajorticklabels(), fontsize = label_size)
        b.set_yticklabels(b.get_ymajorticklabels(), fontsize = label_size)


        # file
        correlation_filename = mfn.get_correlation_filename(experiment_ID, "all", Local_mode=Local_mode)
        if os.path.exists(correlation_filename):
            os.remove(correlation_filename)
        plt.savefig(correlation_filename)
        plt.clf()
        plt.close()

        if Local_mode == True:
            # upload
            uri = datamgmt.upload_file(bucket_name, correlation_filename, s3_output)
            logmgmt.logDebug(experiment_ID, "Out: {}".format(uri), step1_log)
    
    # logging
    logmgmt.logInfo(experiment_ID, "Process End: show_correlation_matrix", step1_log)
        
#------------------------------------------------------------
if __name__ == '__main__':
    None
