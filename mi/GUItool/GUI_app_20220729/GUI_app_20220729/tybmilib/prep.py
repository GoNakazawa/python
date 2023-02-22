#-*- coding: utf-8 -*-
"""入力データに対する前処理の実行module
@author: TOYOBO CO., LTD.

【説明】
学習データの作成、不要列の削除等を実行

"""

# Import functions
from csv import excel_tab
from ctypes import LibraryLoader
import pandas as pd
import boto3
import os
import re
import sagemaker
import tempfile
import msoffcrypto
import configparser
from pathlib import Path
from getpass import getpass
from tybmilib import datamgmt
from tybmilib import myfilename as mfn
from tybmilib import logmgmt

#------------------------------------------------------------
# Read local file `config.ini`.

Local_mode = mfn.get_localmode()


class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするたためのクラス
    """
    
    pass

def present_info(objectives,s3_uri_list,traindata_path_list,role,bucket_name,user_name,experiment_ID):
    """Notebook用のkernel再起動時のグローバル関数リストを出力

    Args:
        objectives (list): 1st argument
        s3_uri_list (list): 2nd argument
        traindata_path_list (list): 3rd argument
        role (str): 4th argument
        bucket_name (str): 5th argument
        user_name (str): 6th argument
        experiment_ID (str): 7th argument

    Returns:
        None
            
    """
    print('------------------------------')
    print('#=========【途中再起動した場合、別セルに貼り付け、実行】以下の情報は、次セクションでも利用します。=========')
    print('objectives = {}'.format(objectives))
    print('s3_uri_list = {}'.format(s3_uri_list))
    print('traindata_path_list = {}'.format(traindata_path_list))
    print(f'role = {role!r}')
    print(f'bucket_name = {bucket_name!r}')
    print(f'user_name = {user_name!r}')
    print(f'experiment_ID = {experiment_ID!r}')
    
    

def read_s3_bucket_data(s3_uri, experiment_ID, excel_password=None):
    """S3にアップロードされた指定オブジェクトをDataFrameとして読み込み

    Args:
        s3_uri (list): 1st argument
        experiment_ID (str): 2nd argument
        excel_password (str): 3rd argument

    Returns:
        pandas.DataFrame: df
            
    """
    # ログ
    step_all_log = mfn.get_step_all_log_filename(experiment_ID, Local_mode=Local_mode)

    try:
        bucket_name, s3_filename = datamgmt.split_s3_path(s3_uri)
        local_folder = mfn.get_csv_data_path(experiment_ID, Local_mode=Local_mode)
        local_file = datamgmt.download_file(bucket_name, s3_filename, local_folder)

        file_directory = local_file.split('/')
        file_name = file_directory[len(file_directory)-1]
        if '.csv' in file_name:
            df = read_csv(local_file, experiment_ID)
        elif '.xlsx' in file_name:
            df = read_excel(local_file, experiment_ID, excel_password)
        else:
            error_msg = "Error: 指定データ形式をcsv形式、xlsx形式に変更して下さい。"
            raise Lib_ParseError(error_msg)

        # 改行ありカラム名変更
        df_columns_list = [s.replace('\n', '') for s in df.columns.tolist()]
        df.rename(columns=dict(zip(df.columns, df_columns_list)), inplace=True)
        print('読み込みデータのカラム名: ' + ",".join([str(_) for _ in list(df.columns)]))
        os.remove(local_file)

        # 末尾の空白を削除
        while("Unnamed" in df.columns[-1]):
            df = df.drop(df.columns[-1], axis=1)

        return df
    except:
        error_msg = 'Error  : 分析対象のs3パスを正しく入力してください'
        logmgmt.raiseError(experiment_ID, error_msg, step_all_log)
        

def read_csv(filename, experiment_ID):
    """S3にアップロードされた指定オブジェクトが.csvファイルであった場合での読み込み機能

    Args:
        filename (str): 1st argument

    Returns:
        pandas.DataFrame: data
            
    """

    # csvファイルの読み込み
    try:
        data = pd.read_csv(filename)
    except:
        data = pd.read_csv(filename, engine = "python")
    
    return data

def read_excel(filename, experiment_ID, excel_password=None):
    """S3にアップロードされた指定オブジェクトが.xlsxファイルであった場合での読み込み機能

    Args:
        filename (str): 1st argument
        excel_password (str): 2nd argument

    Returns:
        pandas.DataFrame: data
            
    """
    
    data_directory = filename.split('/')
    file_dir = Path("/".join(data_directory[:len(data_directory)-1]))
    file = data_directory[len(data_directory)-1]
    # password付きExcelの読み込み
    try:
        data = pd.read_excel(filename, sheet_name=0, engine='openpyxl')
    except:
        try:
            for target_file in file_dir.glob(file):
                with target_file.open("rb") as f,tempfile.TemporaryFile() as tf:
                    office_file = msoffcrypto.OfficeFile(f)
                    office_file.load_key(password=excel_password)
                    office_file.decrypt(tf)
                    data = pd.read_excel(tf, sheet_name=0, engine='openpyxl')
        except:
            error_msg = "Error : Excelファイルに設定されたパスワードと、入力されたパスワードが一致しませんでした。"
            raise Lib_ParseError(error_msg)

    # NULL行削除
    data = data.dropna(how='all')
    data.reset_index()
    
    return data

def drop_cols(df, cols, experiment_ID, inplace=False):
    """DataFrameから指定カラムを削除する機能

    Args:
        df (pandas.DataFrame): 1st argument
        cols (list): 2nd argument
        inplace (boolian): 3rd argument

    Returns:
        pandas.DataFrame: dropped_x
            
    """
    # ログ
    step_all_log = mfn.get_step_all_log_filename(experiment_ID, Local_mode=Local_mode)

    # 削除対象のセット
    df_list = list(df.columns)
    intersection_set = set(df_list) & set(cols)
    dropped_x = df.copy()
    if len(intersection_set) == len(cols):
        for j in range(len(cols)):
            dropped_x = dropped_x.drop(cols[j],axis=1)
        if inplace:
            df = dropped_x
    else:
        difference_set = [i for i in cols if i not in df_list]
        error_msg = "Error  : {}がカラムに含まれていません。不要列設定を見直してください".format(",".join([str(_) for _ in difference_set]))
        logmgmt.raiseError(experiment_ID, error_msg, step_all_log)

    return dropped_x

def attach_missingvalue(x, y):
    """DataFrameにて目的変数に欠損値がある場合、全行削除する機能

    Args:
        x (pandas.DataFrame): 1st argument
        y (pandas.DataFrame): 2nd argument

    Returns:
        pandas.DataFrame: x
        pandas.DataFrame: y
            
    """
    # 目的変数に欠損値がある場合、全行削除
    if y.isnull().sum() >0:
        dropped_y = y.dropna()
        x = x[x.index.isin(list(dropped_y.index))]
        y = dropped_y.reset_index(drop=True)
        x = x.reset_index(drop=True)

    return x,y

# サンプリング用のデータを生成
def create_sampling_prepare(df, experiment_ID, objectives ,mutual=False, drop_null=True):
    """DataFrameにて目的変数に欠損値がある場合、全行削除する機能

    Args:
        df (pandas.DataFrame): 1st argument
        experiment_ID (str): 2nd argument
        objectives (list): 3rd argument
        mutual (boolian): 4th argument
        drop_null (boolian): 5th argument
        
    Returns:
        list: traindata_path_list
            
    """
    # ログ
    step_all_log = mfn.get_step_all_log_filename(experiment_ID)
    
    # 設定
    data = df.copy()
    targets = objectives.copy()
    xs = drop_cols(data,targets,experiment_ID)
    
    # データ型変換
    x_columns = xs.columns.tolist()
    object_col = xs.select_dtypes(include=['object']).columns.to_list()
    for j in x_columns:
        if xs[j].dtype == 'int64':
            xs[j]=xs[j].astype(float)
        if j in object_col:
            pass
        else:
            # 数値型カラムに対して文字列が含まれていた場合、エラー出力処理
            try:
                xs[j] = pd.to_numeric(xs[j],errors='raise')
            except:
                logmgmt.raiseError(experiment_ID, "説明変数{}に文字列データが含まれているため、修正を行って下さい。".format(j), step_all_log)
    
    samplingx_filename = mfn.get_samplingx_filename(experiment_ID, Local_mode=Local_mode)
    if os.path.exists(samplingx_filename):
        os.remove(samplingx_filename)
    xs.to_csv(samplingx_filename, index=False, sep=',')


def create_multi_traindata(df, experiment_ID, objectives, mutual=False, drop_null=True):
    """DataFrameにて目的変数に欠損値がある場合、全行削除する機能

    Args:
        df (pandas.DataFrame): 1st argument
        experiment_ID (str): 2nd argument
        objectives (list): 3rd argument
        mutual (boolian): 4th argument
        drop_null (boolian): 5th argument
        
    Returns:
        list: traindata_path_list
            
    """
    # ログ
    step_all_log = mfn.get_step_all_log_filename(experiment_ID)
    
    # 設定
    data = df.copy()
    targets = objectives.copy()
    ys = data.loc[:,targets]
    xs = drop_cols(data,targets,experiment_ID)

    # データ型変換
    x_columns = xs.columns.tolist()
    object_col = xs.select_dtypes(include=['object']).columns.to_list()
    for j in x_columns:
        if xs[j].dtype == 'int64':
            xs[j]=xs[j].astype(float)
        if j in object_col:
            pass
        else:
            # 数値型カラムに対して文字列が含まれていた場合、エラー出力処理
            try:
                xs[j] = pd.to_numeric(xs[j],errors='raise')
            except:
                logmgmt.raiseError(experiment_ID, "説明変数{}に文字列データが含まれているため、修正を行って下さい。".format(j), step_all_log)
    
    # オリジナルデータの保存
    trainx_filename = mfn.get_trainx_filename(experiment_ID, Local_mode=Local_mode)
    trainy_filename = mfn.get_trainy_filename(experiment_ID, Local_mode=Local_mode)
    xs.to_csv(trainx_filename, index=False, sep=',')
    ys.to_csv(trainy_filename, index=False, sep=',')

    # 目的変数分のデータ生成
    traindata_path_list = []
    for j in range(len(targets)):
        if mutual:                          #複数の目的変数が指定されている場合に、他の目的変数を説明変数として使用するか否か
            x = drop_cols(data,[targets[j]],experiment_ID)
        else:
            x = drop_cols(data,targets,experiment_ID)
        try:
            y = df[targets[j]]
        except:
            logmgmt.raiseError(experiment_ID, "指定された目的変数が対象データ内に含まれていません。", step_all_log)
        
        # 欠損値を含むデータの削除
        if drop_null:
            target_x, target_y = attach_missingvalue(x,y)
        else:
            target_x = x
            target_y = y
            
        # csvファイルを保存
        train_x = pd.concat([target_x,target_y],axis=1)
        train_x = multiply_data(train_x)

        # 文字列の操作
        if '/' in targets[j]:
            targets[j] = targets[j].replace('/', '')
        
        # データ保存先指定
        trainob_filename = mfn.get_trainob_filename(experiment_ID, targets[j], Local_mode=Local_mode)
        train_x.to_csv(trainob_filename, index=False, sep=',')
        traindata_path_list.append(trainob_filename)

    # 生成データ
    print('=========dataフォルダへの格納データ=========')
    for j in range(len(targets)):
        print('目的変数: '+ str(targets[j]))
        print('学習データ: train_' + str(targets[j]) + '.csv')
    print('説明変数のみデータ: train(only_x).csv')
    print('目的変数のみデータ: train(only_y).csv')
            
    return traindata_path_list

def multiply_data(df, num_data=500, method='doubling', inplace=False):
    """指定行数まで入力データを倍加または、GANを利用した生成を行う機能

    Args:
        df (pandas.DataFrame): 1st argument
        num_data (int): 2nd argument
        method (str): 3rd argument
        inplace (boolian): 4th argument
        
    Returns:
        pandas.DataFrame: df_return
            
    """

    """
    # Autopilotの制約である500行を超えるデータへ加工
    if method=='generation':
        print('注意: データ生成を用いた場合、実験上の制約が適切に反映されない場合があります。')
        # 元データを用いたCTGANの学習
        discrete_columns = []               #入力データに連続値ではない変数が存在する場合、カラム名を宣言する必要がある
        ctgan = CTGANSynthesizer(epochs=100)
        ctgan.fit(x, discrete_columns)
        samples = ctgan.sample(num_data)
        print('【参考】追加生成された実験サンプル: additional_samples.csv')
        display(samples)
        
        new_path = 'data' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)
        samples.to_csv('additional_samples.csv')
        os.chdir("..")
            
        # 元データを用いたCTGANの学習
        df_merge = pd.concat([x,samples],axis=0)
        df_merge = df_merge.reset_index(drop=True)
    """

    if method=='doubling':
        df_merge = df.copy()
        while len(df_merge) < num_data:
            df_merge = pd.concat([df_merge, df],axis=0)
        df_merge = df_merge.reset_index(drop=True)    
    
    df_return = df_merge.copy()    
    if inplace:
        df = df_return

    return df_return




def copy_s3_items(source_bucket='', source_prefix='', target_bucket='', target_prefix='', dryrun=False):
    """S3のフォルダ内のアイテムを、別のフォルダにコピーする

    Args:
        source_bucket (str): 1st argument
        source_prefix (str): 2nd argument
        target_bucket (str): 3rd argument
        target_prefix (str): 4th argument
        dryrun (bool): 5th argument

    Returns:
        None
            
    """

    s3client = boto3.client('s3')
    contents_count = 0
    next_token = ''

    while True:
        if next_token == '':
            response = s3client.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)
        else:
            response = s3client.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix, ContinuationToken=next_token)

        if 'Contents' in response:
            contents = response['Contents']
            contents_count = contents_count + len(contents)
            for content in contents:
                relative_prefix = re.sub('^' + source_prefix, '', content['Key'])
                if not dryrun:
                    #print('Copying: s3://' + source_bucket + '/' + content['Key'] + ' To s3://' + target_bucket + '/' + target_prefix + relative_prefix)
                    s3client.copy_object(Bucket=target_bucket, Key=target_prefix + relative_prefix, CopySource={'Bucket': source_bucket, 'Key': content['Key']})
                else:
                    print('DryRun: s3://' + source_bucket + '/' + content['Key'] + ' To s3://' + target_bucket + '/' + target_prefix + relative_prefix)

        if 'NextContinuationToken' in response:
            next_token = response['NextContinuationToken']
        else:
            break

def delete_s3_items(s3_bucket, s3_folder, dryrun=False):
    """S3のフォルダ内のアイテムを、削除する

    Args:
        s3_bucket (str): 1st argument
        s3_folder (str): 2nd argument
        dryrun (bool): 3rd argument

    Returns:
        None
            
    """

    contents_count = 0
    next_token = ''
    s3client = boto3.client('s3')

    while True:
        if next_token == '':
            response = s3client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_folder)
        else:
            response = s3client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_folder, ContinuationToken=next_token)

        if 'Contents' in response:
            contents = response['Contents']
            contents_count = contents_count + len(contents)
            for content in contents:
                if not dryrun:
                    #print("Deleting: s3://" + s3_bucket + "/" + content['Key'])
                    s3client.delete_object(Bucket=s3_bucket, Key=content['Key'])
                else:
                    print("DryRun: s3://" + s3_bucket + "/" + content['Key'])

        if 'NextContinuationToken' in response:
            next_token = response['NextContinuationToken']
        else:
            break 

def delete_model(model_list):
    """モデルを削除する

    Args:
        model_list (list): 1st argument

    Returns:
        None
            
    """
    region = boto3.Session().region_name
    # modelの削除
    if len(model_list) > 0:
        client = boto3.Session().client(service_name='sagemaker',region_name=region)
        for model_name in model_list:
            try:
                response = client.delete_model(ModelName=model_name)
            except Exception as e:
                pass

def delete_resources(dept, user_id, exp_id, model_list):
    """実験結果をまとめて削除する

    Args:
        model_list (list): 1st argument

    Returns:
        None
            
    """
    
    user_bucket = mfn.get_user_s3_bucket(dept)
    s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)
    delete_s3_items(user_bucket, s3_prefix)
    delete_s3_items("mi-modeling", s3_prefix)
    delete_model(model_list)