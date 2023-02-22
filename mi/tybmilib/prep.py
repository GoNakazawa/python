#-*- coding: utf-8 -*-
"""
@author: TOYOBO CO., LTD.
"""
# Import functions
import numpy as np
import pandas as pd
import boto3
import os
import sagemaker
from sagemaker import Session
import tempfile
import msoffcrypto
import sys
from getpass import getpass
from pathlib import Path, WindowsPath
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from ctgan import CTGANSynthesizer, load_demo
from pathlib import Path
from tybmilib import datamgmt
from tybmilib import prep
from tybmilib.logmgmt import logger, stop_watch

#------------------------------------------------------------
def present_info(s3_bucket_path,role,objectives,s3_uri_list,traindata_path_list,experiment_ID):
    logger().info('Process Start  : {}'.format('present_info'))
    logger().debug('In  : {}'.format([s3_bucket_path,role,objectives,s3_uri_list,traindata_path_list,experiment_ID]))
    
    print('------------------------------')
    prefix_list = s3_bucket_path.split('/')
    if prefix_list[1] == 'your_name':
        logger().error('Error  : {}'.format('s3_bucket_pathにおいて、ユーザー名を設定・変更して下さい。'))
        # HTTPのエラーコードを返す処理（flask側での画面ポップアップ用）
    else:
        print('#=========【途中再起動した場合、別セルに貼り付け、実行】以下の情報は、次セクションでも利用します。=========')
        print('objectives = {}'.format(objectives))
        print('s3_uri_list = {}'.format(s3_uri_list))
        print('traindata_path_list = {}'.format(traindata_path_list))
        print(f's3_bucket_path = {s3_bucket_path!r}')
        print(f'role = {role!r}')
        print(f'experiment_ID = {experiment_ID!r}')
        
    logger().info('Process End  : {}'.format('present_info'))
    return None

def read_s3_bucket_data(s3_uri:'list'):
    logger().info('Process Start  : {}'.format('read_s3_bucket_data'))
    logger().debug('In  : {}'.format(s3_uri))
    
    # S3からの利用データダウンロード
    # localフォルダにcsvファイルを格納し、パスを返す形式
    dtm = datamgmt.S3Dao()
    local_path_list = dtm.download_data(s3_uri)
    
    # ログ
    logger().debug('In  : {}'.format(local_path_list))
    df_list = []
    for j in range(len(local_path_list)):
        file_directory = local_path_list[j].split('/')
        file_name = file_directory[len(file_directory)-1]
        if '.csv' in file_name:
            df = read_csv(local_path_list[j].replace('.csv', ''))
        elif '.xlsx' in file_name:
            df = read_excel(local_path_list[j].replace('.xlsx', ''))
        else:
            logger().error('Error  : {}'.format('指定データ形式をcsv形式、xlsx形式に変更して下さい。'))
        # 改行ありカラム名変更
        df_columns_list = [s.replace('\n', '') for s in df.columns.tolist()]
        df.rename(columns=dict(zip(df.columns, df_columns_list)), inplace=True)
        print('読み込みデータのカラム名：' + ",".join([str(_) for _ in list(df.columns)]))
        os.remove(local_path_list[j])
    
    #print('報告：読み込み時S3からローカル上に一時保存されたデータは削除されました。')
    logger().debug('Out  : {}'.format(df))
    logger().info('Process End  : {}'.format('read_s3_bucket_data'))
    return df

def read_csv(filename:'str'):
    # ロギング関連
    logger().info('Process Start  : {}'.format('read_csv'))
    logger().debug('In  : {}'.format(filename))
    
    # csvファイルの読み込み
    data = pd.read_csv('{}.csv'.format(filename), sep=',', header=0, engine='python')
    
    # ロギング関連
    logger().debug('Out  : {}'.format(data))
    logger().info('Process End  : {}'.format('read_csv'))
    return data

def read_excel(filename:'str'):
    # ロギング関連
    logger().info('Process Start  : {}'.format('read_excel'))
    logger().debug('In  : {}'.format(filename))    
    
    data_directory = filename.split('/')
    file_dir = Path("/".join(data_directory[:len(data_directory)-1]))
    file = data_directory[len(data_directory)-1] + ".xlsx"
    
    # password付きExcelの読み込み
    try:
        data = pd.read_excel('{}.xlsx'.format(filename), sheet_name=0)
    except Exception as e:
        for target_file in file_dir.glob(file):
            with target_file.open("rb") as f,tempfile.TemporaryFile() as tf:
                office_file = msoffcrypto.OfficeFile(f)
                office_file.load_key(password=getpass(prompt = 'Please enter file password.'))
                office_file.decrypt(tf)
                try:
                    office_file.decrypt(tf)
                except Exception as e:
                    logger().error('Error  : {}'.format('Excelファイルに設定されたパスワードと、入力されたパスワードが一致しませんでした。'))
                data = pd.read_excel(tf, sheet_name=0)
    
    # NULL行削除
    data = data.dropna(how='all')
    data.reset_index()
    
    # ロギング関連
    logger().debug('Out  : {}'.format(data))
    logger().info('Process End  : {}'.format('read_excel'))
    return data

def drop_cols(df:'pandas.DataFrame',cols:'list',inplace=False):
    # ロギング関連
    logger().info('Process Start  : {}'.format('drop_cols'))
    logger().debug('In  : {}'.format(df))   
    
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
        #print(",".join([str(_) for _ in difference_set]) + 'が対象データに含まれていません。')
        logger().error('Error  : {}が対象データに含まれていません。'.format(",".join([str(_) for _ in difference_set])))
    
    # ロギング関連
    logger().debug('Out  : {}'.format(dropped_x))
    logger().info('Process End  : {}'.format('drop_cols'))
    
    return dropped_x

def attach_missingvalue(x:'pandas.DataFrame',y:'pandas.DataFrame'):
    # ロギング関連
    logger().info('Process Start  : {}'.format('attach_missingvalue'))
    logger().debug('In  : {}'.format(y))
    
    # 目的変数に欠損値がある場合、全行削除
    if y.isnull().sum() >0:
        dropped_y = y.dropna()
        x = x[x.index.isin(list(dropped_y.index))]
        y = dropped_y.reset_index(drop=True)
        x = x.reset_index(drop=True)
    else:
        None
                       
    # ロギング関連
    logger().debug('Out  : {}'.format(y))
    logger().info('Process End  : {}'.format('attach_missingvalue'))
    return x,y

def create_multi_traindata(df:'pandas.DataFrame',objectives:'list',mutual=False,drop_null=True):
    # ロギング関連
    logger().info('Process Start  : {}'.format('create_multi_traindata'))
    logger().debug('In  : {}'.format(df))
    
    # 設定
    data = df.copy()
    targets = objectives.copy()
    new_path = 'data' #フォルダ名
    if not os.path.exists(new_path): #ディレクトリがなかったら
        os.mkdir(new_path) #作成したいフォルダ名を作成
    
    # アウトプット保存先指定
    output_path = 'output' #フォルダ名
    if not os.path.exists(output_path):#ディレクトリがなかったら
        os.mkdir(output_path)#作成したいフォルダ名を作成

    # データ保存先指定
    path = os.getcwd()
    os.chdir(path + '/' + new_path)
    
    traindata_list = []
    traindata_path_list = []
    
    # オリジナルデータの保存
    ys = data.loc[:,targets]
    ys.to_csv('{}.csv'.format('train(only_y)'),index=False,sep=',')
    xs = prep.drop_cols(data,targets)
    
    # 目的変数分のデータ生成
    for j in range(len(targets)):
        if mutual: #複数の目的変数が指定されている場合に、他の目的変数を説明変数として使用するか否か
            x = prep.drop_cols(data,[targets[j]])
        else:
            x = prep.drop_cols(data,targets)
        y = df[targets[j]]
        
        x_columns_list = x.columns.tolist()
        response_columns = [i for i in x_columns_list if i not in targets]
        response_x = x.loc[:,response_columns]
        
        # 欠損値を含むデータの削除
        if drop_null:
            target_x, target_y = prep.attach_missingvalue(x,y)
        else:
            target_x = x
            target_y = y
            
        # csvファイルを保存
        train_x = pd.concat([target_x,target_y],axis=1)
        train_x = prep.multiply_data(train_x)
        
        # 文字列の操作
        if '/' in targets[j]:
            targets[j] = targets[j].replace('/', '')
        train_x.to_csv('{}.csv'.format('train_' + targets[j]),index=False,sep=',')
        traindata_list.append(train_x)            
        
        # path
        file_path = path + '/' + new_path + '/' + 'train_' + targets[j] + '.csv'
        traindata_path_list.append(file_path)
        
    xs.to_csv('{}.csv'.format('train(only_x)'),index=False,sep=',')
    os.chdir(path)
        
    # 生成データ
    print('=========dataフォルダへの格納データ=========')
    for j in range(len(targets)):
        print('目的変数：'+ str(targets[j]))
        print('学習データ：train_' + str(targets[j]) + '.csv')
    print('説明変数のみデータ：train(only_x).csv')
    print('目的変数のみデータ：train(only_y).csv')
    
    # ロギング関連
    logger().debug('Out  : {}'.format(traindata_path_list))
    logger().info('Process End  : {}'.format('create_multi_traindata'))
        
    return traindata_path_list

def multiply_data(df:'pandas.DataFrame',num_data=500,method='doubling',inplace=False):
    # ロギング関連
    logger().info('Process Start  : {}'.format('multiply_data'))
    logger().debug('In  : {}'.format(df))
                       
    # 設定
    x = df.copy()    
    '''
    # データ内に欠損値があった場合の補完
    if x.isnull().values.sum() > 0:
        x_columns = x.columns.tolist()
        imp_mean = IterativeImputer(random_state=0)
        imp_mean.fit(x.values.tolist())
        x = imp_mean.transform(x.values.tolist())
        x = pd.DataFrame(x.tolist())
        x.columns = x_columns
    else:
        None
    '''
    
    # Autopilotの制約である500行を超えるデータへ加工
    if method=='generation':
        print('注意：データ生成を用いた場合、実験上の制約が適切に反映されない場合があります。')
        # 元データを用いたCTGANの学習
        discrete_columns = [] #入力データに連続値ではない変数が存在する場合、カラム名を宣言する必要がある
        ctgan = CTGANSynthesizer(epochs=100)
        ctgan.fit(x, discrete_columns)
        samples = ctgan.sample(num_data)
        print('【参考】追加生成された実験サンプル：additional_samples.csv')
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

    if method=='doubling':
        df_merge = x
        while len(df_merge) < num_data:
            df_merge = pd.concat([df_merge,x],axis=0)
        df_merge = df_merge.reset_index(drop=True)    
    
    df_return = df_merge.copy()    
    if inplace:
        df = df_return
                       
    # ロギング関連
    logger().debug('Out  : {}'.format(df_return))
    logger().info('Process End  : {}'.format('multiply_data'))
                       
    return df_return

def delete_resources(s3_bucket_path,model_list,ep_list):
    # logging関連
    logger().info('Process Start  : {}'.format('delete_resources'))
    logger().debug('In  : {}'.format([s3_bucket_path,model_list,ep_list]))
    
    # SDK関連設定
    client = boto3.client('s3')
    bucket = Session().default_bucket()
    session = sagemaker.Session()
    
    # S3データ削除
    client.delete_object(Bucket=bucket, Key=s3_bucket_path)
    
    # modelの削除
    for model_name in model_list:
        session.delete_model(model_name)
    
    # endpointの削除
    for ep_name in ep_list:
        session.delete_endpoint(ep_name)
    
    # logging関連
    logger().info('Process End  : {}'.format('delete_resources'))
    
    return None
    
    '''
def create_testdata(x:'pandas.DataFrame',y:'pandas.DataFrame',test_ratio=0):
    # 全データから抽出するテストデータの割合を指定
    if test_ratio==0:
        x_train = x
        x_test = x
        y_train = y
        y_test = y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=1)
    return x_train, y_train, x_test, y_test

    def add_nonlinear_terms(self,x:'pandas.DataFrame'):
        x_square = x ** 2
        print('\nAdds square term and cross term')
        for i in range(x.shape[1]):
            print(i + 1, '/', x.shape[1])
            for j in range(x.shape[1]):
                if i == j:
                    x = pd.concat(
                        [x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]],axis=1)
                elif i < j:
                    x_cross = x.iloc[:, i] * x.iloc[:, j]
                    x_cross.name = '{0}*{1}'.format(x.columns[i], x.columns[j])
                    x = pd.concat([x, x_cross], axis=1)
        return x

    def add_target(self,x:'pandas.DataFrame',target:'list'):
        info = []
        new_columns = []
        target_list = target
        for i in range(len(target_list)):
            new_column = 'normalized_' + str(target_list[i])
            x[new_column] = (x[target_list[i]].max() - x[target_list[i]]) / (x[target_list[i]].max() - x[target_list[i]].min())
            z_max = x[target_list[i]].max()
            z_min = x[target_list[i]].min()
            converted_value = [new_column,target_list[i],x[target_list[i]].max(),x[target_list[i]].min()]
            info.append(converted_value)
            new_columns.append(new_column)
        print(f'正規化され、追加されたカラム：{new_columns}')
        return x,info

    '''
#------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()