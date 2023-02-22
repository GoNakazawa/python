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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
import japanize_matplotlib
import scipy.stats as sts
import seaborn as sons
from datetime import datetime
from tqdm import tqdm # 通常
from tqdm import tqdm_notebook as tqdm # notebook用
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
from IPython.display import IFrame,HTML
from pandas.plotting import scatter_matrix
from tybmilib import prep
from tybmilib.logmgmt import logger, stop_watch

#------------------------------------------------------------
# 可視化
def show_plot(df,objectives,method=['profiles','pairplot','correlation_matrix']):
    logger().info('Process Start  : {}'.format('show_plot'))
    logger().debug('In  : {}'.format([df,objectives,method]))
    
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
        logger().debug('str_columns  : {}'.format(str_columns))
        dropped_x = prep.drop_cols(x,str_columns)
    else:
        dropped_x = x.copy()
    logger().debug('show_plot In  : {}'.format(dropped_x))
    
    # 可視化メソッドの呼び出し
    path_list = []
    for i in method:
        if i == 'profiles':
            path = show_profiles(dropped_x,objectives)
        elif i == 'pairplot':
            path = show_scatter(dropped_x,objectives)
        elif i == 'pairplot_1by1':
            path = show_scatter_1by1(dropped_x,objectives)            
        elif i == 'correlation_matrix':
            path = show_correlation_matrix(dropped_x,objectives)
        else:
            logger().error('Error  : 指定された可視化手法{}には、対応していません'.format(i))
        # localpath追加
        path_list.append(path)
    
    # 作成データ
    print('=========outputフォルダへの格納データ=========')
    for j in range(len(path_list)):
        print(str(path_list[j]))
    
    # logging
    logger().info('Process End  : {}'.format('show_plot'))
    
    return None

def show_profiles(df:'pandas.DataFrame',objectives:'list',display=False):
    logger().info('Process Start  : {}'.format('show_profiles'))
    logger().debug('In  : {}'.format(df))
    
    # データフレーム定義
    x = df.copy()
    
    # Change the config when creating the report
    try:
        profile = pdp.ProfileReport(x, 
                                    title="Pandas Profiling Report", 
                                    explorative=True,
                                    vars=dict(num={"low_categorical_threshold": 0}),
                                   )
    except IndexError:
        raise
    except Exception as err:
        raise Exception("IndexError 以外の何かしらのエラーが発生しました") from err

    new_path = 'output' #フォルダ名
    if not os.path.exists(new_path):#ディレクトリがなかったら
        os.mkdir(new_path)#作成したいフォルダ名を作成
    path = os.getcwd()
    os.chdir(path + '/' + new_path)
    
    pre_profile_list = glob.glob('profile_*')
    if len(pre_profile_list) > 0:
        for j in range(len(pre_profile_list)):
            os.remove(pre_profile_list[j])
                
    name = "profile_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".html"
    profile.to_file(name)
    file_path = path + '/' + new_path + '/' + name
    os.chdir(path)
    if display:
        profile.to_widgets()
        
    # logging
    logger().debug('Out  : {}'.format(file_path))
    logger().info('Process End  : {}'.format('show_plot'))
    return file_path

def show_scatter(df:'pandas.DataFrame',objectives:'list'):
    logger().info('Process Start  : {}'.format('show_scatter'))
    logger().debug('In  : {}'.format([df,objectives]))
    
    # データフレーム定義
    x = df.copy()
    
    # 数値データ以外の削除
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how='all')
    logger().debug('Scatter In  : {}'.format(x))
    
    # 処理
    x_target = x[objectives]
    data_list = [x,x_target]
    name_list = ['scatter_all','scatter_only_objectives']
    file_path = []
    process = tqdm(range(len(data_list)))
    for i in process:
        process.set_description("Scatter processing")
        sns.set(style="ticks", font_scale=1.2, color_codes=True, font="IPAexGothic")
        # 出力されるグラフのフォントサイズをカスタマイズ
        matplotlib.rcParams['font.size'] = 5
        sns.set_context('talk', font_scale=0.9);
        
        # pairplot図を出力
        sns.pairplot(data_list[i], diag_kind='kde')
        
        # 画像データの保存
        new_path = 'output' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)
        
        plt.savefig(name_list[i]+'.png')
        plt.close()
        plot_path = path + '/' + new_path + '/' + name_list[i] + '.png'
        file_path.append(plot_path)
        os.chdir(path)
        
    # logging
    logger().debug('Out  : {}'.format(file_path))
    logger().info('Process End  : {}'.format('show_scatter'))
    
    return file_path

def show_scatter_1by1(df:'pandas.DataFrame',objectives:'list'):
    # logging
    logger().info('Process Start  : {}'.format('show_scatter_1by1'))
    logger().debug('In  : {}'.format([df,objectives]))
    
    # データフレーム定義
    x = df.copy()
    
    # 数値データのみ抽出
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how='all')
    
    # 処理
    x_columns = x.columns.tolist()
    
    file_path = []
    process = tqdm(range(len(x_columns)))
    process.set_description("Scatter 1 by 1 processing")
    for i in process:
        sns.set(style="ticks", font_scale=1.2, color_codes=True, font="IPAexGothic")
        # 出力されるグラフのフォントサイズをカスタマイズ
        matplotlib.rcParams['font.size'] = 5
        sns.set_context('talk', font_scale=0.9);
        
        # pairplot図を出力
        sns.pairplot(x, x_vars=[x_columns[i]], y_vars=x_columns, diag_kind='kde')
        
        # 画像データの保存
        new_path = 'output/pair_plots' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)
        
        target = x_columns[i]
        # 文字列操作
        if '/' in target:
            target = target.replace('/', '')
            
        plt.savefig('pairplot_{}.png'.format(target))
        plt.close()
        
        # path指定
        new_path = os.getcwd()
        file_path.append(new_path)
        os.chdir(path)
    
    file_path = list(set(file_path))
    
    # logging
    logger().debug('Out  : {}'.format(file_path))
    logger().info('Process End  : {}'.format('show_scatter_1by1'))
    
    return file_path

def show_correlation_matrix(df:'pandas.DataFrame',objectives:'list'):
    # logging
    logger().info('Process Start  : {}'.format('show_correlation_matrix'))
    logger().debug('In  : {}'.format([df,objectives]))
    
    # データフレーム定義
    x = df.copy()
    
    # 数値カラムのみに省略
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how='all')
    
    x_target = x[objectives]
    data_list = [x,x_target]
    name_list = ['correlation_matrix_all','correlation_matrix_only_objectives']
    file_path = []
    process = tqdm(range(len(data_list)))
    for i in process:
        process.set_description("Correlation matrix processing")
        
        # データ指定
        data = data_list[i]
        
        # 相関行列
        corr_mat = data.corr(method='pearson')
        '''
        corr_mat = np.zeros((data.shape[1], data.shape[1]))
        
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                corrtest = sts.pearsonr(data[data.columns[i]], data[data.columns[j]])
                # stats配下にspearmanrやkendalltauも存在
                corr_mat[i,j] = corrtest[0]
        df_corr = pd.DataFrame(corr_mat, columns=data.columns, index=data.columns)
        '''
        # 描画条件
        fig, axes = plt.subplots(figsize=(50, 40))
        sns.set(style="ticks", font_scale=1.2, color_codes=True, font="IPAexGothic")
        # 出力されるグラフのフォントサイズをカスタマイズ
        matplotlib.rcParams['font.size'] = 5
        sns.set_context('talk', font_scale=0.9);
        
        # ヒートマップ
        sns.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.1f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values
           )
        
        # 画像データの保存
        new_path = 'output' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)
        
        plt.savefig(name_list[i]+'.png')
        plt.close()
        plot_path = path + '/' + new_path + '/' + name_list[i] + '.png'
        file_path.append(plot_path)
        os.chdir(path)
    
    # logging
    logger().debug('Out  : {}'.format(file_path))
    logger().info('Process End  : {}'.format('show_correlation_matrix'))
        
    return file_path

    def table(self,x:'pandas.DataFrame'):        
        new_path = 'data' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)

        html_template = """
        <!doctype html>
        <html lang="ja">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
          </head>
          <body>
            <script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
            <div class="container">
                {table}
            </div>
          </body>
        </html>
        """
        table = x.to_html(classes=["table", "table-bordered", "table-hover"])
        html = html_template.format(table=table)
        name = "dataframe_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".html"
        with open(name, "w") as f:
            f.write(html)
#------------------------------------------------------------
if __name__ == '__main__':
    None