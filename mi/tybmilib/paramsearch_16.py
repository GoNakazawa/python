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
import random
import math
import pprint
import japanize_matplotlib
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture as GMM
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import itertools
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from tqdm.notebook import tqdm
from tybmilib import modeling
from tybmilib import datamgmt
from tybmilib import vis
from tybmilib import prep
from tybmilib import paramsearch
from tybmilib.logmgmt import logger, stop_watch

#------------------------------------------------------------
class Search_Boundary:
    def __init__(self,problemtype):
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
        self.problemtype = problemtype
    
    def set_boundary(self,limit=None,fixed=None,total=None,combination=None,ratio=None):
        # 入力条件のチェック
        ls = [key for key in locals().keys()]
        del ls[0]
        setting_dict = {}
        boundary_list = [limit,fixed,total,combination,ratio]
        for i in range(len(boundary_list)):
            if boundary_list[i] != None:
                if len(boundary_list[i]) > 0:
                    setting_dict[ls[i]] = boundary_list[i]
        return setting_dict
    
    def describe(self,setting_dict):
        # データ読み込み
        pprint.pprint(setting_dict, width=500)
        
        # データ読み込み
        path = os.getcwd()
        data_path = path + "/data/train(only_x)"
        df = prep.read_csv(data_path)
        col_list = df.columns.to_list()
        
        # other_colの判定
        main_col = []
        check = ['combination','total','ratio']
        for j in check:
            if j in setting_dict:
                l = setting_dict[j]
                for i in range(len(l)):
                    main_col[len(main_col):len(main_col)] = l[i]['target']
        main_col = list(set(main_col))
        other_col = list(set(col_list) - set(main_col))        
        
    def setting_format(self,setting_dict):
        format_set = {}
        return format_set
    
    def create_samples(self,setting_dict,objectives,ep_list,number_of_samples=100000,random_seed=32,rate_of_explore_outside=2):
        # logging
        logger().info('Process Start  : {}'.format('create_samples'))
        logger().debug('In  : {}'.format([setting_dict,objectives,ep_list,number_of_samples,random_seed,rate_of_explore_outside]))
        
        # データ読み込み
        path = os.getcwd()
        data_path = path + "/data/train(only_x)"
        df = prep.read_csv(data_path)
        col_list = df.columns.to_list()
        error_messages = []
        
        # プログレスバーの表示
        bar = tqdm(total = float(len(setting_dict)))
        bar.set_description('Sampling progress rate')
        
        # other_colの判定
        main_col = []
        check = ['combination','total','ratio']
        for j in check:
            if j in setting_dict:
                l = setting_dict[j]
                for i in range(len(l)):
                    main_col[len(main_col):len(main_col)] = l[i]['target']
        main_col = list(set(main_col))
        other_col = list(set(col_list) - set(main_col))
        
        # fixed,limitでの制約反映及び、カテゴリカル変数の振り分け
        col_temp = [main_col,other_col]
        df_l = []
        pre_fill = []
        
        for k in range(len(col_temp)):
            target_col = col_temp[k]
            tmp_list = [[0.0 for j in range(0, len(target_col))] for i in range(0, number_of_samples)]
            target_df = pd.DataFrame(tmp_list, columns=target_col)
            
            if 'fixed' in setting_dict:
                bar.update(0.5)
                dict_set = setting_dict['fixed']
                for j in dict_set:
                    if j['target'] in target_col:
                        # logging
                        logger().debug('Parameter Setting [fixed]  : {}'.format(j['target']))
                        
                        # 列名の判定
                        if j['target'] in col_list:
                            # 固定値設定
                            s = j['value']
                            try:
                                v = float(s)
                            except ValueError:
                                v = s
                            target_df[j['target']] = [v for i in range(0, number_of_samples)]
                            pre_fill.append(j['target'])
                            
                        else:
                            error_message = '変数:{}の制約条件_limitは、該当カラム無しのため設定できませんでした。'.format(j['target'])
                            error_messages.append(error_message)

            if 'limit' in setting_dict:
                bar.update(0.5)
                dict_set = setting_dict['limit']
                for j in dict_set:
                    if j['target'] in target_col:
                        # logging
                        logger().debug('Parameter Setting [limit]  : {}'.format(j['target']))
                        
                        # 列名の判定
                        if j['target'] in col_list:
                            # 値範囲設定
                            if j['target'] not in pre_fill:
                                max_j = df[j['target']].max() * rate_of_explore_outside
                                min_j = df[j['target']].min() / rate_of_explore_outside
                                if (int(max_j) - int(min_j)) <= 1*rate_of_explore_outside:
                                    l = [min_j + i*0.01 for i in range((int(abs(max_j-min_j))+1)*100)]
                                else:
                                    l = [min_j + i for i in range(int(abs(max_j-min_j))+1)]
                                l_in_range = [k for k in l if j['range'](k) == True]
                                if len(l_in_range) == 0:
                                    error_message = '変数:{}の制約条件_limitは、元データ範囲及びモデル適用範囲を超えるため、制約条件を設定し直してください。'.format(j['target'])
                                    error_messages.append(error_message)
                                else:
                                    if(max(l_in_range) - min(l_in_range)) < 10:
                                        target_df[j['target']] = [float(random.uniform(min(l_in_range), max(l_in_range))) for i in range(0, number_of_samples)]
                                    else:
                                        target_df[j['target']] = [float(random.randint(int(min(l_in_range)), int(max(l_in_range)))) for i in range(0, number_of_samples)]
                            else:
                                error_message = '変数:{}の制約条件_limitについて、競合する制約条件fixedを優先して反映しました。'.format(j['target'])
                                error_messages.append(error_message)
                            pre_fill.append(j['target'])
                        else:
                            error_message = '変数:{}の制約条件_limitは、該当カラム無しのため設定できませんでした。'.format(j['target'])
                            error_messages.append(error_message)

            # カテゴリ値が含まれていた場合への振り分け対応
            object_col = df.select_dtypes(include=['object']).columns.to_list()
            if len(object_col) > 0:
                ob_target_col = list(set(object_col) & set(target_col))
                if len(ob_target_col) > 0:
                    for j in ob_target_col:
                        # logging
                        logger().debug('Parameter Setting [objective]  : {}'.format(j))
                        
                        if j not in pre_fill:
                            component = list(set(df[j].tolist()))
                            cate_l = []
                            for i in range(0, number_of_samples):
                                r = random.randint(0, len(component)-1)
                                cate_l.append(component[r])
                            target_df[j] = cate_l
                            
                            # 変数振り分け
                            error_message = '変数:{}について、指定カテゴリカル変数が無かったため、ランダムで振り分けを実施しました。'.format(j['target'])
                            error_messages.append(error_message)
                            
                            # 処理済み変数格納
                            pre_fill.append(j)
                            
            # データフレーム作成
            df_l.append(target_df)
            error_messages = list(dict.fromkeys(error_messages))
                            
        # main_colへの制約反映
        pre_fill = list(set(pre_fill))
        main_df = df_l[0]
        other_df = df_l[1]
        
        # 複数の制約条件で競合した場合を判定するため、書き換え済みカラムを保存
        pre_fill_main = []
        
        if 'ratio' in setting_dict:
            # ratioの単独処理では、combination、totalと関連しない部分のみに値振り分けを行う方針を採用（2021/7/30 奥井）
            dict_set = setting_dict['ratio']
            # other_colの反映
            non_ratio_col = []
            check = ['combination','total']
            for j in check:
                if j in setting_dict:
                    l = setting_dict[j]
                    for i in range(len(l)):
                        non_ratio_col[len(non_ratio_col):len(non_ratio_col)] = l[i]['target']
            non_ratio_col = list(set(non_ratio_col))
            
            # 生成サンプルへの制約追加
            for w in range(len(dict_set)):
                comb_list = dict_set[w]['target'].copy()         
                ratio = dict_set[w]['ratio']
                ratio_target_col = list(set(comb_list) - set(non_ratio_col))
                
                # 8/16符号変更
                if len(ratio_target_col) > 0:
                    # logging
                    logger().debug('Parameter Setting [ratio]  : {}'.format(comb_list))
                    
                    # 値設定
                    ratio_a = comb_list[0]
                    ratio_b = comb_list[1]
                    a_v = [df[ratio_a].max(),df[ratio_a].min()]
                    main_v = main_df[comb_list].to_numpy().tolist()
                    for i in range(0, number_of_samples):
                        if sum(main_v[i]) == 0:
                            max_j = a_v[0]
                            min_j = a_v[1]
                            v_a = 0
                            while v == 0:
                                v_a = random.uniform(max_j,min_j)
                            v_b = ratio(v_a)
                        elif main_v[i][0] == 0 and main_v[i][1] != 0:
                            return_ratio = ratio(1)
                            v_b = main_v[i][1]
                            v_a = v_b / return_ratio
                        elif main_v[i][0] != 0 and main_v[i][1] == 0:
                            v_a = main_v[i][0]
                            v_b = ratio(v_a)
                        else:                   
                            v_a = main_v[i][0]
                            v_b = main_v[i][1]
                            
                        # 値反映
                        v_l = [v_a,v_b]
                        for j in range(len(comb_list)):
                            if comb_list[j] not in pre_fill:
                                main_df[comb_list[j]][i] = v_l[j]
                            else:
                                error_message = '組成:{}の制約条件_ratioが他の制約条件(limit,fixed)と競合するため、他の制約条件を優先し反映しました。'.format(comb_list)
                                error_messages.append(error_message)
                    pre_fill_main.append(ratio_target_col)
            bar.update(1)                        
            error_messages = list(dict.fromkeys(error_messages))
        
        # total/combination/ratioの制約反映
        if 'total' in setting_dict:
            dict_set = setting_dict['total']
            
            # 生成サンプルへの制約追加
            for w in range(len(dict_set)):
                # 条件設定
                comb_list = dict_set[w]['target'].copy()
                                
                # 条件設定
                ttl_v = dict_set[w]['total']
                s = main_df[comb_list].sum(axis=1).values.tolist()
                pre_fill_main.append(comb_list)
                
                conf_l = []
                # 組み合わせ制約条件の反映
                if 'combination' in setting_dict:
                    c_dict_set = setting_dict['combination']
                    for w in range(len(c_dict_set)):
                        c_comb_list = c_dict_set[w]['target'].copy()
                        c_col = list(set(comb_list) & set(c_comb_list))
                        if len(c_col) > 0:
                            c_comb_m = c_dict_set[w]['range']
                            conf_l.append([c_comb_list,c_comb_m])
                
                conf_r = []
                # 比例関係の制約条件の反映
                if 'ratio' in setting_dict:
                    r_dict_set = setting_dict['ratio']
                    for w in range(len(r_dict_set)):
                        r_comb_list = r_dict_set[w]['target'].copy()
                        ratio = r_dict_set[w]['ratio']
                        conf_r.append([r_comb_list,ratio])
                
                if sum([x > ttl_v for x in s]) > 0:
                    error_message = '組成:{}の制約条件_totalが他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(comb_list)
                    error_messages.append(error_message)
                else:
                    main_l = main_df[comb_list].to_numpy().tolist()
                    comb_list_value = [[df[i].max(),df[i].min()] for i in main_df.columns.tolist()]
                    for i in range(0, number_of_samples):
                        if len(conf_l) == 0:
                            target_list = list(set(comb_list) - set(pre_fill))
                            pre_total = sum(main_l[i])
                            if pre_total >= ttl_v:
                                for j in range(len(target_list)):
                                    main_df[target_list[j]][i] = 0
                            else:
                                if len(conf_r) == 0:
                                    # 制約条件total単独処理
                                    n_of_target = random.randint(1, len(target_list))
                                    rand_list = random.sample(target_list, n_of_target)

                                    # logging
                                    logger().debug('Parameter Setting [total]  : {}'.format(rand_list))
                                    
                                    # 合計値設定
                                    tmp_total_v = sum(main_df.loc[i,comb_list].to_numpy().tolist())
                                    
                                    for j in range(len(rand_list)):
                                        if j != len(rand_list)-1:
                                            n = comb_list.index(rand_list[j])
                                            max_j = comb_list_value[n][0]
                                            min_j = comb_list_value[n][1]
                                            v = random.uniform(max_j,min_j)
                                            if tmp_total_v + v > ttl_v:
                                                v = ttl_v - tmp_total_v
                                            tmp_total_v += v
                                        else:
                                            v = ttl_v - tmp_total_v
                                        main_df[rand_list[j]][i] = v
                                                                        
                                # ratio関連判定
                                else:
                                    # total + ratio処理
                                    # 組成候補作成
                                    target_list = target_list.copy()
                                    n_of_target = random.randint(1, len(target_list))
                                    rand_list = random.sample(target_list, n_of_target)
                                    
                                    # logging
                                    logger().debug('Parameter Setting [total & ratio]  : {}'.format(rand_list))

                                    # ratio判定
                                    r_check_1 = []
                                    r_check_2 = []
                                    for t in range(len(conf_r)):
                                        r_col = list(set(conf_r[t][0]) & set(rand_list))
                                        if len(r_col) == 1:
                                            r_check_1.append([r_col,conf_r[t][0],conf_r[t][1]])
                                        if len(r_col) == 2:
                                            r_check_2.append([conf_r[t][0],conf_r[t][1]])
                                    
                                    # total値の設定
                                    tmp_total_v = sum(main_df.loc[i,comb_list].to_numpy().tolist())
                                                                        
                                    if len(r_check_2) > 0:
                                        # 将来的に複数個に対応できる必要がある
                                        r_check_2 = r_check_2[0]
                                        r_rand_list = r_check_2[0]
                                        return_ratio = r_check_2[1](1)
                                        ratio = [1,return_ratio]
                                        v = ttl_v / (1+return_ratio)
                                        for j in range(len(r_rand_list)):
                                            if main_df[r_rand_list[j]][i] == 0.0:
                                                main_df[r_rand_list[j]][i] = v*ratio[j]
                                            elif main_df[r_rand_list[j]][i] == v*ratio[j]:
                                                None
                                            else:
                                                error_message = '組成:{}の制約条件_ratioが他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(r_rand_list)
                                                error_messages.append(error_message)        
                                    else:
                                        if len(r_check_1) > 0:
                                            # 将来的に複数個に対応できる必要
                                            r_check_1 = r_check_1.copy()                        
                                            r_check_1 = r_check_1[0].copy()
                                            r = r_check_1[0][0]
                                            another_r_l = r_check_1[1].copy()
                                            another_r_l.remove(r)
                                            another_r = another_r_l[0]

                                            if r in rand_list:
                                                # ratioの比較先に既に値があった場合での判定
                                                another_r_v = main_df[another_r][i]
                                                r_index = r_check_1[1].index(another_r)
                                                if another_r_v > 0:
                                                    if r_index == 1:
                                                        return_ratio = r_check_1[2](1)
                                                        v = another_r_v / return_ratio
                                                    else:
                                                        v = r_check_1[2](another_r_v)
                                                    
                                                    # 既存値が含まれるかの判定
                                                    if main_df[r][i] == 0.0:
                                                        main_df[r][i] = v
                                                        tmp_total_v += v
                                                    elif main_df[r][i] == v:
                                                        None
                                                    else:
                                                        error_message = '組成:{}の制約条件_ratioが他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(r)
                                                        error_messages.append(error_message)
                                                # 単独候補だった場合の判定
                                                elif len(rand_list) == 1:
                                                    v = ttl_v - tmp_total_v
                                                    main_df[r][i] = v
                                                    if r_index == 1:
                                                        main_df[another_r][i] = r_check_1[2](v)
                                                    else:
                                                        return_ratio = r_check_1[2](1)
                                                        main_df[another_r][i] = v / return_ratio
                                                else:
                                                    if main_df[r][i] == 0:
                                                        n = comb_list.index(r)
                                                        max_j = comb_list_value[n][0]
                                                        min_j = comb_list_value[n][1]
                                                        v = random.uniform(max_j,min_j)
                                                        if tmp_total_v + v > ttl_v:
                                                            v = ttl_v - tmp_total_v
                                                        tmp_total_v += v
                                                        main_df[r][i] = v
                                                    else:
                                                        v = main_df[r][i]    
                                                    if r_index == 1:
                                                        main_df[another_r][i] = r_check_1[2](v)
                                                    else:
                                                        return_ratio = r_check_1[2](1)
                                                        main_df[another_r][i] = v / return_ratio
                                                rand_list.remove(r)
                                        else:
                                            None
                                            
                                        # その他total関連カラムの処理
                                        if len(rand_list) > 0:
                                            for j in range(len(rand_list)):
                                                if main_df[rand_list[j]][i] != 0:
                                                    error_message = '組成:{}の制約条件_totalの判定時に、他の制約条件(limit,fixed)を優先し設定しました。'.format(rand_list[j])
                                                    error_messages.append(error_message)
                                                else:
                                                    if j != len(rand_list)-1:
                                                        n = list(main_df.columns).index(rand_list[j])
                                                        max_j = comb_list_value[n][0]
                                                        min_j = comb_list_value[n][1]
                                                        v = random.uniform(max_j,min_j)
                                                        if tmp_total_v + v > ttl_v:
                                                            v = ttl_v - tmp_total_v
                                                        tmp_total_v += v                                                       
                                                    else:
                                                        v = ttl_v - tmp_total_v
                                                    main_df[rand_list[j]][i] = v
                                                    
                        else:
                            target_list = list(set(comb_list) - set(pre_fill))
                            pre_total = sum(main_l[i])
                            
                            # 事前にfixed、limitで値が入力済みで、total制限且つ、combinationに対応(2021/8/17修正 奥井)
                            pre_filled = list(set(comb_list) & set(pre_fill))
                            n_pre_fill = len(pre_filled)
                            
                            if len(conf_r) == 0:
                                # combination + total判定
                                # logging
                                logger().debug('Parameter Setting [total & combination]  : {}'.format(target_list))
                                
                                # total候補内から条件数を満たす候補の選択
                                for t in conf_l:
                                    target_list = t[0]
                                    l = [e for e in [i for i in range(len(target_list)+1)] if e <= len(target_list)]
                                    l_in_range = [k for k in l if t[1](k) == True]
                                    
                                    if n_pre_fill>0:
                                        if len(l_in_range) == 1:
                                            n = l_in_range[0]
                                        else:
                                            n = l_in_range[-1]
                                            
                                        # 既存カラムが入力された場合での処理
                                        n = n - n_pre_fill
                                        if n < 0:
                                            error_message = '組成:{}の制約条件_totalにて、combinationの候補数が他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(comb_list)
                                            error_messages.append(error_message)
                                            rand_list = []
                                        else:
                                            un_pre_filled = list(set(comb_list) - set(pre_fill))
                                            if n >= len(un_pre_filled):
                                                rand_list = un_pre_filled.copy()
                                            else:
                                                rand_list = random.sample(un_pre_filled, n)
                                    else:
                                        if len(l_in_range) == 1:
                                            n = l_in_range[0]
                                        else:
                                            n = random.choice(l_in_range)

                                        # total外の選択候補への値振り分けは、combination側で実施する方針を採用（2021/7/30 奥井）
                                        if n >= len(comb_list):
                                            rand_list = comb_list.copy()
                                        else:
                                            rand_list = random.sample(comb_list, n)
                                    
                                    # 候補判定
                                    tmp_total_v = sum(main_df.loc[i,comb_list].to_numpy().tolist())
                                    if len(rand_list)>0:
                                        for j in range(len(rand_list)):
                                            if main_df[rand_list[j]][i] != 0:
                                                error_message = '組成:{}の制約条件_totalの判定時に、他の制約条件(limit,fixed)を優先し設定しました。'.format(rand_list[j])
                                                error_messages.append(error_message)
                                            else:
                                                if j != len(rand_list)-1:
                                                    n = list(main_df.columns).index(rand_list[j])
                                                    max_j = comb_list_value[n][0]
                                                    min_j = comb_list_value[n][1]
                                                    v = random.uniform(max_j,min_j)
                                                    if tmp_total_v + v > ttl_v:
                                                        v = ttl_v - tmp_total_v
                                                    tmp_total_v += v
                                                else:
                                                    v = ttl_v - tmp_total_v
                                                main_df[rand_list[j]][i] = v
                            else:
                                # combination + total + ratio判定
                                # 事前にfixed、limitで値が入力済みで、total制限且つ、combinationに対応(2021/8/17修正 奥井)
                                pre_filled = list(set(comb_list) & set(pre_fill))
                                n_pre_fill = len(pre_filled)
                                
                                # total候補内から条件数を満たす候補の選択
                                for t in conf_l:
                                    # logging
                                    logger().debug('Parameter Setting [total & combination & ratio]  : {}'.format(target_list))
                                    
                                    target_list = t[0]
                                    l = [e for e in [i for i in range(len(target_list)+1)] if e <= len(target_list)]
                                    l_in_range = [k for k in l if t[1](k) == True]
                                    
                                    if n_pre_fill>0:
                                        if len(l_in_range) == 1:
                                            n = l_in_range[0]
                                        else:
                                            n = l_in_range[-1]
                                        # 既存カラムが入力された場合での処理
                                        n = n - n_pre_fill
                                        if n < 0:
                                            error_message = '組成:{}の制約条件_totalにて、combinationの候補数が他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(comb_list)
                                            error_messages.append(error_message)
                                            rand_list = []
                                        else:
                                            un_pre_filled = list(set(comb_list) - set(pre_fill))
                                            if n >= len(un_pre_filled):
                                                rand_list = un_pre_filled.copy()
                                            else:
                                                rand_list = random.sample(un_pre_filled, n)                                                
                                    else:
                                        if len(l_in_range) == 1:
                                            n = l_in_range[0]
                                        else:
                                            n = random.choice(l_in_range)

                                        # total外の選択候補への値振り分けは、combination側で実施する方針を採用（2021/7/30 奥井）
                                        if n >= len(comb_list):
                                            rand_list = comb_list.copy()
                                        else:
                                            rand_list = random.sample(comb_list, n)
                                            
                                    # ratio判定
                                    r_check_1 = []
                                    r_check_2 = []
                                    for t in range(len(conf_r)):
                                        r_col = list(set(conf_r[t][0]) & set(rand_list))
                                        if len(r_col) == 1:
                                            r_check_1.append([r_col,conf_r[t][0],conf_r[t][1]])
                                        if len(r_col) == 2:
                                            r_check_2.append([conf_r[t][0],conf_r[t][1]])
                                    
                                    # total値の設定
                                    tmp_total_v = sum(main_df.loc[i,comb_list].to_numpy().tolist())
                                    
                                    if len(r_check_2) > 0:
                                        # 将来的に複数個に対応できる必要がある
                                        r_check_2 = r_check_2[0]
                                        r_rand_list = r_check_2[0]
                                        return_ratio = r_check_2[1](1)
                                        ratio = [1,return_ratio]
                                        v = ttl_v / (1+return_ratio)
                                        for j in range(len(r_rand_list)):
                                            if main_df[r_rand_list[j]][i] == 0.0:
                                                main_df[r_rand_list[j]][i] = v*ratio[j]
                                            elif main_df[r_rand_list[j]][i] == v*ratio[j]:
                                                None
                                            else:
                                                error_message = '組成:{}の制約条件_ratioが他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(r_rand_list)
                                                error_messages.append(error_message)        
                                    else:
                                        if len(r_check_1) > 0:
                                            # 将来的に複数個に対応できる必要
                                            r_check_1 = r_check_1.copy()                        
                                            r_check_1 = r_check_1[0].copy()
                                            r = r_check_1[0][0]
                                            another_r_l = r_check_1[1].copy()
                                            another_r_l.remove(r)
                                            another_r = another_r_l[0]

                                            if r in rand_list:
                                                # ratioの比較先に既に値があった場合での判定
                                                another_r_v = main_df[another_r][i]
                                                r_index = r_check_1[1].index(another_r)
                                                if another_r_v > 0:
                                                    if r_index == 1:
                                                        return_ratio = r_check_1[2](1)
                                                        v = another_r_v / return_ratio
                                                    else:
                                                        v = r_check_1[2](another_r_v)
                                                    if main_df[r][i] == 0.0:
                                                        main_df[r][i] = v
                                                        tmp_total_v += v
                                                    elif main_df[r][i] == v:
                                                        None
                                                    else:
                                                        error_message = '組成:{}の制約条件_ratioが他の制約条件(limit,fixed)と競合、矛盾するため、制約条件を設定し直してください。'.format(r)
                                                        error_messages.append(error_message)
                                                # 単独候補だった場合の判定
                                                elif len(rand_list) == 1:
                                                    v = ttl_v - tmp_total_v
                                                    main_df[r][i] = v
                                                    if r_index == 1:
                                                        main_df[another_r][i] = r_check_1[2](v)
                                                    else:
                                                        return_ratio = r_check_1[2](1)
                                                        main_df[another_r][i] = v / return_ratio
                                                else:
                                                    if main_df[r][i] == 0:
                                                        n = comb_list.index(r)
                                                        max_j = comb_list_value[n][0]
                                                        min_j = comb_list_value[n][1]
                                                        v = random.uniform(max_j,min_j)
                                                        if tmp_total_v + v > ttl_v:
                                                            v = ttl_v - tmp_total_v
                                                        tmp_total_v += v
                                                        main_df[r][i] = v
                                                    else:
                                                        v = main_df[r][i]    
                                                    if r_index == 1:
                                                        main_df[another_r][i] = r_check_1[2](v)
                                                    else:
                                                        return_ratio = r_check_1[2](1)
                                                        main_df[another_r][i] = v / return_ratio
                                                rand_list.remove(r)
                                        else:
                                            None
                                            
                                        # その他total関連カラムの処理
                                        if len(rand_list) > 0:
                                            for j in range(len(rand_list)):
                                                if main_df[rand_list[j]][i] != 0:
                                                    error_message = '組成:{}の制約条件_totalの判定時に、他の制約条件(limit,fixed)を優先し設定しました。'.format(rand_list[j])
                                                    error_messages.append(error_message)
                                                else:
                                                    if j != len(rand_list)-1:
                                                        n = list(main_df.columns).index(rand_list[j])
                                                        max_j = comb_list_value[n][0]
                                                        min_j = comb_list_value[n][1]
                                                        v = random.uniform(max_j,min_j)
                                                        if tmp_total_v + v > ttl_v:
                                                            v = ttl_v - tmp_total_v
                                                        tmp_total_v += v                                                       
                                                    else:
                                                        v = ttl_v - tmp_total_v
                                                    main_df[rand_list[j]][i] = v                        
            bar.update(1)                        
            error_messages = list(dict.fromkeys(error_messages))
            
        # 重複削除
        pre_fill_main = list(set(list(itertools.chain.from_iterable(pre_fill_main))))
        
        if 'combination' in setting_dict:
            dict_set = setting_dict['combination']
            for w in range(len(dict_set)):
                comb_list = dict_set[w]['target']
                comb_m = dict_set[w]['range']
                l = [i for i in range(len(comb_list)+1)]
                l_in_range = [k for k in l if comb_m(k) == True]
                
                # logging
                logger().debug('Parameter Setting [combination]  : {}'.format(comb_list))
                
                # ratio判定
                conf_r = []
                # 比例関係の制約条件の反映
                if 'ratio' in setting_dict:
                    r_dict_set = setting_dict['ratio']
                    for w in range(len(r_dict_set)):
                        r_comb_list = r_dict_set[w]['target'].copy()
                        ratio = r_dict_set[w]['ratio']
                        conf_r.append([r_comb_list,ratio])
                
                if len(conf_r) == 0:
                    # combination単独処理
                    # 方針：候補数は毎回判定。かつ、書き換えは行わず条件に合致したセルが0の場合、値振り分け
                    for i in range(0, number_of_samples):
                        # 入力値無しカラムの判定
                        target_list = [comb_list[k] for k in [j for j, x in enumerate(main_df[comb_list].iloc[i].values.tolist()) if x == 0]]
                        target_list = list(set(target_list) - set(pre_fill_main))
                        
                        # logging
                        logger().debug('Parameter Setting [combination]  : {}'.format(target_list))
                        
                        # 設定
                        if len(target_list) > 0:
                            n_of_target = random.choice(l_in_range)
                            if n_of_target > len(comb_list):
                                error_message = '組成:{}の制約条件_combinationの組成候補数が矛盾するため、制約条件を設定し直してください。'.format(comb_list)
                                error_messages.append(error_message)
                            else:
                                pre_list = random.sample(comb_list, n_of_target)
                                rand_list = list(set(pre_list) & set(target_list))
                                if len(rand_list) > 0:
                                    for j in range(len(rand_list)):
                                        v_list = [df[rand_list[j]].max(),df[rand_list[j]].min()]
                                        if v_list == [0,0]:
                                            error_message = '変数:{}の制約条件_combinationについて、実績値が無いため設定できません。他の制約条件limit、fixedを利用して下さい。'.format(rand_list[j])
                                            error_messages.append(error_message)
                                        else:
                                            v_list.sort()
                                            v = 0.0
                                            if (abs(v_list[0]) - abs(v_list[1])) < 10:
                                                while v == 0.0:
                                                    v = float(random.uniform(v_list[0],v_list[1]))
                                            else:
                                                while v == 0.0:
                                                    v = float(random.randint(v_list[0],v_list[1]))
                                            main_df[rand_list[j]][i] = v
                                else:
                                    error_message = '組成:{}の制約条件_combinationの組成候補について、他の制約条件を優先し反映しました。'.format(comb_list)
                                    error_messages.append(error_message)
                                    
                else:
                    # combination + ratio処理
                    # 方針：候補数は毎回判定。かつ、書き換えは行わず条件に合致したセルが0の場合、値振り分け
                    comb_list_value = [[df[i].max(),df[i].min()] for i in main_df.columns.tolist()]

                    for i in range(0, number_of_samples):
                        target_list = [comb_list[k] for k in [j for j, x in enumerate(main_df[comb_list].iloc[i].values.tolist()) if x == 0]]
                        target_list = list(set(target_list) - set(pre_fill_main))
                        
                        # logging
                        logger().debug('Parameter Setting [combination & ratio]  : {}'.format(target_list))
                        
                        # 値判定
                        if len(target_list) > 0:
                            n_of_target = random.choice(l_in_range)
                            if n_of_target > len(comb_list):
                                error_message = '組成:{}の制約条件_combinationの組成候補数が矛盾するため、制約条件を設定し直してください。'.format(comb_list)
                                error_messages.append(error_message)
                            else:
                                pre_list = random.sample(comb_list, n_of_target)
                                rand_list = list(set(pre_list) & set(target_list))
                                
                                if len(rand_list) > 0:
                                    # ratio判定
                                    r_check = []
                                    for t in range(len(conf_r)):
                                        r_col = list(set(conf_r[t][0]) & set(rand_list))
                                        if len(r_col) > 0:
                                            r_check.append([r_col,conf_r[t][0],conf_r[t][1]])

                                    if len(r_check) > 0:
                                        # 将来的に複数個に対応できる必要がある
                                        r_check = r_check.copy()                                        
                                        r_check = r_check[0].copy()
                                        r = r_check[0][0]
                                        another_r_l = r_check[1].copy()
                                        another_r_l.remove(r)
                                        another_r = another_r_l[0]

                                        if r in rand_list:
                                            another_r_v = main_df[another_r][i]
                                            r_index = r_check[1].index(another_r)
                                            if another_r_v > 0:
                                                if r_index == 1:
                                                    return_ratio = r_check[2](1)
                                                    v = another_r_v / return_ratio
                                                else:
                                                    v = r_check[2](another_r_v)
                                                main_df[r][i] = v
                                                rand_list.remove(r)
                                            else:
                                                n = comb_list.index(r)
                                                max_j = comb_list_value[n][0]
                                                min_j = comb_list_value[n][1]
                                                v = random.uniform(max_j,min_j)
                                                main_df[r][i] = v
                                                if another_r not in pre_fill_main:
                                                    if r_index == 1:
                                                        main_df[another_r][i] = r_check[2](v)
                                                    else:
                                                        return_ratio = r_check[2](1)
                                                        main_df[another_r][i] = v / return_ratio
                                                    rand_list.remove(r)

                                    if len(rand_list) > 0:
                                        for j in range(len(rand_list)):
                                            v_list = [df[rand_list[j]].max(),df[rand_list[j]].min()]
                                            if v_list == [0,0]:
                                                error_message = '変数:{}の制約条件_combinationについて、実績値が無いため設定できません。制約条件limit、fixedを利用して下さい。'.format(rand_list[j])
                                                error_messages.append(error_message)
                                            else:
                                                v_list.sort()
                                                v = 0.0
                                                if (abs(v_list[0]) - abs(v_list[1])) < 10:
                                                    while v == 0.0:
                                                        v = float(random.uniform(v_list[0],v_list[1]))
                                                else:
                                                    while v == 0.0:
                                                        v = float(random.randint(v_list[0],v_list[1]))
                                                main_df[rand_list[j]][i] = v
                                    else:
                                        error_message = '組成:{}の制約条件_combinationの組成候補について、他の制約条件を優先し反映しました。'.format(comb_list)
                                        error_messages.append(error_message)
                                            
                                else:
                                    error_message = '組成:{}の制約条件_combinationの組成候補について、他の制約条件を優先し反映しました。'.format(comb_list)
                                    error_messages.append(error_message)
            
            bar.update(1)                        
            error_messages = list(dict.fromkeys(error_messages))
        
        # logging
        if len(error_messages) > 0:
            logger().error('Error  : {}'.format(error_messages))
        
        # 推論用データフレーム作成
        samples = pd.concat([main_df,other_df], axis=1).reindex(columns=col_list)
        
        # エラーメッセージ出力
        #pprint.pprint(error_messages, width=150)
        
        # モデル呼び出し
        mlmodel_list = []
        for i in range(len(ep_list)):
            regressor = modeling._AutopilotRegressor(ep_name=ep_list[i], region_name=self.region,progress_bar=True)
            mlmodel_list.append(regressor)
        
        # 推論
        pre_list = []
        samples = samples.copy()
        for regressor in mlmodel_list:
            pre_list.append(regressor.predict(samples))
        pre_array = np.vstack(pre_list).T
        ys = pd.DataFrame(pre_array,index=samples.index,columns=objectives)
        samples = pd.concat([samples,ys], axis=1)
        
        # 保存先指定
        new_path = 'data' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)
        
        samples.to_csv('{}.csv'.format('Samples'),index=False,sep=',')
        
        # 元ディレクトリ指定
        os.chdir(path)
        
        # logging
        logger().debug('Out  : {}'.format(samples))
        logger().info('Process End  : {}'.format('create_samples'))
                
        return samples
        
class Search:
    def __init__(self,ep_list,objectives,problemtype):
        self.objectives = objectives
        self.problemtype = problemtype
        self.ep_list = ep_list
        
    def search_samples(self, samples, objectives_target, method_list, k_in_knn=1, rate_of_training_samples_inside_ad=1.0, explore_outside=True, rate_of_explore_outside=1.5):
        # logging
        logger().info('Process Start  : {}'.format('search_samples'))
        logger().debug('In  : {}'.format([samples,objectives_target,method_list]))

        for i in method_list:
            if i == 'Simulate':
                se = paramsearch._Simulate(self.ep_list,self.problemtype,self.objectives,objectives_target,i,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)            
            elif i == 'Search_Cluster':
                se = paramsearch._Search_Cluster(self.ep_list,self.problemtype,self.objectives,objectives_target,i,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)                
            elif i == 'Search_Pareto':
                if self.problemtype == 'Regression':
                    se = paramsearch._Search_Pareto(self.ep_list,self.problemtype,self.objectives,objectives_target,i,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)
                else:
                    logger().error('Error  : {}'.format('モデルのproblemtypeが指定された探索手法に対応していません。'))
            elif i == 'Search_Greedy':
                if self.problemtype == 'Regression':
                    se = paramsearch._Search_Greedy(self.ep_list,self.problemtype,self.objectives,objectives_target,i,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside)
                else:
                    logger().error('Error  : {}'.format('モデルのproblemtypeが指定された探索手法に対応していません。'))
            se.search(samples)
        # logging
        logger().info('Process End  : {}'.format('search_samples'))
            
class _Simulate:
    def __init__(self,ep_list,problemtype,objectives,objectives_target,method_name,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside):
        self.ep_list = ep_list
        self.problemtype = problemtype
        self.objectives = objectives
        self.objectives_target = objectives_target
        self.method_name = method_name
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
        self.k_in_knn = k_in_knn
        self.rate_of_training_samples_inside_ad = rate_of_training_samples_inside_ad
        self.explore_outside = explore_outside
        self.rate_of_explore_outside = rate_of_explore_outside
        
    def search(self, samples):
        # logging
        logger().info('Process Start  : {}'.format('Simulate'))
        logger().debug('In  : {}'.format([samples]))
        
        '''
        # モデル呼び出し
        ep_list = self.ep_list
        mlmodel_list = []
        for i in range(len(ep_list)):
            regressor = modeling._AutopilotRegressor(ep_name=ep_list[i], region_name=self.region,progress_bar=True)
            mlmodel_list.append(regressor)
        
        # 推論
        pre_list = []
        samples = samples.copy()
        for regressor in mlmodel_list:
            pre_list.append(regressor.predict(samples))
        pre_array = np.vstack(pre_list).T
        ys = pd.DataFrame(pre_array,index=samples.index,columns=self.objectives)
        samples = pd.concat([samples,ys], axis=1)
        
        '''
        y_in_range = samples.copy()
        number_of_y = len(self.objectives)
        target_list = self.objectives
        
        # AD範囲判定
        k_in_knn = self.k_in_knn
        rate_of_training_samples_inside_ad = self.rate_of_training_samples_inside_ad
        explore_outside = self.explore_outside
        rate_of_explore_outside = self.rate_of_explore_outside
        
        # オリジナルデータ
        path = os.getcwd()
        data_path = path + "/data/train(only_x)"
        x_train = prep.read_csv(data_path)
        
        ls = []
        # ターゲット条件反映
        for w in range(len(self.objectives_target)):
            target = self.objectives_target[w]['target']
            v_range = self.objectives_target[w]['range']
            # ターゲットカラムに関し、条件に合致する行を絞り込み、indexにて元データと突合
            l = samples[target].tolist()
            l_in_range = [k for k in range(len(l)) if v_range(l[k]) == True]
            ls.append(l_in_range)
            
        ls_in_range = ls[0]
        if len(ls) > 0:
            for i in range(len(ls)):
                ls_in_range = list(set(ls_in_range) & set(ls[i]))
                
        limited_samples = samples[samples.index.isin(ls_in_range)].reset_index(drop=True)
        limited_samples = limited_samples.sort_values(by=target_list[0], ascending=False).reset_index(drop=True)
           
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
                dropped_x = prep.drop_cols(x,str_columns)
                    
                # 0.0のみ列の削除
                for c in dropped_x.columns:
                    if (dropped_x[c] == 0.0).all():
                        dropped_x.drop(c, axis=1, inplace=True)                            
                dropped_list.append(dropped_x)
                        
            d_cols = list(set(dropped_list[0].columns.tolist()) & set(dropped_list[1].columns.tolist()))
            x_train = dropped_list[0][d_cols]
            x_predict = dropped_list[1][d_cols]

            # 標準化
            autoscaled_x = (x_train - x_train.mean()) / x_train.std()
            autoscaled_x = autoscaled_x.fillna(0.0).astype('float').reset_index()
            autoscaled_x_pre = (x_predict - x_train.mean()) / x_train.std()
            autoscaled_x_pre = autoscaled_x_pre.fillna(0.0).astype('float').reset_index()
                
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
            # 保存先指定
            new_path = 'output' #フォルダ名
            if not os.path.exists(new_path):#ディレクトリがなかったら
                os.mkdir(new_path)#作成したいフォルダ名を作成
            path = os.getcwd()
            os.chdir(path + '/' + new_path)
            
            # csv格納
            limited_samples.to_csv('samples_{}.csv'.format(self.method_name),index=False,sep=',')
            os.chdir(path)
        
            print('=========outputフォルダへの格納データ=========')
            print('=====【simulate】探索結果:条件を満たす実験サンプル：samples_{}.csv====='.format(self.method_name))
        else:
            logger().error('Error  : {}'.format('指定された物性を達成できる実験条件が得られませんでした。'))
        
        # logging
        logger().info('Process End  : {}'.format('Simulate'))
                    
class _Search_Cluster:
    def __init__(self,ep_list,problemtype,objectives,objectives_target,method_name,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside):
        self.ep_list = ep_list
        self.problemtype = problemtype
        self.objectives = objectives
        self.objectives_target = objectives_target
        self.method_name = method_name
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
        self.k_in_knn = k_in_knn
        self.rate_of_training_samples_inside_ad = rate_of_training_samples_inside_ad
        self.explore_outside = explore_outside
        self.rate_of_explore_outside = rate_of_explore_outside
    
    def search(self, samples, clustering_method='Kmeans', N_clusters=5, eps=50, min_samples=5):
        # logging
        logger().info('Process Start  : {}'.format('Search_Cluster'))
        logger().debug('In  : {}'.format([samples,clustering_method,N_clusters,eps,min_samples]))
        
        '''
        # モデル呼び出し
        ep_list = self.ep_list
        mlmodel_list = []
        for i in range(len(ep_list)):
            regressor = modeling._AutopilotRegressor(ep_name=ep_list[i], region_name=self.region,progress_bar=True)
            mlmodel_list.append(regressor)
        
        # 推論
        pre_list = []
        samples = samples.copy()
        for regressor in mlmodel_list:
            pre_list.append(regressor.predict(samples))
        pre_array = np.vstack(pre_list).T
        ys = pd.DataFrame(pre_array,index=samples.index,columns=self.objectives)
        samples = pd.concat([samples,ys], axis=1)
        '''
        
        y_in_range = samples.copy()
        number_of_y = len(self.objectives)
        target_list = self.objectives
        
        # AD範囲判定
        k_in_knn = self.k_in_knn
        rate_of_training_samples_inside_ad = self.rate_of_training_samples_inside_ad
        explore_outside = self.explore_outside
        rate_of_explore_outside = self.rate_of_explore_outside
        
        # オリジナルデータ
        path = os.getcwd()
        data_path = path + "/data/train(only_x)"
        x_train = prep.read_csv(data_path)
        
        ls = []
        # ターゲット条件反映
        for w in range(len(self.objectives_target)):
            target = self.objectives_target[w]['target']
            v_range = self.objectives_target[w]['range']
            # ターゲットカラムに関し、条件に合致する行を絞り込み、indexにて元データと突合
            l = samples[target].tolist()
            l_in_range = [k for k in range(len(l)) if v_range(l[k]) == True]
            ls.append(l_in_range)
            
        ls_in_range = ls[0]
        if len(ls) > 0:
            for i in range(len(ls)):
                ls_in_range = list(set(ls_in_range) & set(ls[i]))
                
        limited_samples = samples[samples.index.isin(ls_in_range)].reset_index(drop=True)
        
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
                dropped_x = prep.drop_cols(x,str_columns)
                                    
                # 0.0のみ列の削除
                for c in dropped_x.columns:
                    if (dropped_x[c] == 0.0).all():
                        dropped_x.drop(c, axis=1, inplace=True)
                
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
                dropped_x = prep.drop_cols(x,str_columns)
                    
                # 0.0のみ列の削除
                for c in dropped_x.columns:
                    if (dropped_x[c] == 0.0).all():
                        dropped_x.drop(c, axis=1, inplace=True)                            
                dropped_list.append(dropped_x)
                        
            d_cols = list(set(dropped_list[0].columns.tolist()) & set(dropped_list[1].columns.tolist()))
            x_train = dropped_list[0][d_cols]
            x_predict = dropped_list[1][d_cols]

            # 標準化
            autoscaled_x = (x_train - x_train.mean()) / x_train.std()
            autoscaled_x = autoscaled_x.fillna(0.0).astype('float').reset_index()
            autoscaled_x_pre = (x_predict - x_train.mean()) / x_train.std()
            autoscaled_x_pre = autoscaled_x_pre.fillna(0.0).astype('float').reset_index()
                
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
            # 保存先指定
            new_path = 'output' #フォルダ名
            if not os.path.exists(new_path):#ディレクトリがなかったら
                os.mkdir(new_path)#作成したいフォルダ名を作成
            path = os.getcwd()
            os.chdir(path + '/' + new_path)
            
            # 各クラスタラベル代表値の算出
            mean = result.groupby('cluster_labels', as_index=False).mean()
            mean.to_csv('{}.csv'.format('samples_mean_Clustering_by'+models_str[model_number]),index=False,sep=',')
            x_mean = mean.drop(columns=target_list + ['cluster_labels'])
            
            #可視化
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            
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
            plt.legend()
            plt.savefig('Clustering_by_{}.png'.format(str(models_str[model_number])))
            plt.close()

            # クラスタリング結果
            result_ad = result_ad.sort_values(by=target_list[0], ascending=False).reset_index(drop=True)
            result_ad.to_csv('{}.csv'.format('samples_Clustering_by'+models_str[model_number]),index=False,sep=',')
            
            # 元ディレクトリ指定
            os.chdir(path)

            # 出力表示
            print('=========outputフォルダへの格納データ=========')
            print('=====【Search_Cluster】探索結果:条件を満たす実験サンプル：samples_{}.csv====='.format(self.method_name))
            print('クラスタリング結果(クラスター毎の平均値)：Clustering_by_{}.csv'.format(models_str[model_number]))
            print('各特徴量を2次元に次元圧縮した場合でのクラスタリング状況の描画：Clustering_by_{}.png'.format(models_str[model_number]))
        
        else:
            logger().error('Error  : {}'.format('指定された物性を達成できる実験条件が得られませんでした。'))
        logger().info('Process End  : {}'.format('Search_Cluster'))
        
class _Search_Pareto:
    def __init__(self,ep_list,problemtype,objectives,objectives_target,method_name,k_in_knn,rate_of_training_samples_inside_ad,explore_outside,rate_of_explore_outside):
        self.ep_list = ep_list
        self.problemtype = problemtype
        self.objectives = objectives
        self.objectives_target = objectives_target
        self.method_name = method_name
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
        self.k_in_knn = k_in_knn
        self.rate_of_training_samples_inside_ad = rate_of_training_samples_inside_ad
        self.explore_outside = explore_outside
        self.rate_of_explore_outside = rate_of_explore_outside
            
    def search(self,samples,nomalized_object=None):
        # logging
        logger().info('Process Start  : {}'.format('Search_Pareto'))
        logger().debug('In  : {}'.format([samples,nomalized_object]))
                        
        # AD範囲判定
        k_in_knn = self.k_in_knn
        rate_of_training_samples_inside_ad = self.rate_of_training_samples_inside_ad
        explore_outside = self.explore_outside
        rate_of_explore_outside = self.rate_of_explore_outside
        
        if self.problemtype == 'Regression':
            '''
            # モデル呼び出し
            ep_list = self.ep_list
            mlmodel_list = []
            for i in range(len(ep_list)):
                regressor = modeling._AutopilotRegressor(ep_name=ep_list[i], region_name=self.region,progress_bar=True)
                mlmodel_list.append(regressor)

            # 推論
            pre_list = []
            samples = samples.copy()
            for regressor in mlmodel_list:
                pre_list.append(regressor.predict(samples))
            pre_array = np.vstack(pre_list).T
            ys = pd.DataFrame(pre_array,index=samples.index,columns=self.objectives)
            samples = pd.concat([samples,ys], axis=1)
            '''
            y_select = samples.copy()

            # originalデータ範囲特定
            number_of_y = len(self.objectives)
            target_list = self.objectives

            path = os.getcwd()
            data_path = path + "/data/train(only_x)"
            x_train = prep.read_csv(data_path)

            data_path = path + "/data/train(only_y)"
            y_train = prep.read_csv(data_path)
            
            # ターゲット条件反映
            y_in_range = samples.copy()

            ls = []
            for w in range(len(self.objectives_target)):
                target = self.objectives_target[w]['target']
                v_range = self.objectives_target[w]['range']
                # ターゲットカラムに関し、条件に合致する行を絞り込み、indexにて元データと突合
                l = samples[target].tolist()
                l_in_range = [k for k in range(len(l)) if v_range(l[k]) == True]
                ls.append(l_in_range)

            ls_in_range = ls[0]
            if len(ls) > 0:
                for i in range(len(ls)):
                    ls_in_range = list(set(ls_in_range) & set(ls[i]))

            limited_samples = samples[samples.index.isin(ls_in_range)].reset_index(drop=True)
            col_list = limited_samples.columns.to_list()
            
            # AD範囲判定
            if len(limited_samples)>0:
                # 保存先指定
                new_path = 'output' #フォルダ名
                if not os.path.exists(new_path):#ディレクトリがなかったら
                    os.mkdir(new_path)#作成したいフォルダ名を作成
                path = os.getcwd()
                os.chdir(path + '/' + new_path)

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
                    
                    # 文字列行を含むカラムの削除
                    target_columns = [i for i in x_columns if i not in null_columns]
                    str_columns = []
                    for j in range(len(target_columns)):
                        pic = x[[target_columns[j]]][x[target_columns[j]].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
                        if len(pic) > 0:
                            str_columns.append(target_columns[j])            
                    dropped_x = prep.drop_cols(x,str_columns)
                    
                    # 0.0のみ列の削除
                    for c in dropped_x.columns:
                        if (dropped_x[c] == 0.0).all():
                            dropped_x.drop(c, axis=1, inplace=True)
                            
                    dropped_list.append(dropped_x)
                
                d_cols = list(set(dropped_list[0].columns.tolist()) & set(dropped_list[1].columns.tolist()))
                x_train = dropped_list[0][d_cols]
                x_predict = dropped_list[1][d_cols]

                # 標準化
                autoscaled_x = (x_train - x_train.mean()) / x_train.std()
                autoscaled_x = autoscaled_x.fillna(0.0).astype('float').reset_index()
                autoscaled_x_pre = (x_predict - x_train.mean()) / x_train.std()
                autoscaled_x_pre = autoscaled_x_pre.fillna(0.0).astype('float').reset_index()
                
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
                pareto_optimal_samples.to_csv('pareto_optimal_samples.csv', index=False)

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
                        plt.savefig('Pareto_optimal_samples_{}.png'.format('-'.join(t_list)))
                        plt.close()

                elif number_of_y == 2:
                    all_pair = itertools.permutations([j for j in range(number_of_y)], 2)
                    for j in all_pair:
                        t_list = [target_list[i] for i in list(j)]
                        plt.rcParams['font.size'] = 12
                        plt.figure(figsize=(25,20))
                        plt.scatter(y_train.iloc[:, j[0]], y_train.iloc[:, j[1]], color='red', label='training samples')
                        plt.scatter(y_predict.iloc[:, j[0]], y_predict.iloc[:, j[1]], color='grey', label='samples outside AD in prediction')
                        plt.scatter(y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[0]],y_predict.iloc[inside_ad_flag_prediction.values[:, 0], j[1]], color='black',label='samples inside AD in prediction')
                        plt.scatter(y_predict_inside_ad.iloc[pareto_optimal_index, j[0]],y_predict_inside_ad.iloc[pareto_optimal_index, j[1]], color='blue',label='Pareto optimum samples in prediction')
                        plt.xlabel(y_train.columns[j[0]])
                        plt.ylabel(y_train.columns[j[1]])
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
                        xmin = min(y_train.iloc[:, j[0]].min(), y_predict.iloc[:, j[0]].min())
                        xmax = max(y_train.iloc[:, j[0]].max(), y_predict.iloc[:, j[0]].max())
                        ymin = min(y_train.iloc[:, j[1]].min(), y_predict.iloc[:, j[1]].min())
                        ymax = max(y_train.iloc[:, j[1]].max(), y_predict.iloc[:, j[1]].max())
                        plt.xlim([xmin - (xmax - xmin) * 0.1, xmax + (xmax - xmin) * 0.1])
                        plt.ylim([ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1])
                        plt.title('Pareto optimal samples {}'.format('・'.join(t_list)),size=18)
                        plt.savefig('Pareto_optimal_samples_{}.png'.format('-'.join(t_list)), bbox_inches='tight')
                        plt.close()

                # Check the pareto-optimal solutions
                print('=========outputフォルダへの格納データ=========')
                print('パレート解となるサンプル群：pareto_optimal_samples.csv')

                if number_of_y > 2:
                    print('パレート解の描画(使用3変数ごとに3次元で描画)：pareto_optimal_samples_使用変数.png')
                elif number_of_y == 2:
                    print('パレート解の描画(使用2変数ごとに2次元で描画)：pareto_optimal_samples_使用変数.png')

                os.chdir(path)

            else:
                logger().error('Error  : {}'.format('指定された物性を達成できる実験条件が得られませんでした。'))

        # logging
        logger().info('Process End  : {}'.format('Search_Pareto'))
        
class _Search_Greedy:
    def __init__(self,ep_list,problemtype,objectives,objectives_target,method_name):
        self.ep_list = ep_list
        self.problemtype = problemtype
        self.objectives = objectives
        self.objectives_target = objectives_target
        self.method_name = method_name
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
                    
    def search(self,samples,search_direction='max'):
        if self.problemtype == 'Regression':

            # 対象
            target_list = self.objectives
            samples = samples.copy()

            # 探索順番を取得(SHAP値ベース)
            path = os.getcwd()
            data_path = path + "/data/train(only_x)"
            x_train = prep.read_csv(data_path)

            data_path = path + "/data/train(only_y)"
            y_train = prep.read_csv(data_path)

            train_df = pd.concat([x_train, y_train], axis=1)
            drop_col = train_df.select_dtypes(include=['object']).columns.to_list()
            df = train_df.drop(drop_col, axis=1)
            df = df.dropna(how='any')

            x = prep.drop_cols(df,target_list)



            '''
            # shap値の算出
            shap_values_list = []
            for t in range(len(target_list)):
                # xgbootモデルの定義
                xgbreg = xgb.XGBRegressor(booster='gbtree',random_state=2525)
                # パラメータ探索
                xgbreg_cv = GridSearchCV(xgbreg, {'max_depth': [8, 7, 6, 5, 4, 3], 'max_depth': [3,5,7], 'learning_rate': [0.05,0.03,0.01]}, verbose=1)
                # 標準化処理
                scaler = StandardScaler()
                scaler.fit(x)
                scaler.transform(x)
                x_standardization = pd.DataFrame(scaler.transform(x), columns=x.columns)
                x_for_model = x_standardization

                xgbreg_cv.fit(x_for_model, df[target_list[t]])
                xgbreg = xgb.XGBRegressor(**xgbreg_cv.best_params_)
                # xgboostモデルの学習
                xgbreg.fit(x_for_model, df[target_list[t]])
                y_pred = xgbreg.predict(x_for_model)

                # SHAPの定義(説明変数解釈用)
                shap.initjs()
                explainer = shap.TreeExplainer(model=xgbreg, data=x_for_model, feature_perturbation='interventional', model_output='raw')
                shap_values = explainer.shap_values(X=x_for_model)
                shap_values_list.append(shap_values)

            # 探索
            for t in range(len(target_list)):
                shap_value = shap_values_list[t]







                # モデル呼び出し
                eo_list = self.ep_list
                mlmodel_list = []
                for i in range(len(ep_list)):
                    regressor = modeling._AutopilotRegressor(ep_name=ep_list[i], region_name=self.region,progress_bar=True)
                    mlmodel_list.append(regressor)

                # 推論
                pre_list = []
                for regressor in mlmodel_list:
                    pre_list.append(regressor.predict(samples))
                pre_array = np.vstack(pre_list).T
                ys = pd.DataFrame(pre_array,index=samples.index,columns=self.objectives)
                samples = pd.concat([samples,ys], axis=1)





            # 対象
            target_list = self.objectives

            for w in range(len(self.objectives_target)):
                target = self.objectives_target[w]['target']
                v_range = self.objectives_target[w]['range']
                # ターゲットカラムに関し、条件に合致する行を絞り込み、indexにて元データと突合
                y_range = v_range(samples[target])
                y_select = y_select[y_select.index.isin(list(y_range[y_range == True].index))]

            limited_samples = samples[samples.index.isin(list(y_select.index))].reset_index(drop=True)
            #list(limited_samples.columns)

            # 探索順番を取得(SHAP値ベース)
            path = os.getcwd()
            data_path = path + "/data/train(only_x)"
            x_train = prep.read_csv(data_path)

            data_path = path + "/data/train(only_y)"
            y_train = prep.read_csv(data_path)

            train_df = pd.concat([x_train, y_train], axis=1)
            drop_col = train_df.select_dtypes(include=['object']).columns.to_list()
            df = train_df.drop(drop_col, axis=1)
            df = df.dropna(how='any')

            x = prep.drop_cols(df,target_list)

            for t in range(len(target_list)):
                # xgbootモデルの定義
                xgbreg = xgb.XGBRegressor(booster='gbtree',random_state=2525)
                # パラメータ探索
                xgbreg_cv = GridSearchCV(xgbreg, {'max_depth': [8, 7, 6, 5, 4, 3], 'max_depth': [3,5,7], 'learning_rate': [0.05,0.03,0.01]}, verbose=1)
                # 標準化処理
                scaler = StandardScaler()
                scaler.fit(x)
                scaler.transform(x)
                x_standardization = pd.DataFrame(scaler.transform(x), columns=x.columns)
                x_for_model = x_standardization

                xgbreg_cv.fit(x_for_model, df[target_list[t]])
                xgbreg = xgb.XGBRegressor(**xgbreg_cv.best_params_)
                # xgboostモデルの学習
                xgbreg.fit(x_for_model, df[target_list[t]])
                y_pred = xgbreg.predict(x_for_model)

                # SHAPの定義(説明変数解釈用)
                shap.initjs()
                explainer = shap.TreeExplainer(model=xgbreg, data=x_for_model, feature_perturbation='interventional', model_output='raw')
                shap_values = explainer.shap_values(X=x_for_model)





                # 探索順位変更    
                df_shap = pd.DataFrame(data=shap_values,columns=x.columns)
                df_shap_abs = df_shap.abs().mean().sort_values(ascending=False)
                search_order = df_shap_abs.index.to_list()

                # タプルへの値の格納
                points = []
                for i in range(len(search_order)):
                    points.append(limited_samples[search_order[i]].value.tolist())
                points = tuple(points)

                # 探索結果の可視化
                fig = plt.figure(figsize=(30, 20))
                ax = fig.add_subplot(111)

                # 箱ひげ図
                bp = ax.boxplot(points)
                ax.set_xticklabels(search_order)
                plt.title('Box plot')
                plt.xlabel('candidate')
                plt.ylabel('volume')

                # Y軸のメモリのrange
                plt.ylim([0,100])
                plt.grid()
                plt.savefig('Greegy_search.png', bbox_inches='tight')
                plt.close()

            print('=========outputフォルダへの格納データ=========')
            print('Greegy_search.png')
            print('最適な実験条件(周辺値含む)：predicted_search_【主構成となっている材料】.csv')

            for t in range(len(target_list)):
                val_dict = {} # 各説明変数で探索した値を格納
                Optimal_value_dict = {} # 各説明変数の最適値を格納
                done_val_list = [] # 探索終了済の説明変数名を格納

                for i in range(len(search_order)):
                    if search_direction == 'max':
                        Optimal_value_dict[search_param] = val_temp_list[y_temp_ep[0].index(max(y_temp_ep[0]))]
                    elif search_direction == 'min':
                        Optimal_value_dict[search_param] = val_temp_list[y_temp_ep[0].index(min(y_temp_ep[0]))]

                y_dict = {} # 探索後の計算済の目的関数値を説明変数毎に格納
                val_dict = {} # 各説明変数で探索した値を格納
                Optimal_value_dict = {} # 各説明変数の最適値を格納
                done_val_list = [] # 探索終了済の説明変数名を格納

                y_temp_ep = limited_samples[t].to_list() # 探索後の計算済の目的関数値を格納
                for i in range(len(search_order)):
                    search_param = search_order[i]
                    # 探索が終了した変数を保存
                    done_val_list.append(search_param)
                    for i in range(len(target_list)):
                        y_dict_list[i][search_param] = list(map(float, y_temp_ep[i]))
                    val_dict[search_param] = val_temp_list

                # 最適値を保存
                if len(target_list) == 1:
                    if search_direction == 'max':
                        Optimal_value_dict[search_param] = val_temp_list[y_temp_ep[0].index(max(y_temp_ep[0]))]
                    elif search_direction == 'min':
                        Optimal_value_dict[search_param] = val_temp_list[y_temp_ep[0].index(min(y_temp_ep[0]))]



                y_list = [] # 探索後の計算済の目的関数値を格納

                #  探索結果の可視化
                fig = plt.figure(figsize=(30, 20))
                ax = fig.add_subplot(111)

                Coef_pos_fea = coef[coef['Coefficients']>=0]['Features'].to_list()
                Coef_neg_fea = coef[coef['Coefficients']<0]['Features'].to_list()
                step_cnt = 1
                for i in range(len(search_order)):
                    pin = 
                    if search_order[i] in Coef_pos_fea:
                        ax.plot([i+step_cnt for i in range(len(y_dict[search_order[i]]))], y_dict[search_order[i]],marker='o',c='darkblue')
                        step_cnt += pin
                    elif search_order[i] in Coef_neg_fea:
                        ax.plot([i+step_cnt for i in range(len(y_dict[search_order[i]]))], y_dict[search_order[i]][::-1],marker='o',c='darkblue')
                        step_cnt += pin      
                for i in done_val_list:
                    y_temp = y_dict[i]
                    for j in range(len(y_temp)):
                        y_list.append(y_temp[j])
                step_for_viz = pin
                for i in range(len(search_order)):
                    ax.vlines(x=step_for_viz, ymin=min(y_list), ymax=max(y_list), colors='black', linestyle='dashed')
                    ax.text(step_for_viz-pin/1.5,min(y_list),search_order[i],size=18)
                    step_for_viz += pin

                ax.set_xlabel('STEP',size=16)
                ax.set_ylabel(target,size=16)    
                # 文字列操作
                if '/' in target:
                    target = target.replace('/', '')    
                plt.savefig('Search_opimal_multi_value_'+str(target)+'.png')

            # 説明変数の順番を取得
            x_columns = x.columns.to_list()

            for i in range(len(target_const)):
                print('ターゲット群_'+str(i+1)+':',target_const[i])
            blank_columns = []
            for i in range(len(target_const)+1):
                blank_columns.append([])
            target_columns = target_const+[other_const]
            # 各～constで設定されたAcid名を含むカラムを検索
            for w in range(len(target_columns)):
                for i in target_columns[w]:
                    for j in x_columns:
                        if j.startswith(i):
                            blank_columns[w].append(j)
            x_columns_new = [e for inner_list in blank_columns for e in inner_list]    

            # 説明変数の範囲と平均値一覧を取得
            Search_range=x.describe().loc[['mean','min','max']]

            # 探索順番を取得(SHAP値ベース)
            df_shap = pd.DataFrame(data=shap_values,columns=x.columns)
            df_shap_abs = df_shap.abs().mean().sort_values(ascending=False)
            search_order_temp = df_shap_abs.index.to_list()
            search_order = []
            for i in range(len(search_order_temp)):
                if search_order_temp[i] in x_columns_new:
                    search_order.append(search_order_temp[i])

            # 結果を格納するリスト、辞書を作成
            y_dict_list = []
            for i in range(len(target_list)):
                y_dict = {} # 探索後の計算済の目的関数値を説明変数毎に格納
                y_dict_list.append(y_dict)    
            y_list = [] # 探索後の計算済の目的関数値を格納
            val_dict = {} # 各説明変数で探索した値を格納
            Optimal_value_dict = {} # 各説明変数の最適値を格納
            done_val_list = [] # 探索終了済の説明変数名を格納    

            if method == 'sampling':
                # 分布の推定
                x_values = x.values
                mu0 = x.describe().loc[['mean']].values
                cov0 = np.eye(mu0.shape[1]) * 10.
                nD = mu0.shape[1]
                # モデルの記述
                with pm.Model() as mv_norm_1:
                    # mu vector
                    mu = pm.MvNormal('mu', mu=mu0, cov=cov0, shape=mu0.shape)
                    # covariance matrix
                    # ## 各確率変数の分散の事前分布
                    sd_dist = pm.HalfNormal.dist(sd=10, shape=nD)
                    # ## 共分散行列のコレスキー分解Lの各要素（ベクトル）
                    chol_packed = pm.LKJCholeskyCov('chol_packed',eta=1, n=nD, sd_dist=sd_dist)
                    # ## ベクトルとして保持したLを下三角行列Lに変換
                    L = pm.expand_packed_triangular(nD, chol_packed)
                    # ## Lから共分散行列を求める
                    cov = pm.Deterministic('cov', L.dot(L.T))
                    # x (observed)
                    x_obs = pm.MvNormal('x_obs', mu=mu, cov=cov, observed=x_values)
                # サンプリング
                with mv_norm_1:
                    trace = pm.sample(init="adapt_diag")
                # 推定後の平均値と共分散行列を保存
                mu_post = pd.DataFrame(data=trace['mu'].mean(axis=0),columns=x.columns)
                cov_post = pd.DataFrame(data=trace['cov'].mean(axis=0),columns=x.columns,index=x.columns)

            for i in range(len(search_order)):
                # 探索対象の説明変数を指定
                search_param = search_order[i]
                if method == 'sampling':
                    # 探索範囲の取得
                    Sampling = pd.DataFrame(scipy.stats.multivariate_normal(trace['mu'].mean(axis=0)[0],trace['cov'].mean(axis=0),allow_singular=True).rvs(size=sample_size),columns=x.columns)
                    Param_range = Sampling[search_param]
                    value_order = Param_range.sort_values().to_list()
                    value_order_temp = Param_range.sort_values().to_list()
                    value_order = []
                    for i in range(len(value_order_temp)):
                        if value_order_temp[i] < 0:
                            value_order.append(0.)
                        else:
                            value_order.append(value_order_temp[i])

                elif method == 'greedy':
                    # 探索範囲の取得
                    Param_range = Search_range[search_param]
                    minval = Param_range['min']
                    maxval = Param_range['max']
                    valrange = maxval-minval

                    # 探索ステップ幅の指定
                    stepval =valrange/(sample_size-1)

                    value_order = []
                    val_temp = minval

                    for i in range(sample_size):
                        value_order.append(val_temp)
                        val_temp += stepval # 値の更新
                else:
                    print('指定された探索方法はありません')

                y_temp_ep = []
                val_temp_ep = []

                #　↓　20210512_TOYOBO_KT-DX_Ito_Modify　↓　#
                #--ここから--#        
                for regressor in regressor_list:
                    y_temp_list,val_temp_list = calling_endpoint(
                        regressor=regressor, 
                        value_order=value_order, 
                        x_columns=x_columns, 
                        fixed_target=fixed_target, 
                        fixed_value=fixed_value, 
                        search_param=search_param, 
                        done_val_list=done_val_list, 
                        Optimal_value_dict=Optimal_value_dict, 
                        Search_range=Search_range, 
                        initial_search_value=initial_search_value
                    )
                    y_temp_ep.append(y_temp_list)
                #--ここまで--#
                #　↑　20210512_TOYOBO_KT-DX_Ito_Modify　↑　#

                # 探索が終了した変数を保存
                done_val_list.append(search_param)
                for i in range(len(target_list)):
                    y_dict_list[i][search_param] = list(map(float, y_temp_ep[i]))
                val_dict[search_param] = val_temp_list

                # 対象以外の探索済み変数の最適地を保存
                pre_ttl = []
                for c in range(len(target_const)):
                    pre_ttl_vol = 0
                    for a in target_const[c]:
                        pre_ttl_vol += float([0 if n is None else n for n in [Optimal_value_dict.get(a)]][0])
                    pre_ttl.append(pre_ttl_vol)

                # 最適値を保存
                if len(target_list) == 1:
                    if search_direction == 'max':
                        Optimal_value_dict[search_param] = val_temp_list[y_temp_ep[0].index(max(y_temp_ep[0]))]
                    elif search_direction == 'min':
                        Optimal_value_dict[search_param] = val_temp_list[y_temp_ep[0].index(min(y_temp_ep[0]))]

                # 複数ターゲットがある場合での最適値を探索
                else:
                    # ターゲット達成状況
                    target_achieve_list = []
                    target_achieve_index_list = []
                    for d in range(len(target_list)):
                        func = target_value_list[d]
                        target_df = pd.DataFrame(y_temp_ep[d])
                        achieve_value = len(target_df[func(target_df)].dropna(how='all'))
                        achieve_value_index = list(target_df[func(target_df)].dropna(how='all').index)
                        target_achieve_list.append(achieve_value)
                        target_achieve_index_list.append(achieve_value_index)

                    # 優先ターゲットの目標値がモデル上達成している場合、順次別ターゲットの最大値を取得するため目標達成可能な変数範囲内で探索を実施
                    # ヒューリスティックな方法のため実装内容を要検討
                    if target_achieve_list[0] > 0:
                        achieve_index = [i for i, x in enumerate(list(map(float, target_achieve_list))) if x>0]
                        first_target_index = target_achieve_index_list[0]
                        if sum(x > 0 for x in list(map(float, target_achieve_list))) > 1:
                            target_index = target_achieve_index_list[0]
                            for e in achieve_index:
                                y_merge_temp_index = list(set(target_index) & set(target_achieve_index_list[e]))
                                target_index = y_merge_temp_index
                            y_temp = []
                            if len(target_index) > 0:
                                for i in target_index:
                                    y_temp.append(list(map(float, y_temp_ep[max(achieve_index)]))[i])
                                target_y_temp_ep = y_temp_ep[max(achieve_index)]
                            else:
                                for i in first_target_index:
                                    y_temp.append(list(map(float, y_temp_ep[1]))[i])
                                target_y_temp_ep = y_temp_ep[1]
                        else:
                            y_temp = []
                            for i in first_target_index:
                                y_temp.append(list(map(float, y_temp_ep[1]))[i])
                            target_y_temp_ep = y_temp_ep[1]
                        if search_direction == 'max':
                            Optimal_value_dict[search_param] = val_temp_list[list(map(float, target_y_temp_ep)).index(max(y_temp))]
                        elif search_direction == 'min':
                            Optimal_value_dict[search_param] = val_temp_list[list(map(float, target_y_temp_ep)).index(min(y_temp))]
                    else:
                        if search_direction == 'max':
                            Optimal_value_dict[search_param] = val_temp_list[list(map(float, y_temp_ep[0])).index(max(list(map(float, y_temp_ep[0]))))]
                        elif search_direction == 'min':
                            Optimal_value_dict[search_param] = val_temp_list[list(map(float, y_temp_ep[0])).index(min(list(map(float, y_temp_ep[0]))))]

                # 物質量に対する制約
                # ターゲット群について、物質量が制約値になるよう最適値を調整
                uncheck_value = Optimal_value_dict[search_param]
                for c in range(len(target_const)):
                    unexplored_target = 0
                    ttl_vol = 0
                    for i in target_const[c]:
                        unexplored_target += [Optimal_value_dict.get(i)].count(None)
                        ttl_vol += float([0 if n is None else n for n in [Optimal_value_dict.get(i)]][0])

                    if ttl_vol > total_limit_value[c]:
                        Optimal_value_dict[search_param] = uncheck_value - (ttl_vol - total_limit_value[c])
                    elif search_param in target_const[c] and unexplored_target == 0:
                        Optimal_value_dict[search_param] = total_limit_value[c] - pre_ttl[c]

            #  探索結果の可視化
            for t in range(len(target_list)):
                search_process_fig(target=target_list[t],coef=coef, pin=sample_size, search_order=search_order, y_dict=y_dict_list[t],done_val_list=done_val_list) 

            print('=========最適値(説明変数)=========')
            print(Optimal_value_dict)
            print('=========最適値(目的変数)=========')
            y_temp_list = []
            l = []
            for k in range(len(x_columns)):
                if x_columns[k] in done_val_list:
                    l.append(Optimal_value_dict[x_columns[k]])
                else:
                    l.append(Search_range[x_columns[k]]['min'])

            #　↓　20210512_TOYOBO_KT-DX_Ito_Modify　↓　#
            #--ここから--#        
            X_temp = [l] # add
            for regressor in regressor_list: # add
                y_temp = regressor.predict(X_temp)[0]
                y_temp_list.append(y_temp)
            #--ここまで--#
            #　↑　20210512_TOYOBO_KT-DX_Ito_Modify　↑　#

            target_dict = {} # 探索後の出力値を目的変数毎に格納
            for i in range(len(target_list)):
                target_dict[target_list[i]] = y_temp_list[i] 
            print(target_dict)

            # 最適値周辺探索
            print('=========最適値周辺での探索結果(目的変数／説明変数　※探索変数以外は、最適値を利用)=========')
            print('探索方法：各ターゲット群で最も構成の多い物質について、最大量を10%増減(2%刻み※100molの場合2mol刻み)させた場合での推移')

            #　↓　20210512_TOYOBO_KT-DX_Ito_Modify　↓　#
            #--ここから--#        
            search_periphery_value(
                regressor_list=regressor_list, 
                target_list=target_list, 
                x_columns=x_columns, l=l, 
                target_const=target_const, 
                total_limit_value=total_limit_value, 
                done_val_list=done_val_list, 
                Search_range=Search_range, 
                Optimal_value_dict=Optimal_value_dict, 
                initial_search_value=initial_search_value
            )    
            #--ここまで--#
            #　↑　20210512_TOYOBO_KT-DX_Ito_Modify　↑　#
            '''
            
#------------------------------------------------------------
if __name__ == '__main__':
    None