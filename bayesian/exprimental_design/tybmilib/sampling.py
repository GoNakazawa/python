import numpy as np
import pandas as pd
import random
import time
import boto3
import os
import sagemaker
from tybmilib import modeling
from tybmilib import prep
from tybmilib import chemembeding
from tybmilib import myfilename as mfn
from tybmilib import logmgmt

Local_mode = mfn.get_localmode()
if Local_mode:
    from tqdm import tqdm # 通常
else:
    from tqdm import tqdm_notebook as tqdm # notebook用


# サンプル生成のクラス
class Sampling:
    def __init__(self, experiment_ID, combination_selects, combination_pairs, range_cols, range_pairs, fixed_cols, fixed_values,
                 ratio_selects, ratio_pairs, total_selects, total_values, group_selects, group_pairs, group_totals,
                 number_of_sampling=1000, conf_width=1, nega_flag=False, df=pd.DataFrame(data=[])):
        """サンプル生成クラスのコンストラクタ、引数を全て引き受ける
        Args:
            experiment_ID (str): 1st argument
            combination_selects (list): 2nd argument
            combination_pairs (list): 3rd argument
            range_cols (list): 4th argument
            range_pairs (list): 5th argument
            fixed_cols (list): 6th argument
            fixed_values (list): 7th argument
            ratio_selects (list): 8th argument
            ratio_pairs (list): 9th argument
            total_selects (list): 10th argument
            total_values (list): 11th argument
            group_selects (list): 12th argument
            group_pairs (list): 13th argument
            group_totals (list): 14th argument
            number_of_sampling (int): 15th argument
            conf_width (float): 16th argument
            nega_flag (bool): 17th argument
            df (pandas.DataFrame): 18th argument
        Returns:
            None
        
        """
        
        self.experiment_ID = experiment_ID
        if not df.empty:
            self.input_df = df.copy()
        else:
            data_path = mfn.get_samplingx_filename(experiment_ID, Local_mode=Local_mode)
            self.input_df = prep.read_csv(data_path, experiment_ID)

        self.combination_selects=combination_selects
        self.combination_pairs=combination_pairs
        self.range_cols=range_cols
        self.range_pairs=range_pairs
        self.fixed_cols=fixed_cols
        self.fixed_values=fixed_values
        self.ratio_selects=ratio_selects
        self.ratio_pairs=ratio_pairs
        self.total_selects=total_selects
        self.total_values=total_values
        self.group_selects = group_selects
        self.group_pairs = group_pairs
        self.group_totals = group_totals
        self.number_of_sampling=number_of_sampling
        self.conf_width=conf_width
        self.nega_flag=nega_flag
        self.range_mm = {}
        self.log_filename = mfn.get_step3_createsample_log_filename(experiment_ID, Local_mode=Local_mode)
        self.defined_cols = []
    
    ###################################
    # 制約条件可視化
    ###################################
    def describe_condition(self):
        """制約条件表示のラッパー関数
        Args:
            None
        Returns:
            None
        
        """
        self.describe_combination()
        self.describe_range()
        self.describe_fixed()
        self.describe_ratio()
        self.describe_total()
        self.describe_groupsum()

    def describe_combination(self):
        """組み合わせの表示
        Args:
            None
        Returns:
            None
        
        """
        print("COMBINATION")
        disp_list = []
        for i, combination_cols in enumerate(self.combination_selects):
            disp = "{0} < choice({1}) < {2}".format(self.combination_pairs[i][0], ", ".join(combination_cols), self.combination_pairs[i][1])
            disp_list.append(disp)
            print(disp)
        print()
        if not disp_list:
            disp_list.append("なし")
        return ", ".join(disp_list)
        
    def describe_range(self):
        """値範囲の表示
        Args:
            None
        Returns:
            None
        
        """
        print("RANGE")
        disp_list = []
        for i, col in enumerate(self.range_cols):
            disp = "{0} < {1} < {2}".format(self.range_pairs[i][0], col, self.range_pairs[i][1])
            disp_list.append(disp)
            print(disp)
        print()
        if not disp_list:
            disp_list.append("なし")
        return ", ".join(disp_list)

    def describe_fixed(self):
        """固定値の表示
        Args:
            None
        Returns:
            None
        
        """
        print("FIXED")
        disp_list = []
        for i, col in enumerate(self.fixed_cols):
            disp = "{0} = {1}".format(col, self.fixed_values[i])
            disp_list.append(disp)
            print(disp)
        print()
        if not disp_list:
            disp_list.append("なし")
        return ", ".join(disp_list)

    def describe_ratio(self):
        """比率の表示
        Args:
            None
        Returns:
            None
        
        """
        print("RATIO")
        disp_list = []
        for i, ratio_cols in enumerate(self.ratio_selects):
            disp = "{0}:{1} = {2}:{3}".format(ratio_cols[0], ratio_cols[1], self.ratio_pairs[i][0], self.ratio_pairs[i][1])
            disp_list.append(disp)
            print(disp)
        print()
        if not disp_list:
            disp_list.append("なし")
        return ", ".join(disp_list)

    def describe_total(self):
        """合計の表示
        Args:
            None
        Returns:
            None
        
        """
        print("TOTAL")
        disp_list = []
        for i, total_cols in enumerate(self.total_selects):
            disp = "{0} = {1}".format(" + ".join(total_cols), self.total_values[i])
            disp_list.append(disp)
            print(disp)
        print()
        if not disp_list:
            disp_list.append("なし")
        return ", ".join(disp_list)

    def describe_groupsum(self):
        """グループ和の表示
        Args:
            None
        Returns:
            None
        
        """
        print("GROUP SUM")
        disp_list = []
        for j, group_list in enumerate(self.group_selects):
            disp_part_list = []
            for i, gr in enumerate(group_list):
                disp_part = "({0}<choice({1})<{2})".format(self.group_pairs[j][i][0], ",".join(gr), self.group_pairs[j][i][1])
                disp_part_list.append(disp_part)
            disp_part_concat = (" + ").join(disp_part_list)
            disp = "{0} = {1}".format(disp_part_concat, self.group_totals[j])
            disp_list.append(disp)
            print(disp)
        print()
        if not disp_list:
            disp_list.append("なし")
        return ", ".join(disp_list)
    
    def describe_sampling_range(self):
        """サンプリング範囲の表示
        Args:
            None
        Returns:
            None
        
        """
        print("SAMPLING RANGE (INT)")
        for col in self.range_mm:
            print("{0} < {1} < {2}".format(round(self.range_mm[col]["mn"]), col, round(self.range_mm[col]["mx"])))
        print()

    
    ###################################
    # サンプリング範囲の計算
    ###################################
    def calc_sampling_range(self, df):
        """値範囲を最小値と最大値で決定
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            None
        
        """
        df_sampling = df.copy()

        # 辞書型のデータ定義
        range_mm = {}
        for df_col in df.columns:
            range_mm[df_col] = {}

        # 各説明変数のサンプリング範囲を最小値以上、最大値以下に設定
        for col in df.columns:
            range_mm[col]["mn"] = df_sampling[col].min()
            range_mm[col]["mx"] = df_sampling[col].max()
            value_width = range_mm[col]["mx"] - range_mm[col]["mn"]
            value_middle = (range_mm[col]["mx"] + range_mm[col]["mn"]) / 2

            # サンプリング範囲を、conf_widthに応じた範囲に拡大
            ex_value_width = value_width * self.conf_width
            range_mm[col]["mn"] = value_middle - (ex_value_width / 2)
            range_mm[col]["mx"] = value_middle + (ex_value_width / 2)

            # nega_flagがFalseの場合、負の範囲を除外する
            if self.nega_flag == False and range_mm[col]["mn"] - ex_value_width < 0:
                range_mm[col]["mn"] = 0

        return range_mm

    ###################################
    # サンプリング実行
    ###################################
    def create_samples(self):
        """サンプル生成のラッパー関数
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_groupsum
        
        """
        # 初期の値範囲設定
        self.range_mm = self.calc_sampling_range(self.input_df)

        # データフレーム作成
        zeros = np.zeros((self.number_of_sampling, len(self.input_df.columns)))
        ini_df = pd.DataFrame(data=zeros, columns=self.input_df.columns)

        # 各制約条件に基づいてサンプリング
        df_combination = self.sampling_combination(ini_df)
        df_groupsum_prepare = self.sampling_groupsum_prepare(df_combination)
        df_default = self.sampling_default(df_groupsum_prepare)
        df_range = self.sampling_range(df_default)
        df_fixed = self.sampling_fixed(df_range)
        df_ratio = self.sampling_ratio(df_fixed)
        df_total = self.sampling_total(df_ratio)
        df_remain = self.sampling_remained(df_total)
        df_check = self.sampling_check(df_remain)
        df_groupsum = self.sampling_groupsum(df_check)
        
        # 最終的なサンプリング範囲
        self.range_mm = self.recalc_sampling_range(df_groupsum)

        # サンプル生成結果の行数表示
        print("サンプリング回数 {}".format(self.number_of_sampling))
        print("サンプル生成数 {}".format(len(df_groupsum)))

        return df_groupsum


    def sampling_combination(self, df):
        """組み合わせの判定
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        # 組み合わせで指定された列は、スパースに値を埋める
        for i, combination_cols in enumerate(self.combination_selects):
            for row in range(len(df_sampling)):
                choices = random.sample(combination_cols, random.randint(self.combination_pairs[i][0]+1, self.combination_pairs[i][1]-1))
                for choice in choices:
                    df_sampling[choice][row] = 1
        
        self.describe_combination()
        return df_sampling


    def sampling_groupsum_prepare(self, df):
        """グループ和の判定（前半、サンプリング箇所を設定）
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        # グループで選ばれる個数を確保できない場合、その行を削除
        np_drop_idx = np.array([])
        for j, group_list in enumerate(self.group_selects):
            for i, gr in enumerate(group_list):
                for row in range(len(df_sampling)):
                    # 既にCOMBINATIONで選択されている列数を抽出（COMBINATIONを優先するため）
                    df_choice = df_sampling[list(gr)].iloc[row]
                    df_choice_bool = df_choice[df_choice!=0]
                    e_choice_list = df_choice_bool.index.tolist()
                    # 固定値、合計で指定された列の値は変更しないため、選ばれないように設定
                    choice_pool = list(set(gr) - set(e_choice_list))
                    e_choice_num = len(e_choice_list)
                    # 行ごとに判定。選ばれる候補列の数が、既に指定されている列数よりも小さい場合に、サンプリングを行う
                    lower_int = self.group_pairs[j][i][0]+1
                    upper_int = self.group_pairs[j][i][1]-1
                    
                    if lower_int <= len(choice_pool) and upper_int >= e_choice_num:
                        feature_int = random.randint(lower_int, upper_int)
                        if feature_int > len(choice_pool):
                            feature_int = len(choice_pool)
                        choices = random.sample(choice_pool, feature_int)
                        for choice in choices:
                            df_sampling[choice][row] = 1
                    # それ以外の行は削除リストへ
                    else:
                        np_drop_idx = np.append(np_drop_idx, row)
        # グループで選ばれる個数を確保できない場合、その行を削除
        np_drop_idx = np.unique(np_drop_idx)
        df_sampling = df_sampling.drop(index=np_drop_idx).reset_index(drop=True)
        return df_sampling

    def sampling_default(self, df):
        """組み合わせ、グループ和で指定されなかった列にサンプリング個所を設定
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()
        all_combination_cols = [e for row in self.combination_selects for e in row]
        all_group_cols = [e for group_list in self.group_selects for row in group_list for e in row]
        remained_cols = list(set(df.columns) - set(all_combination_cols) - set(all_group_cols))
        
        for col in remained_cols:
            df_sampling[col] = 1
            
        return df_sampling


    def sampling_range(self, df):
        """値範囲の判定（値範囲の更新）
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        # サンプリング範囲を、値範囲で設定された範囲に変更して、値を埋める
        for i, col in enumerate(self.range_cols):
            self.range_mm[col]["mn"] = self.range_pairs[i][0]
            self.range_mm[col]["mx"] = self.range_pairs[i][1]

        self.describe_range()
        return df_sampling    


    def sampling_fixed(self, df):
        """固定値の判定
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        # 固定値で指定された列は、必ず固定値で埋める
        for col, val in zip(self.fixed_cols, self.fixed_values):
            if (self.range_mm[col]["mn"] > val or self.range_mm[col]["mx"] < val):
                error_msg = "Error: 指定した{}の固定値は、サンプリング範囲外です".format(col)
                print(error_msg)
                logmgmt.raiseError(self.experiment_ID, error_msg, self.log_filename)
            else:
                # 値範囲を固定値のみに限定
                self.range_mm[col]["mn"] = self.range_mm[col]["mx"] = val
                # サンプリング
                df_sampling[col] *= val

        self.defined_cols = self.fixed_cols.copy()
        self.describe_fixed()
        return df_sampling


    def sampling_ratio(self, df):
        """比例の判定
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        temp_fixed_cols = self.defined_cols.copy()
        for ratio_cols, ratio in zip(self.ratio_selects, self.ratio_pairs):
            # 固定値で指定された列は、値を変更しない
            # 値範囲で指定された列は、値範囲の中でサンプリングを行う
            fixed_ratio_cols = list(set(temp_fixed_cols) & set(ratio_cols))
            range_ratio_cols = list((set(self.range_cols) & set(ratio_cols)) - set(fixed_ratio_cols))

            # 固定値で両方指定されている場合、処理スキップ
            if len(fixed_ratio_cols) == 2:
                error_msg = "Error: 比率で指定した列{}は、両方とも固定値で指定されています".format(", ".join(fixed_ratio_cols))
                print(error_msg)
                logmgmt.raiseError(self.experiment_ID, error_msg, self.log_filename)

            # 固定値で片方指定されている場合、もう片方を比率で合わせる
            elif len(fixed_ratio_cols) == 1:
                fixed_col = fixed_ratio_cols[0]
                other_col = list(set(fixed_ratio_cols) ^ set(ratio_cols))[0]
                ratio_scale = ratio[ratio_cols.index(other_col)] / ratio[ratio_cols.index(fixed_col)]

                # もう片方が値範囲で指定されている場合、値範囲内に入っているかを判定
                scaled_value = self.range_mm[fixed_col]["mn"] * ratio_scale
                if len(range_ratio_cols) == 1:
                    ratio_lower = self.range_mm[other_col]["mn"]
                    ratio_upper = self.range_mm[other_col]["mx"]
                    if scaled_value < ratio_lower or ratio_upper < scaled_value:
                        error_msg = "Error: 列{}で、比率とサンプリング範囲が競合しています".format(other_col)
                        print(error_msg)
                        logmgmt.raiseError(self.experiment_ID, error_msg, self.log_filename)

                # サンプリング
                df_sampling[other_col] = df_sampling[fixed_col] * ratio_scale
                temp_fixed_cols.append(other_col)
                # 固定値に合わせた比率計算のため、固定値側へ追加
                if len(df_sampling[other_col].unique()) <= 2:
                    self.range_mm[other_col]["mn"] = self.range_mm[other_col]["mx"] = scaled_value
                    self.fixed_cols.append(other_col)
                    self.defined_cols.append(other_col)

            # 固定値で指定されていない場合
            elif len(fixed_ratio_cols) == 0:

                # 比率に合わせて、値範囲を更新してサンプリング
                def calc_both_range(df, range_mm, colA, colB, ratio_scale):
                    df_sampling=df.copy()

                    x_upper = range_mm[colA]["mx"] * ratio_scale
                    x_lower = range_mm[colA]["mn"] * ratio_scale
                    y_upper = range_mm[colB]["mx"] / ratio_scale
                    y_lower = range_mm[colB]["mn"] / ratio_scale

                    range_mm[colA]["mx"] = range_mm[colA]["mx"] if range_mm[colA]["mx"] < y_upper else y_upper
                    range_mm[colA]["mn"] = range_mm[colA]["mn"] if range_mm[colA]["mn"] > y_lower else y_lower
                    range_mm[colB]["mx"] = range_mm[colB]["mx"] if range_mm[colB]["mx"] < x_upper else x_upper
                    range_mm[colB]["mn"] = range_mm[colB]["mn"] if range_mm[colB]["mn"] > x_lower else x_lower

                    # 比率を考慮した結果、値範囲との矛盾が生じた場合はエラー
                    if range_mm[colA]["mx"] < range_mm[colA]["mn"] or range_mm[colB]["mx"] < range_mm[colB]["mn"]:
                        error_msg = "Error: 列{}と{}で、比率とサンプリング範囲が競合しています".format(colA,colB)
                        print(error_msg)
                        logmgmt.raiseError(self.experiment_ID, error_msg, self.log_filename)
                        #return df_sampling

                    df_sampling[colA] *= [random.uniform(range_mm[colA]["mn"], range_mm[colA]["mx"]) for i in range(len(df_sampling))]
                    s_cond = (df_sampling[colA] > 0) & (df_sampling[colB] > 0)
                    df_sampling[colB][s_cond] *= df_sampling[colA] * ratio_scale
                    df_sampling[colB][~s_cond] *= [random.uniform(range_mm[colB]["mn"], range_mm[colB]["mx"]) for i in range(len(df_sampling[colB][~s_cond]))]
                    return df_sampling

                def calc_either_range(df, range_mm, colA, colB, ratio_scale):
                    df_sampling=df.copy()

                    range_mm[colB]["mn"] = range_mm[colA]["mn"] * ratio_scale
                    range_mm[colB]["mx"] = range_mm[colA]["mx"] * ratio_scale

                    df_sampling[colA] *= [random.uniform(range_mm[colA]["mn"], range_mm[colA]["mx"]) for i in range(len(df_sampling))]
                    s_cond = (df_sampling[colA] > 0) & (df_sampling[colB] > 0)
                    df_sampling[colB][s_cond] *= df_sampling[colA] * ratio_scale
                    df_sampling[colB][~s_cond] *= [random.uniform(range_mm[colB]["mn"], range_mm[colB]["mx"]) for i in range(len(df_sampling[colB][~s_cond]))]
                    return df_sampling

                # 片方のみ値範囲で指定されている場合、値範囲で指定されていないサンプリング範囲を変更
                if len(range_ratio_cols) == 1:
                    col = range_ratio_cols[0]
                    other_col = other_col = list(set(ratio_cols) - set(col))[0]
                    ratio_scale = ratio[ratio_cols.index(other_col)] / ratio[ratio_cols.index(col)]
                    df_sampling = calc_either_range(df_sampling, self.range_mm, col, other_col, ratio_scale)

                # 両方とも値範囲で指定されている、もしくは両方とも値範囲で指定されていない場合、両方のサンプリング範囲を変更
                else:
                    if len(range_ratio_cols) == 2:
                        print("Warning: 比率と値範囲が競合しているため、両方を満たすサンプリング範囲が設定されます。値範囲のサンプリング範囲ではなくなりました")
                    if ratio[0] < ratio[1]:            
                        ratio_scale = ratio[1] / ratio[0]
                        df_sampling = calc_both_range(df_sampling, self.range_mm, ratio_cols[0], ratio_cols[1], ratio_scale)
                    else:
                        ratio_scale = ratio[0] / ratio[1]
                        df_sampling = calc_both_range(df_sampling, self.range_mm, ratio_cols[1], ratio_cols[0], ratio_scale)
                temp_fixed_cols.append(ratio_cols[0])
                temp_fixed_cols.append(ratio_cols[1])

        self.describe_ratio()
        return df_sampling    


    def sampling_total(self, df):
        """合計値の判定
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        temp_fixed_cols = self.fixed_cols.copy()
        for total_cols, total_val in zip(self.total_selects, self.total_values):
            # 条件判定用
            fixed_total_cols = list(set(temp_fixed_cols) & set(total_cols))
            not_fixed_cols = list(set(fixed_total_cols) ^ set(total_cols))
            ratio_cols = [e for row in self.ratio_selects for e in row]
            ratio_total_cols = list(set(ratio_cols) & set(total_cols) & set(not_fixed_cols))

            # 固定値の和が、合計値を超えている場合はエラー
            fixed_sum = 0
            for col in fixed_total_cols:
                fixed_sum += self.range_mm[col]["mx"]
                if fixed_sum > total_val:
                    error_msg = "Error: 列{}は固定値と競合しているため、指定した合計値を満たすことができません".format(", ".join(total_cols))
                    print(error_msg)
                    logmgmt.raiseError(self.experiment_ID, error_msg, self.log_filename)

            # 他の条件で指定されていないが、合計で指定されている列について、適当にサンプリングする
            for col in not_fixed_cols:
                if col not in ratio_total_cols:
                    df_sampling[col] *= [random.uniform(self.range_mm[col]["mn"], self.range_mm[col]["mx"]) for i in range(len(df_sampling))]
            
            # 各行について、合計で指定された列の和を計算し、正規化して値を和に合わせる
            df_sum = df_sampling[total_cols].sum(axis=1)
            # 固定で指定された列については、値を変更しない
            df_fixed_sum = df_sampling[fixed_total_cols].sum(axis=1)
            for col in not_fixed_cols:
                df_sampling[col] = df_sampling[col] / (df_sum - df_fixed_sum) * (total_val - df_fixed_sum)
                # 合計で指定された列は、固定値に追加
                temp_fixed_cols.append(col)
                self.defined_cols.append(col)

            # 固定で指定されておらず、比率で指定されており、合計で指定されていない列を修正
            for col in ratio_total_cols:
                for ratio_cols in self.ratio_selects:
                    other_col = list(set(col) ^ set(ratio_cols))[0]
                    if (col in ratio_cols) and (other_col not in ratio_total_cols):
                        r_i = self.ratio_selects.index(ratio_cols)
                        ratio = self.ratio_pairs[r_i][ratio_cols.index(other_col)] / self.ratio_pairs[r_i][ratio_cols.index(col)]
                        s_cond = (df_sampling[col] > 0) & (df_sampling[other_col] > 0)
                        df_sampling[other_col][s_cond] = df_sampling[col] * ratio
                        # 合計で指定された列は、固定値に追加
                        temp_fixed_cols.append(col)
                        self.defined_cols.append(col)

        # 正規化の結果でn/aになったレコードを、0に変換
        df_sampling = df_sampling.fillna(0)
        self.describe_total()
        return df_sampling


    def sampling_remained(self, df):
        """条件指定されていない列のサンプリング、値範囲に合わせてランダム
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        # これまでの条件に指定されなかった列をサンプリング
        ratio_cols = [e for row in self.ratio_selects for e in row]
        total_cols = [e for row in self.total_selects for e in row]
        not_defined_cols = list(set(df.columns.tolist()) ^ set(self.defined_cols + ratio_cols + total_cols))

        for col in not_defined_cols:
            df_sampling[col] *= [random.uniform(self.range_mm[col]["mn"], self.range_mm[col]["mx"]) for i in range(len(df_sampling))]

        return df_sampling


    def sampling_check(self, df):
        """各行の最終チェック、及び条件外のレコード削除（グループ和の合計以外）
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling
        
        """
        df_sampling = df.copy()

        # 値範囲で指定された列について、値範囲外の数値がある場合、行ごと削除
        for col in self.range_cols:
            col_cond = ~((self.range_mm[col]["mn"] <= df_sampling[col]) & (df_sampling[col] <= self.range_mm[col]["mx"]))
            zero_cond = (df_sampling[col] != 0)
            df_not = df_sampling[(col_cond & zero_cond)]
            df_sampling = df_sampling.drop(df_not.index)

        # 組み合わせでスパースかつ固定値指定などで、合計値を満たしていない場合、行ごと削除
        for i, total_cols in enumerate(self.total_selects):
            col_cond = df_sampling[total_cols].sum(axis=1).round() != self.total_values[i]
            df_not = df_sampling[col_cond]
            df_sampling = df_sampling.drop(df_not.index)

        # 負の値が入っている行を削除
        if self.nega_flag==False:
            for col in df.columns.tolist():
                nega_cond = df_sampling[col] < 0
                df_not = df_sampling[nega_cond]
                df_sampling = df_sampling.drop(df_not.index)

        return df_sampling.reset_index(drop=True)

 
    def sampling_groupsum(self, df):
        """グループ和の設定（後半：和を合わせる）
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            pandas.DataFrame: df_sampling[df_cond]
        
        """
        df_sampling = df.copy().reset_index(drop=True)
        temp_defined_cols = self.defined_cols.copy()

        # グループごとに、固定値と合計で指定された列以外を正規化し、グループ和に合わせる
        for j, group_list in enumerate(self.group_selects):
            group_cols = [e for row in group_list for e in row]
            defined_group_cols = list(set(group_cols) & set(temp_defined_cols))
            not_defined_group_cols = list(set(group_cols) - set(temp_defined_cols))

            # 各行について、グループ和で指定された列の和を計算し、正規化して値を和に合わせる
            df_sum = df_sampling[group_cols].sum(axis=1)
            # 固定値と合計で指定された列については、値を変更しない
            df_defined_sum = df_sampling[defined_group_cols].sum(axis=1)
            for col in not_defined_group_cols:
                df_sampling[col] = df_sampling[col] / (df_sum - df_defined_sum) * (self.group_totals[j] - df_defined_sum)
                # グループ和で指定された列は、固定値に追加
                temp_defined_cols.append(col)
            df_sampling = df_sampling.fillna(0)
            temp_defined_cols = list(set(temp_defined_cols))

            # 固定で指定されておらず、比率で指定されており、グループ和で指定されていない列を修正
            ratio_cols = [e for row in self.ratio_selects for e in row]
            ratio_group_cols = list(set(group_cols) & set(ratio_cols) & set(not_defined_group_cols))
            for col in ratio_group_cols:
                for ratio_cols in self.ratio_selects:
                    other_col = list(set(ratio_cols) - set(col))[0]
                    if (col in ratio_cols) and (other_col not in ratio_group_cols):
                        r_i = self.ratio_selects.index(ratio_cols)
                        ratio = self.ratio_pairs[r_i][ratio_cols.index(other_col)] / self.ratio_pairs[r_i][ratio_cols.index(col)]
                        s_cond = (df_sampling[col] > 0) & (df_sampling[other_col] > 0)
                        df_sampling[other_col][s_cond] = df_sampling[col] * ratio

        # グループ和を満たしていない行を削除（フィルタリング条件生成）
        df_cond = pd.Series(np.random.rand(len(df_sampling)))
        df_cond[:] = True
        for j, group_list in enumerate(self.group_selects):
            group_cols = [e for row in group_list for e in row]
            df_cond = df_cond & (df_sampling[group_cols].sum(axis=1).round(10) == self.group_totals[j])
        self.describe_groupsum()
        return df_sampling[df_cond]


    def recalc_sampling_range(self, df):
        """サンプル生成結果のサンプリング範囲
        
        Args:
            df (pandas.DataFrame): 1st argument
        Returns:
            dict: new_range_mm
        
        """
        df_sampling = df.copy()
        new_range_mm = self.range_mm.copy()
        
        for col in df.columns:
            if self.range_mm[col]["mn"] != self.range_mm[col]["mx"]:
                new_range_mm[col]["mn"] = df_sampling[col].min()
                new_range_mm[col]["mx"] = df_sampling[col].max()

        return new_range_mm



# モデル推論用のクラス
class Inference:
    def __init__(self, experiment_ID, user_ID, bucket_name):
        """モデル推論用のクラスのコンストラクタ
        
        Args:
            experimentID (str): 1st argument
            user_ID (str): 2nd argument
            bucket_name (str): 3rd argument
        Returns:
            None
        
        """
        self.experiment_ID = experiment_ID
        self.user_ID = user_ID
        self.bucket = bucket_name
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session(boto3.session.Session())


    def convert_chemical_structure(self, chemical_feature, samples):
        """モデル推論時に、分子構造組込への変換
        
        Args:
            chemical_feature (class): 1st argument
            samples (pandas.DataFrame): 2nd argument
        Returns:
            pandas.DataFrame: samples_predicted
        
        """
        if chemical_feature != '':
            # データフレームを化学構造表現を含む特徴量に変換
            samples_predicted = chemical_feature.generate_fingerprint_dataset(samples,objectives=[])
        else:
            samples_predicted = samples.copy()
        return samples_predicted


    def model_inference(self, model_name, samples_predicted, target):
        """モデル推論実行
        
        Args:
            model_name (str): 1st argument
            samples_predicted (pandas.DataFrame): 2nd argument
            target (str): 3rd argument
        Returns:
            numpy.array: predicted
        
        """
        btr = modeling._AutopilotBatchjobRegressor(model_name, self.experiment_ID, self.user_ID, self.bucket, self.region)
        predicted = btr.predict(samples_predicted, target, sampling=True)
        return predicted


    def multi_model_inference(self, samples, objectives, model_list, chemical_feature):
        """モデル推論実行
        
        Args:
            samples (pandas.DataFrame): 1st argument
            objectives (list): 2nd argument
            moedl_list (list): 3rd argument
            chemical_feature (class): 4th argument
        Returns:
            pandas.DataFrame: result_samples
        
        """
        # 化学構造への変換
        samples_converted = self.convert_chemical_structure(chemical_feature, samples)
        
        # モデル推論
        pre_list = []
        process = tqdm(range(len(objectives)))
        process.set_description("predicting SampleData")
        for j in process:
            predicted = self.model_inference(model_list[j], samples_converted, objectives[j])
            pre_list.append(predicted)
        pre_array = np.vstack(pre_list).T
        ys = pd.DataFrame(pre_array,index=samples.index,columns=objectives)
        result_samples = pd.concat([samples,ys], axis=1)

        sample_filename = mfn.get_sample_filename(self.experiment_ID, Local_mode=Local_mode)
        result_samples.to_csv(sample_filename, index=False, sep=',')
        return  result_samples