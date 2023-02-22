from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, session, Response, send_from_directory, flash
import pandas as pd
from tybmilib import myfilename as mfn
from tybmilib import logmgmt
from tool_app import viewlib as vlib

class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするたためのクラス
    """
    pass

#チェックボックスのリストを分割
def split_list(l, s):
    length = len(l)
    n=0
    ans=[]
    for i in l:
        ans.append(l[n:n+s:1])
        n+=s
        if n>=length:
            break
    return ans

# 制約条件の設定
def check_number(num1, num2 = ""):
    if num1 != "" and num2 == "":
        try:
            num1_f = float(num1)
        except:
            return False
    elif num1 == "" and num2 != "":
        try:
            num2 = float(num2)
        except:
            return False
    elif num1 != "" and num2 != "":
        try:
            num1_f = float(num1)
            num2_f = float(num2)
        except:
            return False
    return True

# 制約条件の引数を設定するクラス
class SetCondition:
    def __init__(self, exp_id, df_columns, range_mm):
        self.exp_id = exp_id
        self.df_columns = df_columns
        self.range_mm = range_mm
        self.step_all_log = mfn.get_step_all_log_filename(exp_id)

    # エラー時、メッセージを更新してリダイレクト
    def error_redirect(self, error_msg):
        vlib.update_error(self.exp_id, error_msg, "step3")
        logmgmt.logError(self.exp_id, error_msg, self.step_all_log)
        raise Lib_ParseError(error_msg)

    # 組み合わせの条件について、サンプリング用の引数を作成
    def get_combination_condition(self, combination_cb, combination_lower, combination_upper):
        # 組み合わせを指定
        combination_cb_n = split_list(combination_cb, len(self.df_columns))
        combination_selects = []
        combination_pairs = []
        combination_dict = {"target": [], "combination_lower": [], "combination_upper": []}

        for i in range(len(combination_cb_n)):
            lower = combination_lower[i]
            upper = combination_upper[i]
            empty_cb = all([x=='' for x in combination_cb_n[i]])
            if empty_cb or (lower == '' and upper == ''):
                continue 
            else:
                if check_number(lower, upper) == False:
                    error_msg = "Error : [COMBINATION] 制約条件には数値を入力してください"
                    self.error_redirect(error_msg)
                # チェックされた列を代入
                combination_list = []
                for elm in combination_cb_n[i]:
                    if elm != '':
                        combination_list.append(elm)
                # 上限値と下限値の確認
                if lower != "" and upper == "":
                    lower_i = int(lower)
                    upper_i = len(combination_list) + 1
                elif lower == "" and upper != "":
                    lower_i = -1
                    upper_i = int(upper)
                elif lower != "" and upper != "":
                    lower_i = int(lower)
                    upper_i = int(upper)

                # エラー判定
                if lower_i >= upper_i:
                    error_msg = "Error : [COMBINATION] 上限値より下限値に大きい値が入力されました"
                    self.error_redirect(error_msg)
                if len(combination_list) <= lower_i or upper_i > len(combination_list) + 1  or lower_i < -1 or upper_i < -1:
                    error_msg = "Error : [COMBINATION] チェックした個数が指定した組み合わせ個数の範囲外です"
                    self.error_redirect(error_msg)
                if upper_i-lower_i == 1:
                    error_msg = "Error : [COMBINATION] 選択数の差は2以上にしてください"
                    self.error_redirect(error_msg)

                # 引数作成
                combination_selects.append(combination_list)                
                combination_pairs.append((lower_i, upper_i))
                combination_dict["target"].append(combination_cb_n[i])
                combination_dict["combination_lower"].append(lower)
                combination_dict["combination_upper"].append(upper)
        return combination_selects, combination_pairs, combination_dict

    # 値範囲の条件について、サンプリング用の引数を作成
    def get_range_condition(self, range_sel, range_lower, range_upper):
        range_cols = []
        range_pairs = []
        range_dict = {"target": [], "range_lower": [], "range_upper": []}

        for i in range(len(range_sel)):
            lower = range_lower[i]
            upper = range_upper[i]
            if range_sel[i]=='' or (lower == '' and upper == ''):
                continue
            else:
                if check_number(lower, upper) == False:
                    error_msg = 'Error: [RANGE] 制約条件には数値を入力してください'
                    self.error_redirect(error_msg)
                # 上限値と下限値の確認
                if lower != "" and upper == "":
                    lower_f = float(lower)
                    upper_f = round(self.range_mm[range_sel[i]]["mx"], 2)
                elif lower == "" and upper != "":
                    lower_f = round(self.range_mm[range_sel[i]]["mn"], 2)
                    upper_f = float(upper)
                elif lower != "" and upper != "":
                    lower_f = float(lower)
                    upper_f = float(upper)
                
                # エラー判定
                if lower_f > upper_f:
                    error_msg = 'Error : [RANGE] 上限値より下限値に大きい値が入力されました'
                    self.error_redirect(error_msg)
                
                # 引数作成
                range_cols.append(range_sel[i])
                range_pairs.append((lower_f, upper_f))
                range_dict["target"].append(range_sel[i])
                range_dict["range_lower"].append(lower)
                range_dict["range_upper"].append(upper)
        return range_cols, range_pairs, range_dict

    # 固定値の条件について、サンプリング用の引数を作成
    def get_fixed_condition(self, fixed_sel, fixed_val):
        fixed_cols = []
        fixed_values = []
        fixed_dict = {"target": [], "value": []}

        for i in range(len(fixed_sel)):
            if fixed_sel[i] == '' or fixed_val[i] == '':
                continue
            else:
                if check_number(fixed_val[i]) == False:
                    error_msg = 'Error : [FIXED] 制約条件には数値を入力してください'
                    self.error_redirect(error_msg)

                # 引数作成
                fixed_cols.append(fixed_sel[i])
                fixed_values.append(float(fixed_val[i]))
                fixed_dict["target"].append(fixed_sel[i])
                fixed_dict["value"].append(fixed_val[i])
        return fixed_cols, fixed_values, fixed_dict

    # 固定値の条件について、サンプリング用の引数を作成
    def get_ratio_condition(self, ratio1_sel, ratio2_sel, ratio1_val, ratio2_val):
        ratio_selects = []
        ratio_pairs = []
        ratio_dict = {"target1": [], "target2": [], "ratio1": [], "ratio2": []}

        for i in range(len(ratio1_val)):
            val1 = ratio1_val[i]
            val2 = ratio2_val[i]
            if (ratio1_sel[i] == '' or ratio2_sel[i] == '') or (val1 == '' or val2 == ''):
                continue
            else:
                if check_number(val1, val2) == False:
                    error_msg = "Error : [RATIO] 制約条件には数値を入力してください"
                    self.error_redirect(error_msg)
                # エラー判定
                if ratio1_sel[i] == ratio2_sel[i]:
                    error_msg = "Error : [RATIO] 同じ列を選択しています"
                    self.error_redirect(error_msg)
                if float(val1) <= 0 or float(val2) <= 0:
                    error_msg = "Error : [RATIO] 設定した値が0以下です"
                    self.error_redirect(error_msg)
                
                # 引数作成
                ratio_selects.append([ratio1_sel[i], ratio2_sel[i]])
                ratio_pairs.append((float(val1), float(val2)))
                ratio_dict["target1"].append(ratio1_sel[i])
                ratio_dict["target2"].append(ratio2_sel[i])
                ratio_dict["ratio1"].append(val1)
                ratio_dict["ratio2"].append(val2)
        return ratio_selects, ratio_pairs, ratio_dict

    # 合計値の条件について、サンプリング用の引数を作成
    def get_total_condition(self, total_cb, total_val):
        total_cb_n = split_list(total_cb, len(self.df_columns))
        total_selects = []
        total_values = []
        total_dict = {"target": [], "total": []}

        for i in range(len(total_cb_n)):
            empty_cb = all([x=='' for x in total_cb_n[i]])
            if empty_cb or total_val[i] == '':
                continue
            else:
                if check_number(total_val[i]) == False:
                    error_msg = 'Error : [TOTAL] 制約条件には数値を入力してください'
                    self.error_redirect(error_msg)
                # チェックされた列を代入
                total_list = []
                for elm in total_cb_n[i]:
                    if elm != '':
                        total_list.append(elm)
                
                # 引数作成
                total_selects.append(total_list)
                total_values.append(float(total_val[i]))
                total_dict["target"].append(total_cb_n[i])
                total_dict["total"].append(total_val[i])
        return total_selects, total_values, total_dict

    # グループ和の条件について、サンプリング用の引数を作成
    def get_groupsum_condition(self, group_cb_list, group_lower_list, group_upper_list, group_total_vals):
        group_selects = []
        group_pairs = []
        group_totals = []
        group_dict_list = []
        group_dict_total = {"total": []}

        for j, group_cb in enumerate(group_cb_list):
            group_cb_n = split_list(group_cb, len(self.df_columns))
            group_selects_parts = []
            group_pairs_parts = []
            group_dict = {"target": [], "lower": [], "upper": []}

            # 先に合計値の判定
            if group_total_vals[j] == "":
                continue
            else:
                if check_number(group_total_vals[j]) == False:
                    error_msg = 'Error : [GROUPSUM] 合計には数値を入力してください'
                    self.error_redirect(error_msg)

                for i in range(len(group_cb_n)):
                    empty_cb = all([x=='' for x in group_cb_n[i]])
                    lower = group_lower_list[j][i]
                    upper = group_upper_list[j][i]
                    if empty_cb or (lower == '' and upper == ''):
                        continue 
                    else:
                        if check_number(lower, upper) == False:
                            error_msg = 'Error : [GROUPSUM] 選択数には数値を入力してください'
                            self.error_redirect(error_msg)
                        
                        # チェックされた列を代入                        
                        group_list = []
                        for elm in group_cb_n[i]:
                            if elm != '':
                                group_list.append(elm)

                        # 上限値と下限値の確認
                        if lower != "" and upper == "":
                            lower_i = int(lower)
                            upper_i = len(group_list) + 1
                        elif lower == "" and upper != "":
                            lower_i = -1
                            upper_i = int(upper)
                        elif lower != "" and upper != "":
                            lower_i = int(lower)
                            upper_i = int(upper)
                        # エラー判定
                        if lower_i >= upper_i:
                            error_msg = "Error : [GROUPSUM] 上限値より下限値に大きい値が入力されました"
                            self.error_redirect(error_msg)
                        if len(group_list) <= lower_i or upper_i > len(group_list) + 1 or lower_i < -1 or upper_i < -1:
                            error_msg = "Error : [GROUPSUM] チェックした個数が指定した組み合わせ個数の範囲外です"
                            self.error_redirect(error_msg)
                        if upper_i-lower_i == 1:
                            error_msg = "Error : [GROUPSUM] 選択数の差は2以上にしてください"
                            self.error_redirect(error_msg)
                        
                        # 引数の要素作成
                        group_selects_parts.append(group_list)
                        group_pairs_parts.append((lower_i, upper_i))
                    group_dict["target"].append(group_cb_n[i])
                    group_dict["lower"].append(lower)
                    group_dict["upper"].append(upper)
                # 引数作成
                group_selects.append(group_selects_parts)
                group_pairs.append(group_pairs_parts)
                group_totals.append(float(group_total_vals[j]))
            group_dict_list.append(group_dict)
            group_dict_total["total"].append(group_total_vals[j])

        return group_selects, group_pairs, group_totals, group_dict_list, group_dict_total

    def get_target_condition(self, target_sel, target_lower, target_upper):
        target_cols = []
        target_pairs = []
        if target_sel==[""]:
            error_msg = 'Error: [TARGET] 目的変数が設定されていません'
            self.error_redirect(error_msg)

        for i in range(len(target_sel)):
            lower = target_lower[i]
            upper = target_upper[i]
            if target_sel[i]=='' or (lower == '' and upper == ''):
                continue
            else:
                if check_number(lower, upper) == False:
                    error_msg = 'Error: [TARGET] 探索条件には数値を入力してください'
                    self.error_redirect(error_msg)
                if lower != "" and upper == "":
                    lower_f = float(lower)
                    upper_f = round(self.range_mm[target_sel[i]]["mx"], 2)
                elif lower == "" and upper != "":
                    lower_f = round(self.range_mm[target_sel[i]]["mn"], 2)
                    upper_f = float(upper)
                elif lower != "" and upper != "":
                    lower_f = float(lower)
                    upper_f = float(upper)
                elif lower == "" and upper == "":
                    lower_f = round(self.range_mm[target_sel[i]]["mn"], 2)
                    upper_f = round(self.range_mm[target_sel[i]]["mx"], 2)

                if lower_f > upper_f:
                    error_msg = 'Error : [TARGET] 上限値より下限値に大きい値が入力されました'
                    self.error_redirect(error_msg)
                target_cols.append(target_sel[i])
                target_pairs.append((lower_f, upper_f))
        
        # TARGETの確認用
        print("TARGET")
        disp_list = []
        for i, col in enumerate(target_cols):
            disp = "{0} < {1} < {2}".format(target_pairs[i][0], col, target_pairs[i][1])
            disp_list.append(disp)
        if not disp_list:
            disp_list.append("なし")
        target_message = "\n, ".join(disp_list)

        return target_cols, target_pairs, target_message

    def get_step_dict(self, step_sel, step_val, default_step_value=0.01):
        step_dict = {}
        for col in self.df_columns:
            step_dict[col] = default_step_value
        if step_sel != [""]:
            for i, sel in enumerate(step_sel):
                if check_number(step_val[i]) == False:
                    error_msg = 'Error: [STEP] 刻み値には数値を入力してください'
                    self.error_redirect(error_msg)
                step_dict[sel] = float(step_val[i])
        return step_dict