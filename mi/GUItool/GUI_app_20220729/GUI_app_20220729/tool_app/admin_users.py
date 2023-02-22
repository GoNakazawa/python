from tool_app import app
import pandas as pd
import boto3
from datetime import datetime, timedelta
from flask_mail import Mail, Message
import string
import secrets
import re
from tool_app.mymongo import Experiments, Users
from tybmilib import myfilename as mfn


#メール関連の設定
SENDER_MAIL_ADDRESS = 'matsubara0225@gmail.com'
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = SENDER_MAIL_ADDRESS
app.config['MAIL_PASSWORD'] = 'aznazgahftlucjss'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

# メール送信
def send_mail(mail_title, mail_body, mail_recipients):
    msg = Message(mail_title, sender = SENDER_MAIL_ADDRESS, recipients = mail_recipients)
    msg.body = mail_body
    mail.send(msg)
    return True

# 部署名一覧の取得
def get_dept_list(new_dept=""):
    dept_df = pd.read_csv(mfn.get_department_filename())
    dept_list = list(dept_df["department"])
    if new_dept != "":
        dept_list.append(new_dept)
    dept_list.sort()
    dept_list = [str.upper(dept) for dept in dept_list]
    return dept_list

# パスワードチェック（大文字小文字数字が1つずつ）
def check_passwd(passwd):
    if len(passwd)>=10 and len(re.findall("[a-z]", passwd))>0 and len(re.findall("[A-Z]", passwd))>0 and len(re.findall("[0-9]", passwd))>0:
        return True
    else:
        return False

# パスワード生成
def get_random_password_string(length):
    pass_chars = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(pass_chars) for x in range(length))
    return password

# 新規ユーザー登録用
def set_new_user_dict(username, dept, admin):
    add_dict = {}
    add_dict["userId"] = username
    add_dict["department"] = dept
    add_dict["admin"] = admin
    add_dict["passWd"] = get_random_password_string(10)
    add_dict["passWd_change"] = 0
    add_dict["page_count"] = 0
    add_dict["vis_count"] = 0
    add_dict["model_count"] = 0
    add_dict["infer_count"] = 0
    add_dict["sagemaker_time"] = 0
    return add_dict

# ユーザーの利用状況取得
def write_usage_info():
    usr = Users()
    usr_list = usr.get_all_users()
    usage_df = pd.DataFrame(columns=["userId", "department", "page", "vis", "model", "infer"])
    usage_df["userId"] = usr_list

    for i, user_id in enumerate(usr_list):
        user = usr.find_one_userid(user_id)
        usage_df["department"][i] = user["department"]
        usage_df["page"][i] = user["page_count"]
        usage_df["vis"][i] = user["vis_count"]
        usage_df["model"][i] = user["model_count"]
        usage_df["infer"][i] = user["infer_count"]

    usage_df.to_csv(mfn.get_usage_filename(), index=False)

# 実行月の1日を取得
def get_begin_of_month():
    #return date.today().replace(day=1).isoformat()
    return "2022-05-01"

# 実行日を取得
def get_today():
    #return date.today().replace(day=16).isoformat()
    return "2022-06-01"

def get_total_cost_date_range():
    start_date = get_begin_of_month()
    end_date = get_today()

    # get_cost_and_usage()のstartとendに同じ日付は指定不可のため、
    # 「今日が1日」なら、「先月1日から今月1日（今日）」までの範囲にする
    if start_date == end_date:
        end_of_month = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=-1)
        begin_of_month = end_of_month.replace(day=1)
        return begin_of_month.date().isoformat(), end_date
    return start_date, end_date

# 合計請求額取得を取得
def get_total_billing():
    client = boto3.client('ce')
    start_date, end_date = get_total_cost_date_range()
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost']
    )

    return {
        'start': response['ResultsByTime'][0]['TimePeriod']['Start'],
        'end': response['ResultsByTime'][0]['TimePeriod']['End'],
        'billing': response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'],
    }

# 各サービスの詳細請求金額を取得
def get_service_billings():
    client = boto3.client('ce')
    start_date, end_date = get_total_cost_date_range()

    #CostExplorer.Client.get_cost_and_usage
    response = client.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )

    billings = []
    for item in response['ResultsByTime'][0]['Groups']:
        billings.append({
            'service_name': item['Keys'][0],
            'billing': item['Metrics']['UnblendedCost']['Amount']
        })
    return billings