import os
import shutil
from datetime import datetime
import configparser
import sagemaker
config_ini = configparser.ConfigParser()
config_ini_path = os.path.join(os.path.dirname(__file__), 'cfg.ini')

def get_localmode():
    #Local_mode = False
    if os.path.exists(config_ini_path):
        # iniファイルが存在する場合、ファイルを読み込む
        with open(config_ini_path, encoding='utf-8') as fp:
            config_ini.read_file(fp)
            # iniの値取得
            read_default = config_ini['PATH']
            Local_mode = read_default.getboolean('Local_mode')

            return Local_mode
    else:
        print("no path")

def get_aws_role():
    Local_mode = get_localmode()
    if Local_mode==True:
        if os.path.exists(config_ini_path):
            # iniファイルが存在する場合、ファイルを読み込む
            with open(config_ini_path, encoding='utf-8') as fp:
                config_ini.read_file(fp)
                # iniの値取得
                read_default = config_ini['PATH']
                aws_role = read_default.get('role')
                return aws_role
    else:
        return sagemaker.get_execution_role()

def clear_folder(folder_name):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def prepare_folder(exp_id="xxx"):
    Local_mode = get_localmode()

    if Local_mode == False:
        data_path = get_nb_data_path()
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        output_path = get_nb_output_path()
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        log_path = get_nb_log_path()
        if not os.path.exists(log_path):
            os.mkdir(log_path)
    elif Local_mode == True:
        expid_path = get_expid_path(exp_id, relative=False)
        clear_folder(expid_path)
        csv_path = get_csv_path(exp_id)
        clear_folder(csv_path)
        html_path = get_html_path(exp_id, relative=False)
        clear_folder(html_path)
        img_path = get_img_path(exp_id, relative=False)
        clear_folder(img_path)
        log_path = get_log_path(exp_id)
        clear_folder(log_path)

    each_folder = get_each_structure_path(exp_id, Local_mode=Local_mode)
    list_folder = get_list_structure_path(exp_id, Local_mode=Local_mode)
    bit_folder = get_bit_structure_path(exp_id, Local_mode=Local_mode)
    clear_folder(each_folder)
    clear_folder(list_folder)
    clear_folder(bit_folder)

def get_nb_data_path():
    data_path = os.path.join(os.getcwd(), "data")
    return data_path

def get_nb_output_path():
    output_path = os.path.join(os.getcwd(), "output")
    return output_path

def get_nb_log_path():
    log_path = os.path.join(os.getcwd(), "log")
    return log_path

def get_csv_data_path(exp_id, Local_mode=True):
    if Local_mode==True:
        csv_path = get_csv_path(exp_id)
    elif Local_mode==False:
        csv_path = get_nb_data_path()
    return csv_path

# img / output folder 上のファイル名定義
def define_img_output_path(exp_id, path_name, Local_mode=True, relative=False):
    if Local_mode == True:
        def_img_path = os.path.join(get_img_path(exp_id, relative), path_name)
    elif Local_mode == False:
        def_img_path = os.path.join(get_nb_output_path(), path_name)
    return def_img_path

# html / output folder 上のファイル名定義
def define_html_output_path(exp_id, path_name, Local_mode=True, relative=False):
    if Local_mode == True:
        def_html_path = os.path.join(get_html_path(exp_id, relative), path_name)
    elif Local_mode == False:
        def_html_path = os.path.join(get_nb_output_path(), path_name)
    return def_html_path

# csv / data folder 上のファイル名定義
def define_csv_data_path(exp_id, path_name, Local_mode=True):
    def_csv_path = os.path.join(get_csv_data_path(exp_id, Local_mode=Local_mode), path_name)
    return def_csv_path

# csv / output folder 上のファイル名定義
def define_csv_output_path(exp_id, path_name, Local_mode=True):
    if Local_mode == True:
        def_csv_path = os.path.join(get_csv_path(exp_id), path_name)
    elif Local_mode == False:
        def_csv_path = os.path.join(get_nb_output_path(), path_name)
    return def_csv_path

def define_log_path(exp_id, path_name, Local_mode=True):
    if Local_mode == True:
        def_log_path = os.path.join(get_log_path(exp_id), path_name)
    elif Local_mode == False:
        def_log_path = os.path.join(get_nb_log_path(), path_name)
    return def_log_path

def get_static_path():
    if os.path.exists(config_ini_path):
        # iniファイルが存在する場合、ファイルを読み込む
        with open(config_ini_path, encoding='utf-8') as fp:
            config_ini.read_file(fp)
            # iniの値取得
            read_default = config_ini['PATH']
            cd_path = read_default.get('cd_path')

            static_path = os.path.join(cd_path, "static")
            return static_path

    else:
        print("no path")
    

def get_zip_path():
    zip_path = os.path.join(get_static_path(), "zip")
    if not os.path.exists(zip_path):
        os.mkdir(zip_path)
    return zip_path

def get_default_csv_path():
    default_csv_path = os.path.join(get_static_path(), "csv")
    if not os.path.exists(default_csv_path):
        os.mkdir(default_csv_path)
    return default_csv_path

def get_approval_filename():
    approval_name = os.path.join(get_default_csv_path(), "approval.csv")
    return approval_name

def get_department_filename():
    department_name = os.path.join(get_default_csv_path(), "department.csv")
    return department_name

def get_users_filename():
    users_name = os.path.join(get_default_csv_path(), "users.csv")
    return users_name

def get_usage_filename():
    usage_name = os.path.join(get_default_csv_path(), "usage.csv")
    return usage_name

def get_expid_path(exp_id, relative=False):
    if relative:
        expid_path = os.path.join("../static", exp_id)
    else:
        expid_path = os.path.join(get_static_path(), exp_id)
    return expid_path

def get_csv_path(exp_id):
    csv_path = os.path.join(get_expid_path(exp_id), "csv")
    return csv_path

def get_html_path(exp_id, relative=False):
    html_path = os.path.join(get_expid_path(exp_id, relative), "html")
    return html_path

def get_img_path(exp_id, relative=False):
    img_path = os.path.join(get_expid_path(exp_id, relative), "img")
    return img_path

def get_log_path(exp_id):
    log_path = os.path.join(get_expid_path(exp_id), "log")
    return log_path

def get_defaultcsv_filename():
    default_csv_name = os.path.join(get_default_csv_path(), "Preview_base.csv")
    return default_csv_name

def get_preview_filename(exp_id):
    df_s3_file_name = os.path.join(get_csv_path(exp_id), "df_s3_preview.csv")
    return df_s3_file_name

def get_master_filename(exp_id):
    master_name = os.path.join(get_csv_path(exp_id), "df_master.csv")
    return master_name

def get_scatter_filename(exp_id, mode, Local_mode=True, relative=False):
    if mode=="objectives":
        scatter_filename = get_scatter_objectives_filename(exp_id, Local_mode, relative)
    elif mode=="all":
        scatter_filename = get_scatter_all_filename(exp_id, Local_mode, relative)
    else:
        scatter_filename = ""
    return scatter_filename

def get_scatter_objectives_filename(exp_id, Local_mode=True, relative=False):
    path_name = "scatter_only_objectives.png" 
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_scatter_all_filename(exp_id, Local_mode=True, relative=False):
    path_name = "scatter_all.png"
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_correlation_filename(exp_id, mode, Local_mode=True, relative=False):
    if mode=="objectives":
        correlation_filename = get_correlation_objectives_filename(exp_id, Local_mode, relative)
    elif mode=="all":
        correlation_filename = get_correlation_all_filename(exp_id, Local_mode, relative)
    else:
        correlation_filename = ""
    return correlation_filename

def get_correlation_objectives_filename(exp_id, Local_mode=True, relative=False):
    path_name = "correlation_matrix_only_objectives.png"
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_correlation_all_filename(exp_id, Local_mode=True, relative=False):
    path_name = "correlation_matrix_all.png"
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_profile_filename(exp_id, Local_mode=True, relative=False):
    path_name = "profile.html"
    return define_html_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_coefficients_filename(exp_id, mode, target_name, Local_mode=True, relative=False):
    if mode=="all":
        coefficients_filename = get_coefficients_all_filename(exp_id, target_name, Local_mode, relative)
    elif mode=="limit":
        coefficients_filename = get_coefficients_limit_filename(exp_id, target_name, Local_mode, relative)
    else:
        coefficients_filename = ""
    return coefficients_filename

def get_coefficients_all_filename(exp_id, target_name, Local_mode=True, relative=False):
    path_name = "Coefficients(All)_{0}.png".format(target_name)
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_coefficients_limit_filename(exp_id, target_name, Local_mode=True, relative=False):
    path_name = "Coefficients(Limit)_{0}.png".format(target_name)
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_test_filename(exp_id, target_name, Local_mode=True, relative=False):
    path_name = "test_{0}.png".format(target_name)
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_confusion_filename(exp_id, target_name, Local_mode=True, relative=False):
    path_name = "Confusion_matrix_{0}.png".format(target_name)
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)


def get_shap_filename(exp_id, Local_mode=True, relative=False):
    path_name = "shap_report.html"
    return define_html_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_trainx_filename(exp_id, Local_mode=True):
    path_name = "train(only_x).csv"
    return define_csv_data_path(exp_id, path_name, Local_mode=Local_mode)

def get_trainy_filename(exp_id, Local_mode=True):
    path_name = "train(only_y).csv"
    return define_csv_data_path(exp_id, path_name, Local_mode=Local_mode)

def get_samplingx_filename(exp_id, Local_mode=True):
    path_name = "sampling(only_x).csv"
    return define_csv_data_path(exp_id, path_name, Local_mode=Local_mode)

def get_simulate_filename(exp_id, Local_mode=True):
    path_name = 'Simulate.csv'
    return define_csv_output_path(exp_id, path_name, Local_mode=Local_mode)

def get_cluster_filename(exp_id, Local_mode=True):
    path_name = 'Cluster.csv'
    return define_csv_output_path(exp_id, path_name, Local_mode=Local_mode)

def get_cluster_mean_filename(exp_id, Local_mode=True):
    path_name = 'Cluster_mean.csv'
    return define_csv_output_path(exp_id, path_name, Local_mode=Local_mode)

def get_pareto_filename(exp_id, Local_mode=True):
    path_name = 'Pareto.csv'
    return define_csv_output_path(exp_id, path_name, Local_mode=Local_mode)

def get_cluster_img_filename(exp_id, Local_mode=True, relative=False):
    path_name = "samples_Clustering.png"
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_pareto_img_filename(exp_id, target1, target2, target3="", Local_mode=True, relative=False):
    if target3 != "":
        path_name = "Pareto_optimal_samples_{0}-{1}-{2}.png".format(target1, target2, target3)
    else:
        path_name = "Pareto_optimal_samples_{0}-{1}.png".format(target1, target2)

    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_trainob_filename(exp_id, target_name, Local_mode=True):
    path_name = "train_{0}.csv".format(target_name)
    return define_csv_data_path(exp_id, path_name, Local_mode=Local_mode)


def get_sample_filename(exp_id, Local_mode=True):
    path_name = "Samples.csv"
    return define_csv_data_path(exp_id, path_name, Local_mode=Local_mode)



def get_shapresult_target_filename(exp_id, target, Local_mode=True):
    path_name = "shap_analysis_{}.json".format(target)
    return define_html_output_path(exp_id, path_name, Local_mode=Local_mode)

def get_s3_report_filename_prefix(prefix):
    s3_report = "report.html"
    s3_report_filename_prefix = os.path.join(prefix, s3_report)
    return s3_report_filename_prefix

def get_s3_analysis_filename_prefix(prefix):
    s3_analysis = "analysis.json"
    s3_analysis_filename_prefix = os.path.join(prefix, s3_analysis)
    return s3_analysis_filename_prefix

def get_shaptg_filename(exp_id, target, Local_mode=True, relative=False):
    path_name = "report_{0}.html".format(target)
    return define_html_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_each_structure_path(exp_id, Local_mode=True, relative=False):
    path_name = "each_structure"
    each_path = define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)
    return each_path

def get_list_structure_path(exp_id, Local_mode=True, relative=False):
    path_name = "list_structure"
    list_path = define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)
    return list_path

def get_bit_structure_path(exp_id, Local_mode=True, relative=False):
    path_name = "bit_structure"
    bit_path = define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)
    return bit_path

def get_maccs_each_filename(exp_id, k_num, idx, Local_mode=True, relative=False):
    path_name = "maccs_{0:04d}_{1:04d}.png".format(k_num, idx)
    maccs_each_filename = os.path.join(get_each_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return maccs_each_filename

def get_maccs_list_filename(exp_id, k_num, Local_mode=True, relative=False):
    path_name = "MACCS_{0:04d}.png".format(k_num)
    maccs_list_filename = os.path.join(get_list_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return maccs_list_filename

def get_maccs_bit_filename(exp_id, k_num, Local_mode=True, relative=False):
    path_name = "MACCS_{0:04d}.png".format(k_num)
    maccs_list_filename = os.path.join(get_bit_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return maccs_list_filename

def get_source_filename(exp_id, Local_mode=True):
    path_name = "bitID_SourceName.csv"
    return define_csv_output_path(exp_id, path_name, Local_mode=Local_mode)

def get_mfp_each_filename(exp_id, struc, bit_n, Local_mode=True, relative=False):
    path_name = "{0}_{1:04d}.png".format(struc, bit_n)
    mfp_each_filename = os.path.join(get_each_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return mfp_each_filename

def get_mfp_structure_list_filename(exp_id, struc, Local_mode=True, relative=False):
    path_name = '{}.png'.format(struc)
    mfp_structure_list_filename = os.path.join(get_list_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return mfp_structure_list_filename

def get_mfp_bit_filename(exp_id, bit_n, Local_mode=True, relative=False):
    path_name = 'bit_{0:04d}.png'.format(bit_n)
    mfp_bit_list_filename = os.path.join(get_bit_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return mfp_bit_list_filename

def get_smiles_filename(exp_id, Local_mode=True, relative=False):
    path_name = "smiles.png"
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_shaptop_filename(exp_id, topnum, Local_mode=True, relative=False):
    path_name = "shap_result_{0:02}.png".format(topnum)
    shaptop_filename = os.path.join(get_bit_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return shaptop_filename

def get_shaptop_target_filename(exp_id, rank, target, Local_mode=True, relative=False):
    path_name = "shap_result_{0}_{1:02}.png".format(target, rank)
    shaptop_filename = os.path.join(get_bit_structure_path(exp_id, Local_mode=Local_mode, relative=relative), path_name)
    return shaptop_filename

def get_shaprank_filename(exp_id, Local_mode=True, relative=False):
    path_name = "shap_rank.png"
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_shaprank_target_filename(exp_id, target, Local_mode=True, relative=False):
    path_name = "shap_rank_{}.png".format(target)
    return define_img_output_path(exp_id, path_name, Local_mode=Local_mode, relative=relative)

def get_step1_visualize_log_filename(exp_id, Local_mode=True):
    current_time = datetime.now()
    today_date, now_time = str(current_time).split(" ")
    path_name = "{}_step1_visualize.log".format(today_date)
    return define_log_path(exp_id, path_name, Local_mode=Local_mode)

def get_step2_modelbuilding_log_filename(exp_id, Local_mode=True):
    current_time = datetime.now()
    today_date, now_time = str(current_time).split(" ")
    path_name = "{}_step2_modelbuilding.log".format(today_date)
    return define_log_path(exp_id, path_name, Local_mode=Local_mode)

def get_step3_createsample_log_filename(exp_id, Local_mode=True):
    current_time = datetime.now()
    today_date, now_time = str(current_time).split(" ")
    path_name = "{}_step3_sampling.log".format(today_date)
    return define_log_path(exp_id, path_name, Local_mode=Local_mode)

def get_step3_paramsearch_log_filename(exp_id, Local_mode=True):
    current_time = datetime.now()
    today_date, now_time = str(current_time).split(" ")
    path_name = "{}_step3_paramsearch.log".format(today_date)
    return define_log_path(exp_id, path_name, Local_mode=Local_mode)

def get_step_all_log_filename(exp_id, Local_mode=True):
    current_time = datetime.now()
    today_date, now_time = str(current_time).split(" ")
    path_name = "{}_step_all.log".format(today_date)
    return define_log_path(exp_id, path_name, Local_mode=Local_mode)

def get_user_s3_bucket(user_dept):
    user_s3_bucket = "mi-" + str.lower(user_dept)
    return user_s3_bucket

def get_modeling_s3_bucket(region_name="", Local_mode=True):
    if Local_mode:
        return "mi-modeling"
    else:
        return "mi-modeling-" + region_name

def get_user_s3_prefix(user_id, exp_id):
    user_s3_prefix = user_id + "/" + exp_id + "/"
    return user_s3_prefix

def get_user_s3_output(user_id, exp_id):
    user_s3_output = get_user_s3_prefix(user_id, exp_id) + "output"
    return user_s3_output

def get_s3_csv_path(user_id, exp_id):
    csv_path = os.path.join(get_user_s3_output(user_id, exp_id), "csv")
    return csv_path

def get_s3_html_path(user_id, exp_id):
    html_path = os.path.join(get_user_s3_output(user_id, exp_id), "html")
    return html_path

def get_s3_img_path(user_id, exp_id):
    img_path = os.path.join(get_user_s3_output(user_id, exp_id), "img")
    return img_path

def get_s3_modeling_path(user_id, exp_id):
    modeling_path = os.path.join(get_user_s3_output(user_id, exp_id), "modeling")
    return modeling_path