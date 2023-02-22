import os
from datetime import timedelta
import boto3
from tybmilib import prep
from tybmilib import myfilename as mfn



def get_s3_file_list(user_dept, user_id, exp_id):
    """S3上のファイルリストを出力
    
    Args:
        user_dept (str): 1st argument
        user_id (str): 2nd argument
        exp_id (str): 3rd argument
    Returns:
        list: input_s3_file_list
        list: time_array
    
    """
    input_s3_file_list = []
    time_array = []
    s3_client = boto3.client('s3')

    user_s3_bucket = mfn.get_user_s3_bucket(user_dept)
    user_s3_prefix = mfn.get_user_s3_prefix(user_id, exp_id)

    input_obj = s3_client.list_objects_v2(Bucket=user_s3_bucket, Prefix=user_s3_prefix, Delimiter='/')
    if not "Contents" in input_obj:
        s3_client.put_object(Bucket=user_s3_bucket, Key=user_s3_prefix)
    else:
        input_s3_list = [content['Key'] for content in input_obj['Contents']]
        for s3_file in input_s3_list[1:]:
            input_s3_file_list.append(os.path.basename(s3_file))
        if len(input_s3_file_list) > 0:
            for i, content in enumerate(input_obj["Contents"]):
                if i != 0:
                    time_array.append(content["LastModified"].replace(tzinfo = None) + timedelta(hours=9))
            input_info = zip(time_array, input_s3_file_list)
            input_info_sorted = sorted(input_info, reverse=True)
            time_array, input_s3_file_list = zip(*input_info_sorted)
    return input_s3_file_list, time_array


def create_user_bucket(bucket_name):
    """S3バケットを作成
    
    Args:
        bucket_name (str): 1st argument
    Returns:
        None
    
    """
    region = boto3.Session().region_name
    s3_client = boto3.client('s3', region_name=region)
    location = {'LocationConstraint': region}
    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)            
    except Exception as e:
        print(str(e))


def delete_all_s3_objects(bucket_name):
    """S3バケット上のオブジェクトを全て削除
    
    Args:
        bucket_name (str): 1st argument
    Returns:
        None
    
    """
    s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(bucket_name)
    s3_bucket.objects.all().delete()


def delete_user_bucket(bucket_name):
    """S3バケットを削除
    
    Args:
        bucket_name (str): 1st argument
    Returns:
        None
    
    """
    region = boto3.Session().region_name
    s3_client = boto3.client('s3', region_name=region)
    try:
        delete_all_s3_objects(bucket_name)
        s3_client.delete_bucket(Bucket=bucket_name)
    except Exception as e:
        print(str(e))


def upload_file(bucket_name, local_filename, s3_prefix):
    """設定したS3パスにアップロード
    
    Args:
        bucket_name (str): 1st argument
        local_filename (str): 2nd argument
        s3_prefix (str): 3rd argument
    Returns:
        str: s3uri
    
    """
    if os.path.exists(local_filename):
        filename = os.path.basename(local_filename)
        s3_filename = os.path.join(s3_prefix, filename)

        s3_client = boto3.resource('s3')
        s3_client.Bucket(bucket_name).upload_file(local_filename, s3_filename)

        return "s3://{0}/{1}".format(bucket_name, s3_filename)


def upload_file_list(bucket_name, local_file_list, s3_prefix):
    """設定したS3パスに複数のファイルをアップロード
    
    Args:
        bucket_name (str): 1st argument
        local_file_list (list): 2nd argument
        s3_prefix (str): 3rd argument
    Returns:
        str: s3uri
    
    """
    s3_uri_list = []
    for local_filename in local_file_list:
        s3_uri = upload_file(bucket_name, local_filename, s3_prefix)
        s3_uri_list.append(s3_uri)
    return s3_uri_list


def download_file(bucket_name, s3_filename, local_folder):
    """設定したS3パスからダウンロード
    
    Args:
        bucket_name (str): 1st argument
        s3_filename (str): 2nd argument
        local_folder (str): 3rd argument
    Returns:
        str: local_filename
    
    """
    if not os.path.exists(local_folder):
        os.mkdir(local_folder)
    filename = os.path.basename(s3_filename)
    local_filename = os.path.join(local_folder, filename)

    s3_client = boto3.resource('s3') #S3オブジェクトを取得
    s3_client.Bucket(bucket_name).download_file(s3_filename, local_filename)

    return local_filename


def copy_s3folders_with_change_dept(old_dept, new_dept, username):
    """あるS3バケット上のファイル群を、全て別のS3バケット上へと移動（コピーして削除） 
    
    Args:
        old_dept (str): 1st argument
        new_dept (str): 2nd argument
        username (str): 3rd argument
    Returns:
        None
    
    """
    old_bucket = mfn.get_user_s3_bucket(old_dept)
    new_bucket = mfn.get_user_s3_bucket(new_dept)
    s3_dir = username + "/"
    prep.copy_s3_items(old_bucket, s3_dir, new_bucket, s3_dir)
    prep.delete_s3_items(old_bucket, s3_dir)


def split_s3_path(s3_uri):
    """S3パスからバケットとプレフィックスを取得
    
    Args:
        s3_uri (str): 1st argument
    Returns:
        str: bucket
        str: key

    """
    path_parts=s3_uri.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key