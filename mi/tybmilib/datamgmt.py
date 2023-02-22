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
from botocore.exceptions import ClientError
from tybmilib.logmgmt import logger, stop_watch

#------------------------------------------------------------
class S3Dao:
    def __init__(self):
        self.role = sagemaker.get_execution_role()
        self.region = boto3.Session().region_name
        self.session = sagemaker.Session()
        self.bucket = sagemaker.Session().default_bucket()

    def upload_data(self,file_path_list:'list',s3_bucket_path=None):
        # logging
        logger().info('Process Start  : {}'.format('upload_data'))
        logger().debug('In  : {}'.format([file_path_list,s3_bucket_path]))
        
        # S3指定
        s3_uri_list = []
        for j in range(len(file_path_list)):
            file = file_path_list[j]
            data_directory = file.split('/')
            file_name = data_directory[len(data_directory)-1]
            #S3関連
            if s3_bucket_path == None:
                prefix = 'sagemaker/for_all/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                prefix = s3_bucket_path
            s3 = boto3.resource('s3')
            try:
                s3.Bucket(self.bucket).upload_file(file,prefix + '/' + file_name)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger().error('Error  : {}'.format('The object does not exist.'))
                else:
                    raise
                
            uri = 's3://' + self.bucket + '/' + prefix + '/' + file_name
            s3_uri_list.append(uri)
        print('=========s3指定バケットへの格納データ=========')
        for i in range(len(s3_uri_list)):
            print(s3_uri_list[i])
        
        # logging
        logger().debug('Out  : {}'.format(s3_uri_list))
        logger().info('Process End  : {}'.format('upload_data'))
        
        return s3_uri_list
    
    def download_data(self,s3_path_list:'list'):
        # logging
        logger().info('Process Start  : {}'.format('download_data'))
        logger().debug('In  : {}'.format(s3_path_list))
        
        # ローカルフォルダ指定
        new_path = 'data' #フォルダ名
        if not os.path.exists(new_path):#ディレクトリがなかったら
            os.mkdir(new_path)#作成したいフォルダ名を作成
        path = os.getcwd()
        os.chdir(path + '/' + new_path)
        rocal_path_list = []
        for j in range(len(s3_path_list)):
            s3_file = s3_path_list[j]
            data_directory = s3_file.split('/')
            file_name = data_directory[len(data_directory)-1]
            in_bucket_path = "/".join(data_directory[3:])
            s3 = boto3.resource('s3')
            try:
                s3.Bucket(data_directory[2]).download_file(in_bucket_path,file_name)
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger().error('Error  : {}'.format('The object does not exist.'))
                else:
                    raise
            rocal_path = path + '/' + new_path + '/' + file_name
            rocal_path_list.append(rocal_path)
        os.chdir(path)
        '''
        print('=========dataフォルダへの格納データ=========')
        for i in range(len(rocal_path_list)):
            print(rocal_path_list[i])
        '''
        # logging
        logger().debug('Out  : {}'.format(rocal_path_list))
        logger().info('Process End  : {}'.format('download_data'))
        
        return rocal_path_list
    
class MetadataDAO:
    def __init__(self,connection,user,user_pass,dbtype='mongodb'):
        self.connection = connection
        self.user = user
        self.user_pass = user_pass
        self.dbtype = dbtype
        
    def create_newMI(self):
        None
        
class _MongodbmetadataDao:
    def __init__(self,connection,user,user_pass):
        self.connection = connection
        self.user = user
        self.user_pass = user_pass
        
    def create_newMI(self):
        None

#------------------------------------------------------------
if __name__ == '__main__':
    None