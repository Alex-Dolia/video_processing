import boto3
import os 

def download_directory_from_s3(bucket_name, remote_dir_name, save_path):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name) 
    for obj in bucket.objects.filter(Prefix = remote_dir_name):
        new_path = obj.key.replace(remote_dir_name, save_path)
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))
        bucket.download_file(obj.key, new_path)