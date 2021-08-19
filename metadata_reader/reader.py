import boto3
import botocore.exceptions
import json
import os
import time
import math

from tqdm import tqdm

#CITIES=['01_rio', '02_vegas', '03_paris', '04_shanghai', '05_khartoum', '06_atlanta', '07_moscow', '08_mumbai', '09_san', '10_dar', 
CITIES=['11_rotterdam']
url_field="url_m"

def get_aws_access_key_id():
    try:
        with open('./aws-secrets.txt', 'r') as secret_file:
            lines = secret_file.readlines()
            aki = lines[1].strip().split('=')[1]
            return aki
    except IOError as err:
        print(f'{err}')
        return None
def get_aws_secret_access_key():
    try:
        with open('./aws-secrets.txt', 'r') as secret_file:
            lines = secret_file.readlines()
            sak = lines[2].strip().split('=')[1]
            return sak
    except IOError:
        return None

def get_aws_session_token():
    try:
        with open('./aws-secrets.txt', 'r') as secret_file:
            lines = secret_file.readlines()
            st = lines[3].strip().split('=')[1]
            return st
    except IOError:
        return None

def read_metadata():
    metadata = {}
    urls = {}
    # for key in cfg['cities']:
        # city=key.replace(" ", "_")
    for city in CITIES:
        urls[city]=set()
        file_path=f'./data/{city}/metadata.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                loaded = json.load(f)
                for img in loaded['images']:
                    if url_field in img and not img[url_field] in urls:
                        urls[city].add(img[url_field])
                metadata[city]= loaded

    return metadata, urls

def send_to_lambda(urls):
    aki = get_aws_access_key_id()
    sak = get_aws_secret_access_key()
    st = get_aws_session_token()

    failed_paths=set()
    lambda_client = boto3.client('lambda', region_name='us-east-2', aws_access_key_id=aki, aws_secret_access_key=sak, aws_session_token=st)
    for city in CITIES:
        for url in tqdm(urls[city], desc=city):
            try:
                file_name = url.split('/')[-1]
                lambda_payload = json.dumps({"city":city,"url":url}).encode()
                lambda_client.invoke(FunctionName='witw_uploader', 
                     InvocationType='Event',
                     Payload=lambda_payload)
            except botocore.exceptions.ClientError as cerr:
                    print(f'error uploading file {city}/{file_name}')
                    print(f'{cerr}')
                    failed_paths.add(f'{city}/{url}')

    if(len(failed_paths) > 0):
        file_path=os.path.join('./', 'failed_urls.txt')
        print(f"Some uploads failed. writing {len(failed_paths)} urls to {file_path}")
        try:
            with open(file_path, 'w') as f:
                for url in failed_paths:
                    f.write(f'{url}\n')

        except Exception as err:
            print(f"error: {err} writing to file {file_path}")

def main():
    metadata,urls = read_metadata()
    send_to_lambda(urls)

if __name__ == '__main__':  # pragma: no cover
    main()
    