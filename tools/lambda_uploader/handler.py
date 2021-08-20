import boto3
import botocore.exceptions
import json
import os
import httpx

def handler(event, context):
    BUCKET = os.getenv('BUCKET')
    if 'url' in event and 'city' in event:
        url = event['url']
        city = event['city']
        file_name = url.split('/')[-1]
        client = boto3.client('s3')
        chunks=bytearray()
        try:
            with httpx.stream("GET", url) as response:
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    
        except httpx.ReadTimeout as err:
            return {
                'statusCode': 500,
                'body': json.dumps(err)
            }

        try:
            client.put_object(Body=chunks, Bucket=BUCKET, Key=f'{city }/{file_name}')
        except botocore.exceptions.ClientError as cerr:
            return {
                'statusCode': 500,
                'body': json.dumps(cerr)
            }
    else:
        if not 'url' in event:
            return{
            'statusCode':400,
            'body': json.dumps('No url paramter provided')
            }
        if not 'city' in event:
            return{
            'statusCode':400,
            'body': json.dumps('No city paramter provided')
            }
