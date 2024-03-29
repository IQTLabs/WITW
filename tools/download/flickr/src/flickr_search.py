import boto3
import botocore.exceptions
import json
import os
import httpx
import time
import math

from flickrapi import FlickrAPI
from flickrapi.exceptions import FlickrError
from pprint import pprint
from tqdm import tqdm

from config import parse_config
# from metadata import *
# from secrets import *

PRIVACY_FILTER = 1 #only public metadata
CONTENT_TYPE = 1 #only metadata
HAS_GEO = True  #only geotagged metadata
GEO_CTX = 0     #0=all, 1=indoor, 2=outdoor
BUCKET = 'witw-quick'
SECRETS_DIR = '/run/secrets'
DATA_DIR = '/data'

def get_aws_access_key_id():
    try:
        with open(f'{SECRETS_DIR}/aws_secrets', 'r') as secret_file:
            lines = secret_file.readlines()
            aki = lines[1].strip().split('=')[1]
            return aki
    except IOError as err:
        print(f'{err}')
        return None
def get_aws_secret_access_key():
    try:
        with open(f'{SECRETS_DIR}/aws_secrets', 'r') as secret_file:
            lines = secret_file.readlines()
            sak = lines[2].strip().split('=')[1]
            return sak
    except IOError:
        return None

def get_aws_session_token():
    try:
        with open(f'{SECRETS_DIR}/aws_secrets', 'r') as secret_file:
            lines = secret_file.readlines()
            st = lines[3].strip().split('=')[1]
            return st
    except IOError:
        return None

def get_secret(secret_name):
    try:
        with open('{0}/{1}'.format(SECRETS_DIR, secret_name), 'r') as secret_file:
            return secret_file.read()
    except IOError:
        return None

def est_area(bbox, radius=6.371E6):
    # Estimated area of a small bounding box, in m^2
    return (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) \
        * math.cos((bbox[1]+bbox[3])/2 * math.pi / 180.) \
        * (math.pi / 180.) **2 * radius**2

def get_usable_bounding_boxes(nominal_boxes, cfg):
    FLICKR_PUBLIC = get_secret('flickr_api_key')
    FLICKR_SECRET = get_secret('flickr_api_secret')
    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
    boxes = []
    working = nominal_boxes.copy()

    license = "1,2,3,4,5,6,7,8,9,10"
    extras ='description,license,date_upload,date_taken,original_format,'
    extras+='last_update,geo,tags, machine_tags, o_dims, media,'
    extras+='url_m,url_n,url_z,url_c,url_l,url_o'

    city_total=0

    # print('  area_km2 count type   bounding_box')
    while len(working) > 0:
        box = working.pop()
        temp = list(map(str, box))
        str_box = ",".join(temp)
        box_area = est_area(box)
        divide_flag = False
        if box_area > cfg["max_area"]:
            total_imgs = -1
            divide_flag = True
        else:
            time.sleep(cfg["time_delay"])
            try:
                box_pics = flickr.photos.search(
                    privacy_filter=PRIVACY_FILTER, bbox=str_box,
                    content_type=CONTENT_TYPE,
                    has_geo=HAS_GEO, geo_context=GEO_CTX,
                    license=license, extras=extras, per_page=cfg["page_size"])
                total_imgs = int(box_pics['photos']['total'])
                divide_flag = (total_imgs >= cfg["density_limit"] and box_area > cfg["min_area"])
            except FlickrError as err:
                print(f'Error retrieving intitial page for bounding box {bbox}')
                print(f'{err}')
        # print('%10.4f %5i %s %s' % (box_area/1.E6, total_imgs, 'branch'
        #                             if divide_flag else 'leaf  ', box))
        if divide_flag:
            new_box_1 = box.copy()
            new_box_2 = box.copy()
            if box[2] - box[0] > box[3] - box[1]: #wide
                border = (box[0] + box[2])/2
                new_box_1[2] = border
                new_box_2[0] = border
            else: #tall
                border = (box[1] + box[3])/2
                new_box_1[3] = border
                new_box_2[1] = border
            working.append(new_box_1)
            working.append(new_box_2)
        elif total_imgs == 0:
            continue
        else:
            city_total += total_imgs
            boxes.append(box)

    print(city_total)
    return boxes

def get_known_urls(cfg):
    urls = {}
    for key in cfg['cities']:
        city=key.replace(" ", "_")
        file_path=f'{DATA_DIR}/{city}/urls.txt'
        city_urls=set()
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    city_urls.add(line.strip())
        urls[key] = city_urls
    return urls

def write_urls(urls, cfg):
    for key in cfg['cities']:
        city=key.replace(" ", "_")
        directory=os.path.join(DATA_DIR, city)
        if not os.path.exists(directory):
            os.mkdir(directory)

        file_path=os.path.join(directory, 'urls')
        if cfg['cities'][key]['download'] != 'photos':
            print(f"printing {len(urls[city])} urls for city {city} at {file_path}")
            try:
                with open(file_path, 'w') as f:
                    for url in urls[city]:
                        f.write(f'{url}\n')
                    f.flush()
                    f.close()
            except Exception as err:
                print(f"error: {err} opening file {file_path}")                  

def get_metadata(cfg):
    metadata = None
    urls = get_known_urls(cfg)
    metadata, urls = read_metadata(cfg, urls)
    if cfg['refresh_metadata']:
        print('fetching metadata')
        metadata,urls = fetch_metadata(cfg, metadata, urls)
        print('writing metadata')
        write_metadata(metadata, cfg)
        print('writing url list')
        write_urls(urls, cfg)
        

    return metadata

def fetch_metadata(cfg, metadata, urls):
    FLICKR_PUBLIC = get_secret('flickr_api_key')
    FLICKR_SECRET = get_secret('flickr_api_secret')

    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

    license = "1,2,3,4,5,6,7,8,9,10"
    extras ='description,license,date_upload,date_taken,original_format,'
    extras+='last_update,geo,tags, machine_tags, o_dims, media,'
    extras+='url_m,url_n,url_z,url_c,url_l,url_o'

    inserted_ids=[]

    for key in cfg['cities']:
        count=0
        dl_limit = cfg['cities'][key]['download_limit']

        if dl_limit != -1 and dl_limit > 1000:
            boxes = get_usable_bounding_boxes(list(cfg['cities'][key]['bounding_boxes']), cfg)
        else:
            boxes = list(cfg['cities'][key]['bounding_boxes'])
        city_urls = urls[key]

        if not key in metadata:
            metadata[key]={}
            metadata[key]['image_count'] = 0
            metadata[key]['images'] = []
        total = 0

        for bbox in tqdm(boxes, desc=key):
            temp = list(map(str, bbox))
            bbox_str = ",".join(temp)

            time.sleep(cfg["time_delay"])
            total_pages=0
            try:
                city_pics = flickr.photos.search(
                    privacy_filter=PRIVACY_FILTER, bbox=bbox_str,
                    content_type=CONTENT_TYPE,
                    has_geo=HAS_GEO, geo_context=GEO_CTX,
                    license=license, extras=extras, per_page=cfg["page_size"])
                total_pages = city_pics['photos']['pages']
                total += int(city_pics['photos']['total'])
            except FlickrError as err:
                print(f'Error retrieving intitial page for bounding box {bbox}')
                print(f'{err}')

            for p in range(1, total_pages):
                try:
                    time.sleep(cfg["time_delay"])
                    city_pics = flickr.photos.search(
                        privacy_filter=PRIVACY_FILTER, bbox=bbox_str,
                        content_type=CONTENT_TYPE,
                        has_geo=HAS_GEO, geo_context=GEO_CTX,
                        license=license, extras=extras, per_page=cfg["page_size"],
                        page=p)
                    for ph in city_pics['photos']['photo']:
                        # metadata[key]['images'].append(ph)
                        if dl_limit != -1 and count > dl_limit:
                            break
                        if cfg["url_field"] in ph and not ph[cfg["url_field"]] in city_urls:
                            metadata[key]['images'].append(ph)
                            city_urls.add(ph[cfg["url_field"]])
                            metadata[key]['image_count']+=1
                            count += 1

                except FlickrError as err:
                    print(f'Error retrieving page {p} for bounding box {bbox}')
                    print(f'{err}')

        # metadata[key]['image_count'] = total
        # print(f"length of inserted ids for {key}: {len(inserted_ids)}")
        # print(f"total for {key}: {len(metadata[key]['images'])}")
    return metadata, urls

def write_metadata(metadata, cfg):
    for key in metadata:
        city=key.replace(" ", "_")
        directory=os.path.join(DATA_DIR,city)
        if not os.path.exists(directory):
            os.mkdir(directory)

        file_path=os.path.join(directory,'metadata.json')
        dl_flag =cfg['cities'][key]['download'] 
        if cfg['cities'][key]['download'] != 'photos':
            with open(file_path, 'w') as f:
                json.dump(metadata[key], f, indent=2)

def read_metadata(cfg, urls):
    metadata = {}
    for key in cfg['cities']:
        city=key.replace(" ", "_")
        file_path=f'{DATA_DIR}/{city}/metadata.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                loaded = json.load(f)
                for img in loaded['images']:
                    if cfg["url_field"] in img and not img[cfg["url_field"]] in urls:
                        urls[city].add(img[cfg["url_field"]])
                metadata[city]= loaded

    return metadata, urls

def download_photos(metadata, cfg):
    aki = get_aws_access_key_id()
    sak = get_aws_secret_access_key()
    st = get_aws_session_token()

    client = boto3.client('s3', region_name='us-east-2', aws_access_key_id=aki, aws_secret_access_key=sak, aws_session_token=st)
    lambda_client = boto3.client('lambda', region_name='us-east-2', aws_access_key_id=aki, aws_secret_access_key=sak, aws_session_token=st)

    for key in metadata:
        if cfg['cities'][key]['download'] == 'metadata':
            continue

        city=key.replace(" ", "_")
        directory=f'{DATA_DIR}/{city}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        file_path=os.path.join(directory,'/metadata.json')
        print(f'uploading metadata for {city} from file {file_path}')
        if os.path.exists(file_path):
            try:
                client.upload_file(file_path, Bucket=BUCKET, Key=f'{city}/metadata.json')
            except botocore.exceptions.ClientError as cerr:
                print(f'error uploading metadata file for {city}')
                print(f'{cerr}')

        photo_list = metadata[key]['images']
        dl_limit = cfg['cities'][key]['download_limit']
        for idx in range(0, len(photo_list)):
            if dl_limit != -1 and idx >= dl_limit:
                break

            if cfg["url_field"] in photo_list[idx]:
                url = photo_list[idx][cfg["url_field"]]
                lambda_payload = json.dumps({"city":city,"url":url}).encode()
                response = lambda_client.invoke(FunctionName='witw_uploader', 
                     InvocationType='RequestResponse',
                     Payload=lambda_payload)
                
                # file_name = url.split('/')[-1]
                # file_path=f'{directory}/{file_name}'
                # chunks=bytearray()
                # with open(file_path, 'wb') as download_file:
                try:
                    with httpx.stream("GET", url) as response:

                        with tqdm(unit_scale=True, unit_divisor=1024, unit="B") as progress:
                            num_bytes_downloaded = response.num_bytes_downloaded
                            for chunk in response.iter_bytes():
                                chunks.extend(chunk)
                                download_file.write(chunk)
                                progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                                num_bytes_downloaded = response.num_bytes_downloaded
                except httpx.ReadTimeout as err:
                    print(f'error downloading file {city}/{file_name}')

                # try:
                #     client.put_object(Body=chunks, Bucket=BUCKET, Key=f'{city }/{file_name}')
                # except botocore.exceptions.ClientError as cerr:
                #     print(f'error uploading file {city}/{file_name}')
                #     print(f'{cerr}')

        file_path=os.path.join(directory,'metadata.json')
        if os.path.exists(file_path):
            client.upload_file(file_path, Bucket=BUCKET, Key=f'{city }/metadata.json')

def main(config_file):
    config = parse_config(config_file)
    metadata = get_metadata(config)
    
    download_photos(metadata, config)

if __name__ == '__main__':  # pragma: no cover
    main('./config.yaml')
