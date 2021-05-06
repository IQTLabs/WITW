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

URL_FIELD='url_m'
PAGE_SIZE=100
DENSITY_LIMIT=4000

PRIVACY_FILTER = 1 #only public metadata
CONTENT_TYPE = 1 #only metadata
HAS_GEO = True  #only geotagged metadata
GEO_CTX = 0     #0=all, 1=indoor, 2=outdoor

MIN_AREA = 1. #m^2; bounding boxes smaller than this are never divided
MAX_AREA = 2.E6 #m^2; bounding boxes larger than this are always divided

TIME_DELAY = 2.

def get_secret(secret_name):
    try:
        with open('/run/secrets/{0}'.format(secret_name), 'r') as secret_file:
            return secret_file.read()
    except IOError:
        return None

def est_area(bbox, radius=6.371E6):
    # Estimated area of a small bounding box, in m^2
    return (bbox[3]-bbox[1]) * (bbox[2]-bbox[0]) \
        * math.cos((bbox[1]+bbox[3])/2 * math.pi / 180.) \
        * (math.pi / 180.) **2 * radius**2

def get_usable_bounding_boxes(nominal_boxes):
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

    print('  area_km2 count type   bounding_box')
    while len(working) > 0:
        box = working.pop()
        temp = list(map(str, box))
        str_box = ",".join(temp)
        box_area = est_area(box)
        divide_flag = False
        if box_area > MAX_AREA:
            total_imgs = -1
            divide_flag = True
        else:
            time.sleep(TIME_DELAY)
            box_pics = flickr.photos.search(
                privacy_filter=PRIVACY_FILTER, bbox=str_box,
                content_type=CONTENT_TYPE,
                has_geo=HAS_GEO, geo_context=GEO_CTX,
                license=license, extras=extras, per_page=PAGE_SIZE)
            total_imgs = int(box_pics['photos']['total'], base=10)
            divide_flag = (total_imgs >= DENSITY_LIMIT and box_area > MIN_AREA)
        print('%10.4f %5i %s %s' % (box_area/1.E6, total_imgs, 'branch'
                                    if divide_flag else 'leaf  ', box))
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

def get_metadata(cfg):
    FLICKR_PUBLIC = get_secret('flickr_api_key')
    FLICKR_SECRET = get_secret('flickr_api_secret')

    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

    license = "1,2,3,4,5,6,7,8,9,10"
    extras ='description,license,date_upload,date_taken,original_format,'
    extras+='last_update,geo,tags, machine_tags, o_dims, media,'
    extras+='url_m,url_n,url_z,url_c,url_l,url_o'

    metadata = {}
    inserted_ids=[]

    for key in cfg:
        boxes = get_usable_bounding_boxes(list(cfg[key]['bounding_boxes']))
        metadata[key]={}
        # metadata[key]['image_count'] = total
        metadata[key]['images'] = []
        total = 0

        for bbox in tqdm(boxes, desc=key):
            temp = list(map(str, bbox))
            bbox_str = ",".join(temp)

            time.sleep(TIME_DELAY)
            city_pics = flickr.photos.search(
                privacy_filter=PRIVACY_FILTER, bbox=bbox_str,
                content_type=CONTENT_TYPE,
                has_geo=HAS_GEO, geo_context=GEO_CTX,
                license=license, extras=extras, per_page=PAGE_SIZE)
            total_pages = city_pics['photos']['pages']
            total += int(city_pics['photos']['total'], base=10)

            for p in range(1, total_pages):
                try:
                    time.sleep(TIME_DELAY)
                    city_pics = flickr.photos.search(
                        privacy_filter=PRIVACY_FILTER, bbox=bbox_str,
                        content_type=CONTENT_TYPE,
                        has_geo=HAS_GEO, geo_context=GEO_CTX,
                        license=license, extras=extras, per_page=PAGE_SIZE,
                        page=p)
                    for ph in city_pics['photos']['photo']:
                        metadata[key]['images'].append(ph)
                        # if not ph['id'] in inserted_ids:
                        #     metadata[key]['images'].append(ph)
                        #     inserted_ids.append(ph['id'])

                except FlickrError as err:
                    print(f'Error retrieving page {p} for bounding box {bbox}')
                    print(f'{err}')

        metadata[key]['image_count'] = total
        # print(f"length of inserted ids for {key}: {len(inserted_ids)}")
        # print(f"total for {key}: {len(metadata[key]['images'])}")
    return metadata

def write_metadata(metadata, cfg):
    for key in metadata:
        city=key.replace(" ", "_")
        directory=f'/data/{city}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        file_path=f'{directory}/metadata.json'
        if cfg[key]['download'] != 'photos':
            with open(file_path, 'w') as f:
                json.dump(metadata[key], f, indent=2)

def download_photos(metadata, cfg):
    for key in metadata:
        if cfg[key]['download'] == 'metadata':
            continue

        city=key.replace(" ", "_")
        directory=f'/data/{city}'
        if not os.path.exists(directory):
            os.mkdir(directory)

        photo_list = metadata[key]['photo']
        dl_limit = cfg[key]['download_limit']
        for idx in range(0, len(photo_list)):
            if idx >= dl_limit:
                break

            if URL_FIELD in photo_list[idx]:
                url = photo_list[idx][URL_FIELD]
                file_name = url.split('/')[-1]
                file_path=f'{directory}/{file_name}'
                with open(file_path, 'wb') as download_file:
                    with httpx.stream("GET", url) as response:

                        with tqdm(unit_scale=True, unit_divisor=1024, unit="B") as progress:
                            num_bytes_downloaded = response.num_bytes_downloaded
                            for chunk in response.iter_bytes():
                                download_file.write(chunk)
                                progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                                num_bytes_downloaded = response.num_bytes_downloaded


def main(config_file):
    config = parse_config(config_file)
    metadata = get_metadata(config)
    write_metadata(metadata, config)
    download_photos(metadata, config)

if __name__ == '__main__':  # pragma: no cover
    main('./config.yaml')
