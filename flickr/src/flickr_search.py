import os
from flickrapi import FlickrAPI
from pprint import pprint

from config import parse_config

def get_secret(secret_name):
    try:
        with open('/run/secrets/{0}'.format(secret_name), 'r') as secret_file:
            return secret_file.read()
    except IOError:
        return None

FLICKR_PUBLIC = get_secret('flickr_api_key')
FLICKR_SECRET = get_secret('flickr_api_secret')

PRIVACY_FILTER = 1 #only public photos
CONTENT_TYPE = 1 #only photos
HAS_GEO = True  #only geotagged photos
GEO_CTX = 2     #flickr api values looking for outdoor images only

#DC=[[[-77.0990375584,38.8024915206],[-76.9067768162,38.8024915206],[-76.9067768162,39.0012743239],[-77.0990375584,39.0012743239],[-77.0990375584,38.8024915206]]]
cfg = parse_config('../config.yaml')
flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

extras ='description,license,date_upload,date_taken,original_format,'
extras+='last_update,geo,tags, machine_tags, o_dims, media,'
extras+='url_m,url_n,url_z,url_c,url_l,url_o'

# bounds = '-77.0990375584, 38.8024915206, -76.9067768162, 39.0012743239'
photos = {}
for key in cfg:
    print(f'{key}')
    photos[key]=[]
    for idx, bbox in enumerate(cfg[key]['bounding_boxes']):
        city_pics = flickr.photos.search(privacy_filter=PRIVACY_FILTER, bbox=bbox, content_type=CONTENT_TYPE,
        has_geo=HAS_GEO, geo_context=GEO_CTX, extras=extras, per_page=100)
        photos[key].insert(idx,city_pics['photos'])

pprint(photos)