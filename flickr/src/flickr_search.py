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

def get_metadata(cfg):
    FLICKR_PUBLIC = get_secret('flickr_api_key')
    FLICKR_SECRET = get_secret('flickr_api_secret')

    PRIVACY_FILTER = 1 #only public metadata
    CONTENT_TYPE = 1 #only metadata
    HAS_GEO = True  #only geotagged metadata
    GEO_CTX = 2     #flickr api values looking for outdoor images only

    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

    extras ='description,license,date_upload,date_taken,original_format,'
    extras+='last_update,geo,tags, machine_tags, o_dims, media,'
    extras+='url_m,url_n,url_z,url_c,url_l,url_o'

    metadata = {}
    for key in cfg:
        print(f'{key}')
        metadata[key]=[]
        for idx, bbox in enumerate(cfg[key]['bounding_boxes']):
            city_pics = flickr.photos.search(privacy_filter=PRIVACY_FILTER, bbox=bbox, content_type=CONTENT_TYPE,
            has_geo=HAS_GEO, geo_context=GEO_CTX, extras=extras, per_page=100)
            metadata[key].insert(idx,city_pics['photos'])

    return metadata

def main(config_file):
    config = parse_config(config_file)
    metadata = get_metadata(config)
    pprint(metadata)

if __name__ == '__main__':  # pragma: no cover
    main('../config.yaml')