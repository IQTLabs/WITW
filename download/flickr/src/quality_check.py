import json
import os
import httpx
import time

from config import parse_config
from metadata import *
from secrets import *

def get_provisional_list():
    pl = []
    file_path = "../provisional_complete_list.txt"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pl.append(line.strip())
    return pl

def get_provisional_list_by_city(provisional_list, cities):
    by_city={}
    for item in provisional_list:
        i = item.split('/')
        if len(i) == 3 and i[1] in cities:
            if not i[1] in by_city:
                by_city[i[1]] = []
            by_city[i[1]].append(i[2])
    
    return by_city

def check_duplicates(urls, metadata, provisional_list, cities, url_field):
    all_urls=[]
    pl_by_city = get_provisional_list_by_city(provisional_list, cities)
    metadata_dups={}
    provisional_dups={}
    no_urlm={}
    for city in cities:
        city_urls = list( f'{city}/{url}' for url in urls[city])
        all_urls.extend(city_urls)
        temp=set()
        temp_add=temp.add
        metadata_dups[city] = set( x for x in city_urls if x in temp or temp_add(x) )
        temp=set()
        temp_add=temp.add
        provisional_dups[city] = set( x for x in pl_by_city[city] if x in temp or temp_add(x) )

        no_urlm[city] = set( x['id'] for x in metadata[city]['images'] if not url_field in x )

        print(f'{city}: {len(city_urls)} - provisional: {len(pl_by_city[city])} - diff: {len(city_urls) - len(pl_by_city[city])} - {len(metadata_dups[city])} - {len(provisional_dups[city])} - {len(no_urlm[city])}')

    print(f'all: {len(all_urls)} - provisional_list: {len(provisional_list)}')


def main(config_file):
    config = parse_config(config_file)
    file_root = '../../flickr_data'
    url_field = config['url_field']
    cities = list(c.replace(" ", "_") for c in get_cities(config))

    urls = get_known_urls(file_root, cities)
    metadata, urls = read_metadata(file_root, cities, url_field)
    prov_list = get_provisional_list()

    check_duplicates(urls, metadata, prov_list, cities, url_field)
    
    

if __name__ == '__main__':  # pragma: no cover
    main('../config.yaml')