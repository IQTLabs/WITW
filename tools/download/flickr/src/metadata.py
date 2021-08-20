import json
import os
import httpx
import time

def get_cities(cfg):
	return cfg['cities'].keys()

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

def read_metadata(file_root, cities, url_field):
    metadata = {}
    urls = {}
    # for key in cfg['cities']:
        # city=key.replace(" ", "_")
    for city in cities:
        urls[city]=set()

        file_path=f'{file_root}/{city}/metadata.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                loaded = json.load(f)
                for img in loaded['images']:
                    if url_field in img and not img[url_field] in urls:
                        urls[city].add(img[url_field])
                metadata[city]= loaded

    return metadata, urls

def get_known_urls(file_root, cities):
    urls = {}
    for key in cities:
        city=key.replace(" ", "_")
        file_path=f'{file_root}/{city}/urls.txt'
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
        directory=os.path.join('/data', city)
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

def get_metadata(cfg, file_root):
    metadata = None
    cities = get_cities(cfg)
    url_field = cfg['url_field']
    urls = get_known_urls(file_root, cities)
    metadata, urls = read_metadata(file_root, cities, url_field)
    if cfg['refresh_metadata']:
        print('fetching metadata')
        metadata,urls = fetch_metadata(cfg, metadata, urls)
        print('writing metadata')
        write_metadata(metadata, cfg, file_root)
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

def write_metadata(metadata, cfg, file_root):
    for key in metadata:
        city=key.replace(" ", "_")
        directory=os.path.join(file_root,city)
        if not os.path.exists(directory):
            os.mkdir(directory)

        file_path=os.path.join(directory,'metadata.json')
        dl_flag =cfg['cities'][key]['download'] 
        if cfg['cities'][key]['download'] != 'photos':
            with open(file_path, 'w') as f:
                json.dump(metadata[key], f, indent=2)
