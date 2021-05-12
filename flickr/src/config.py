import yaml

def parse_config(cfg_file='./config.yaml'):
    with open(cfg_file, 'r') as cf:
        parsed = yaml.safe_load(cf)
        effective_cfg={}
        effective_cfg["url_field"] = parsed["url_field"]
        effective_cfg["page_size"] = parsed["page_size"]
        effective_cfg["density_limit"] = parsed["density_limit"]
        effective_cfg["min_area"] = parsed["min_area"]
        effective_cfg["max_area"] = parsed["max_area"]
        effective_cfg["time_delay"] = parsed["time_delay"]
        effective_cfg["refresh_metadata"] = parsed["refresh_metadata"]
        effective_cfg["cities"] = {}
        for city in parsed["cities"]:
            city_cfg={}
            city_cfg["bounding_boxes"] = city["bounding_boxes"]
            city_cfg["download"] = city["download"]
            city_cfg["download_limit"] = city["download_limit"]
            effective_cfg["cities"][city["name"]] = city_cfg

    return effective_cfg
