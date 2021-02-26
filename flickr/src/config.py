import yaml

def parse_config(cfg_file='./config.yaml'):
    with open(cfg_file, 'r') as cf:
        parsed = yaml.safe_load(cf)
        effective_cfg={}
        for city in parsed["cities"]:
            city_cfg={}
            city_cfg["bounding_boxes"] = city["city"]["bounding_boxes"]
            city_cfg["download"] = city["city"]["download"]
            effective_cfg[city["city"]['name']] = city_cfg

    return effective_cfg
