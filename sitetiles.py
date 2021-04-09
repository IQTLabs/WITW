#!/usr/bin/env python

import json
import pandas as pd

columns = {'author':'owner', 'license_code_ground':'license', 'lat':'latitude', 'lon':'longitude', 'url':'url_m', 'height':'height_m', 'width':'width_m'}
columns_reverse = {value:key for key, value in columns.items()}

def json_to_dataframe(path):
    metadata = json.load(open(path))
    print(len(metadata['photo']))
    photos = None
    for page in metadata:
        if photos is None:
            photos = page['photo']
        else:
            photos.extend(page['photo'])
    df = pd.DataFrame(photos)
    df = df[columns.values()]
    df.rename(columns=columns_reverse, inplace=True)
    return df


def main(df):
    print(df)


if __name__ == '__main__':
    df = json_to_dataframe('../api/flickr_data/02_vegas/metadata.json')
    main(df)
