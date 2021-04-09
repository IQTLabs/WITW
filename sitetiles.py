#!/usr/bin/env python

import os
import json
import pandas as pd
from osgeo import osr
from osgeo import gdal

columns = {'id':'id', 'author':'owner', 'license_code_ground':'license', 'lat':'latitude', 'lon':'longitude', 'url':'url_m', 'height':'height_m', 'width':'width_m'}
columns_reverse = {value:key for key, value in columns.items()}

names = [
    '01_rio',
    '02_vegas',
    '03_paris',
    '04_shanghai',
    '05_khartoum',
    '06_atlanta',
    '07_moscow',
    '08_mumbai',
    '09_san',
    '10_dar',
    '11_rotterdam',
]

epsgs = [
    32723,
    32611,
    32631,
    32651,
    32636,
    32616,
    32637,
    32643,
    32620,
    32737,
    32631,
]


def json_to_dataframe(path):
    metadata = json.load(open(path))
    # photos = None
    # for page in metadata:
    #     if photos is None:
    #         photos = page['photo']
    #     else:
    #         photos.extend(page['photo'])
    df = pd.DataFrame(metadata['photo'])
    df = df[columns.values()]
    df.rename(columns=columns_reverse, inplace=True)
    return df


def clip(df, aoi=1, max_out=10,
         sat_dir='/local_data/geoloc/sat/utm',
         out_dir='/local_data/geoloc/sat/tiles'):
    print(df)
    print(df.iloc[0])

    # Set up coordinate conversion
    photo_crs = osr.SpatialReference()
    photo_crs.ImportFromEPSG(4326)
    sat_crs = osr.SpatialReference()
    sat_crs.ImportFromEPSG(epsgs[aoi-1])
    ct = osr.CoordinateTransformation(photo_crs, sat_crs)

    # Specify satellite image
    sat_path = os.path.join(sat_dir, names[aoi-1] + '.tif')
    sat_file = gdal.Open(sat_path)

    for i in range(len(df)):
        if max_out is not None and i >= max_out:
            break

        # Get UTM coordinates
        lon, lat = (float(df.loc[i, 'lon']), float(df.loc[i, 'lat']))
        easting, northing = ct.TransformPoint(lat, lon)[:2]

        out_path = os.path.join(out_dir, names[aoi-1],
                                df.loc[i, 'id'] + '.tif')
        #gdal.Translate(out_path, sat_file)

    sat_file = None


if __name__ == '__main__':
    df = json_to_dataframe('../api/flickr_data/02_vegas/metadata.json')
    clip(df, aoi=2)
