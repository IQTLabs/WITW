#!/usr/bin/env python

import os
import json
import tqdm
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


def json_to_dataframe(path, aoi):
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
    df['aoi'] = aoi
    return df


def clip(df, edge=250., max_out=5,
         sat_dir='/local_data/geoloc/sat/utm',
         out_dir='/local_data/geoloc/sat/tiles'):

    # Assumes same AOI for entire DataFrame
    aoi = df.loc[0, 'aoi']

    # Set up coordinate conversion
    photo_crs = osr.SpatialReference()
    photo_crs.ImportFromEPSG(4326)
    sat_crs = osr.SpatialReference()
    sat_crs.ImportFromEPSG(epsgs[aoi-1])
    ct = osr.CoordinateTransformation(photo_crs, sat_crs)

    # Specify satellite image
    sat_path = os.path.join(sat_dir, names[aoi-1] + '.tif')
    sat_file = gdal.Open(sat_path)

    num_tiles = min(len(df), max_out)
    for i in tqdm.tqdm(range(num_tiles)):

        # Get UTM coordinates
        lon, lat = (float(df.loc[i, 'lon']), float(df.loc[i, 'lat']))
        easting, northing = ct.TransformPoint(lat, lon)[:2]

        # Write a tile of satellite imagery
        out_path = os.path.join(out_dir, names[aoi-1],
                                df.loc[i, 'id'] + '.tif')
        window = [easting - edge/2., northing + edge/2.,
                  easting + edge/2., northing - edge/2.]
        gdal.Translate(out_path, sat_file, projWin=window)

    sat_file = None


if __name__ == '__main__':
    df = json_to_dataframe('../api/flickr_data/02_vegas/metadata.json', aoi=2)
    clip(df)
