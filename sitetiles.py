#!/usr/bin/env python

import os
import json
import tqdm
import pandas as pd
from osgeo import osr
from osgeo import gdal

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

columns = {'id':'id', 'author':'owner', 'license_code_ground':'license', 'lat':'latitude', 'lon':'longitude', 'url':'url_m', 'height':'height_m', 'width':'width_m'}
columns_reverse = {value:key for key, value in columns.items()}


def json_to_dataframe(path, aoi):
    metadata = json.load(open(path))
    df = pd.DataFrame(metadata['images'])
    df = df[columns.values()]
    df.rename(columns=columns_reverse, inplace=True)
    df['aoi'] = aoi
    return df


def csv_to_dataframe(path):
    df = pd.read_csv(path, sep=',', header=0)
    return df


def clip(dframe, edge=225., max_out=None,
         sat_dir='/local_data/geoloc/sat/utm',
         out_dir='/local_data/geoloc/sat/tiles'):

    # Loop through all AOIs in the dataframe
    for aoi in dframe['aoi'].unique():
        df = dframe[dframe['aoi']==aoi]
        df.reset_index(inplace=True)
        aoi = int(aoi)
        print('AOI', aoi)

        # Set up coordinate conversion
        photo_crs = osr.SpatialReference()
        photo_crs.ImportFromEPSG(4326)
        sat_crs = osr.SpatialReference()
        sat_crs.ImportFromEPSG(epsgs[aoi-1])
        ct = osr.CoordinateTransformation(photo_crs, sat_crs)

        # Specify satellite image
        sat_path = os.path.join(sat_dir, names[aoi-1] + '.tif')
        sat_file = gdal.Open(sat_path)

        if max_out is not None:
            num_tiles = min(len(df), max_out)
        else:
            num_tiles = len(df)
        for i in tqdm.tqdm(range(num_tiles)):

            # Get UTM coordinates
            lon, lat = (float(df.loc[i, 'lon']), float(df.loc[i, 'lat']))
            easting, northing = ct.TransformPoint(lat, lon)[:2]

            # Write a tile of satellite imagery
            out_path = os.path.join(out_dir, names[aoi-1],
                                    df.loc[i, 'id'] + '.jpg')
            window = [easting - edge/2., northing + edge/2.,
                      easting + edge/2., northing - edge/2.]
            gdal.Translate(out_path, sat_file, projWin=window)

        sat_file = None


if __name__ == '__main__':
    if True: #JSON DATA
        dfs = []
        for aoi in range(1,1+11):
            path = os.path.join('../api/data', names[aoi-1], 'metadata.json')
            df = json_to_dataframe(path, aoi=aoi)
            print(aoi, len(df))
            df.drop_duplicates(inplace=True, ignore_index=True)
            print(aoi, len(df))
            dfs.append(df)
        df = pd.concat(dfs)
        print('all', len(df))
        
        df.to_csv('../api/candidate_photos.csv', index=False)
        #clip(df)

    if False: #CSV DATA
        df = csv_to_dataframe('landmark_locations.csv')
        clip(df)
