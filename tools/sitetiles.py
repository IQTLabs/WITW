#!/usr/bin/env python

import os
import sys
import csv
import json
import tqdm
import numpy as np
import pandas as pd
from osgeo import osr
from osgeo import gdal

gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

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

fullnames = [
    'Rio de Janeiro',
    'Las Vegas',
    'Paris',
    'Shanghai',
    'Khartoum',
    'Atlanta',
    'Moscow',
    'Mumbai',
    'San Juan',
    'Dar es Salaam',
    'Rotterdam'
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

licenses = [
    ['All Rights Reserved',
     ''],
    ['Attribution-NonCommercial-ShareAlike License',
     'https://creativecommons.org/licenses/by-nc-sa/2.0/'],
    ['Attribution-NonCommercial License',
     'https://creativecommons.org/licenses/by-nc/2.0/'],
    ['Attribution-NonCommercial-NoDerivs License',
     'https://creativecommons.org/licenses/by-nc-nd/2.0/'],
    ['Attribution License',
     'https://creativecommons.org/licenses/by/2.0/'],
    ['Attribution-ShareAlike License',
     'https://creativecommons.org/licenses/by-sa/2.0/'],
    ['Attribution-NoDerivs License',
     'https://creativecommons.org/licenses/by-nd/2.0/'],
    ['No known copyright restrictions',
     'https://www.flickr.com/commons/usage/'],
    ['United States Government Work',
     'http://www.usa.gov/copyright.shtml'],
    ['Public Domain Dedication (CC0)',
     'https://creativecommons.org/publicdomain/zero/1.0/'],
    ['Public Domain Mark',
     'https://creativecommons.org/publicdomain/mark/1.0/']
]

columns = {'id':'id', 'author':'owner', 'surface_license_code':'license', 'lat':'latitude', 'lon':'longitude', 'surface_url':'url_m', 'surface_height':'height_m', 'surface_width':'width_m'}
columns_reverse = {value:key for key, value in columns.items()}


def json_to_dataframe(path, aoi):
    # Load dataframe from API metadata JSON file
    metadata = json.load(open(path))
    df = pd.DataFrame(metadata['images'])
    df = df[columns.values()]
    df.rename(columns=columns_reverse, inplace=True)
    df['aoi'] = aoi
    return df


def csv_to_dataframe(path):
    # Load dataframe from CSV file
    df = pd.read_csv(path, sep=',', header=0, dtype={'id':str})
    return df


def annotate_dataframe(df):
    # Adds additional information to json_to_dataframe() output
    df['surface_license_code'] = df['surface_license_code'].astype(int)
    df['surface_height'] = df['surface_height'].astype(int)
    df['surface_width'] = df['surface_width'].astype(int)
    df['aoi_name'] = df['aoi'].replace(range(1,1+len(names)), fullnames)
    df['surface_license'] = df['surface_license_code'].replace(
        range(len(licenses)), [x[0] for x in licenses])
    df['surface_license_url'] = df['surface_license_code'].replace(
        range(len(licenses)), [x[1] for x in licenses])
    df['overhead_license'] = 'Attribution-ShareAlike License'
    df['overhead_license_url'] = 'https://creativecommons.org/licenses/by-sa/4.0/'
    satellite_conditions = [df['aoi'].isin([1, 6, 11]),
                            df['aoi'].isin([2, 3, 4, 5, 7, 8, 9, 10])]
    satellite_names = ['WorldView-2', 'WorldView-3']
    df['overhead_satellite'] = np.select(satellite_conditions,
                                         satellite_names,
                                         default='NotSpecified')
    df['surface_path'] = 'surface/' + df['id'] + '.jpg'
    df['overhead_path'] = 'overhead/' + df['id'] + '.jpg'


def download(df, out_dir='/local_data/geoloc/sat/photos'):
    # Download Flickr photos
    # Note: Not recommended for production use
    for idx, row in df.iterrows():
        url = row['surface_url']
        dest = os.path.join(out_dir, row['id'] + '.jpg')
        cmd = 'wget ' + url + ' ' + dest
        print(cmd)
        os.system(cmd)


def clip(dframe, edge=225., max_out=None,
         sat_dir='/local_data/geoloc/sat/utm',
         out_dir='/local_data/geoloc/sat/tiles'):

    # Loop through all AOIs in the dataframe
    for aoi in np.sort(dframe['aoi'].unique()):
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
            out_path = os.path.join(out_dir, df.loc[i, 'id'] + '.jpg')
            window = [easting - edge/2., northing + edge/2.,
                      easting + edge/2., northing - edge/2.]
            gdal.Translate(out_path, sat_file, projWin=window)

        sat_file = None


if __name__ == '__main__':
    json_dir = '/local_data/geoloc/terrestrial/metadata'
    csv_path = '/local_data/geoloc/dataset/dataset.csv'

    if 'csv' in sys.argv[1:]: # JSON METADATA -> CSV FILE
        dfs = []
        for aoi in range(1,1+11):
            path = os.path.join(json_dir, names[aoi-1], 'metadata.json')
            df = json_to_dataframe(path, aoi=aoi)
            # Remove redundant entries and those without URL
            df.drop(df[df['surface_url'].isnull()].index, inplace=True)
            df.drop_duplicates(inplace=True, ignore_index=True)
            # Add additional information columns
            annotate_dataframe(df)
            dfs.append(df)
            print(aoi, len(df))
        df = pd.concat(dfs)
        print('all', len(df))

        # Optional shuffle
        # df = df.sample(frac=1).reset_index(drop=True)
        # Optional cutoff
        # df = df.head(500)

        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    if 'dataset' in sys.argv[1:]: # CSV FILE -> DATASET
        df = csv_to_dataframe(csv_path)
        # download(df)
        clip(df)
