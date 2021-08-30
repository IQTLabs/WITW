#!/usr/bin/env python

import csv
import argparse
import numpy as np
import pandas as pd
import tqdm

# Modified from: CosmiQ Solaris
# https://github.com/CosmiQ/solaris/blob/master/solaris/preproc/sar.py
def haversine(lat1, lon1, lat2, lon2, rad=False, radius=6.371E6):
    """
    Haversine formula for distance between two points given their
    latitude and longitude, assuming a spherical earth.
    """
    if not rad:
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * radius * np.arcsin(np.sqrt(a))


def main(input_path, output_path, threshold, randomize=True):

    # Input and output dataframes
    dfi = pd.read_csv(input_path, sep=',', header=0, dtype={'id':str})
    dfo = dfi.iloc[0:0,:].copy()

    # Loop through AOIs
    aois = np.sort(dfi['aoi'].unique())
    for aoi in aois:
        print('AOI', aoi)
        dfai = dfi[dfi.aoi == aoi]
        dfao = dfai.iloc[0:0,:].copy()
        if randomize:
            dfai = dfai.sample(frac=1).reset_index(drop=True)
        for index, row in tqdm.tqdm(dfai.iterrows(), total=len(dfai)):
            lat = np.array(row['lat'])
            lon = np.array(row['lon'])
            dists = haversine(lat, lon, dfao['lat'], dfao['lon'])
            if len(dists) > 0:
                min_dist = np.min(dists)
            else:
                min_dist = np.inf
            if min_dist >= threshold:
                dfao = dfao.append(row)
        dfo = dfo.append(dfao)

    # Write output to disk
    dfo.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('threshold', nargs='?', type=float, default=10.)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.threshold)
