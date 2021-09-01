#!/usr/bin/env python

import sys
import argparse

sys.path.append('../../model')
import cvig_fov as cvig
#Globals = cvig.Globals
#device = cvig.device


def main(option, dataset_format, csv_path, surface_dir, overhead_dir):

    dataset = cvig.ImagePairDataset(dataset_format, csv_path)
    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(overhead_dir, exist_ok=True)

if __name__ == '__main__':

    default_dataset = 'cvusa'
    default_csv = '/local_data/cvusa/all-19zl.csv'
    default_surface = '/local_data/geoloc/cvusamod/bingmap/19'
    default_overhead = '/local_data/geoloc/cvusamod/streetview/panos'

    parser = argparse.ArgumentParser()
    parser.add_argument('option',
                        default=0,
                        help='Image modification option')
    parser.add_argument('--dataset',
                        default=default_dataset,
                        choices=['cvusa', 'witw'],
                        help='Dataset format')
    parser.add_argument('--csv',
                        default=default_csv,
                        help='Path to source dataset\'s CSV file')
    parser.add_argument('--surface',
                        default=default_surface,
                        help='Directory to save surface images')
    parser.add_argument('--overhead',
                        default=default_overhead,
                        help='Directory to save overhead images')
    args = parser.parse_args()
    main(args.option, args.dataset, args.csv, args.surface, args.overhead)
