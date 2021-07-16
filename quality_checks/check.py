#!/usr/bin/env python3

def check_duplicates(metadata):


def get_metadata(dir):


def main():
    metadata,urls = read_metadata()
    send_to_lambda(urls)

if __name__ == '__main__':  # pragma: no cover
    main()