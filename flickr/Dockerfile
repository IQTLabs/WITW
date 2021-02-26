FROM python:3.9-slim
LABEL maintainer="Ryan Ashley <rashley@iqt.org>"

COPY flickr/src .
COPY flickr/config.yaml .

RUN apt-get update && \
  python3 -m pip install -U pip && \
  pip3 install --no-cache-dir -r requirements.txt && \
  apt -y autoremove --purge && rm -rf /var/cache/* /root/.cache/*

CMD python3 flickr_search.py