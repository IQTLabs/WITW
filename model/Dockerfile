FROM nvidia/cuda:11.2.0-base
LABEL maintainer="Ryan Ashley <rashley@iqt.org>"

RUN apt-get update && apt-get install -y python3 python3-pip

COPY . /witw-model
WORKDIR /witw-model
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "cvig_fov.py"]
