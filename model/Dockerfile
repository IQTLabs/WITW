FROM nvidia/cuda:11.2.0-base
LABEL maintainer="Ryan Ashley <rashley@iqt.org>"

RUN apt-get update && apt-get install -y python3 python3-pip

COPY . .
RUN pip3 install -r requirements.txt
CMD ["python3", "cvig_fov.py"]