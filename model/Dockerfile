FROM nvidia/cuda:11.2.0-base
LABEL maintainer="Ryan Ashley <rashley@iqt.org>"

COPY . .
RUN pip install -r requirements.txt
CMD ["python", "cvig_fov.py"]