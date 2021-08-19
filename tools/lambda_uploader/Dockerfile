FROM public.ecr.aws/lambda/python:3.8

COPY handler.py   ./
COPY requirements.txt ./
RUN python3 -m pip install -U pip && \
  pip3 install --no-cache-dir -r requirements.txt

CMD ["handler.handler"]
