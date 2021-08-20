
# Install 
```
pip install -r requirements.txt
```

# Setup 
User must set the dataset paths in the dictionary `dataset_paths` in `class Globals` located in [cvig_fov.py](cvig_fov.py)

# Usage 
```
> python cvig_fov.py -h
usage: cvig_fov.py [-h] [--mode {train,test}] [--dataset {cvusa,witw}] [--fov {6-360}]

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,test}   Run mode. [Default = train]
  --dataset {cvusa,witw}
                        Dataset to use. [Default = cvusa]
  --fov {6-360}         The field of view for cropping street level images. [Default = 360]
```
## Using the Docker container
Using the docker container requires the abilty to interface with Nvidia GPUs. To ensure that your machine supports Nvidia GPUs and has the proper drivers installed please run `nvidia-smi`. The docker container can be run natively but requires some specific flags to be set:
```
> docker run -it -v "<path to data>:/witw-model/data" -v "$(PWD)/weights:/witw-model/weights" --ipc=host --gpus all witw-model --mode {train, test} --dataset {cvusa,witw} --fov {6-360}

```
In order to streamline this process a `Makefile` has been provided as a shorthand. To use it simply set environment variables `DATA` with the path to your data, and `FOV` with the desired FOV value and invoke using make.
```
> export DATA=/home/user/data
> export FOV=180
> make train_cvusa
```
Accepted make values are: `train_cvusa, test_cvusa, train_witw, test_witw, build`

# Training FOV model 
Example: 
```
python cvig_fov.py --mode=train --dataset=cvusa --fov=90
```

# Testing FOV model 
Example: 
```
python cvig_fov.py --mode=test --dataset=cvusa --fov=90
```

# Notes
Script has been tested with Python 3.8.2
