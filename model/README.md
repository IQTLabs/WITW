
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
