# FaceLandmark

*__NOTE: This repo is not ready and is on active development.__*



## Getting Started

A separate Python environment is recommended.
+ Python3.5+ (Python3.5, Python3.6 are tested)
+ Pytorch == 1.0
+ opencv4 (opencv3.4.5 is tested also)
+ numpy

install dependences using `pip`
```bash
pip3 install numpy opencv-python
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision (optional)
```
or install using `conda`
```bash
conda install opencv numpy
conda install pytorch-cpu torchvision-cpu -c pytorch
```
If you want to train your model, GPU version is recommended.
```bash
conda install pytorch torchvision -c pytorch
```

## Usage
1. clone the repo first.
```bash
git clone https://github.com/siriusdemon/ShuffleNet-Face-Landmark.git
cd ShuffleNet-Face-Landmark
```
2. for detection, follow `detect.py` script
```bash
python detect.py
```
3. for training, you can define your own model in `shuffflenet.py` and replace in `train.py`. `train.py` already provides an example.
```bash
python train.py
```


# Reference & Thanks
+ https://github.com/jaxony/ShuffleNet.git
