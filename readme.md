# PQTrack: Finegrained reid tracking method by Part Querier.

# News

# Introduction

# Results and Models

## Detector

## Reid

# Running 
## Install 

We implement PQTrack based on [MMDectection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv). 

**Step 0.** Install [MMEngie](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim):
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install PQTrack.

Install the project to local environment.

```bash
git clone xxxxx
cd PQTrack
pip install -v -e . -r requirements/tracking.txt
```
**Step 2.** Install TrackEval.
```
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

## Data 

Please follow the guide in [MMDectection-tracking](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/tracking_dataset_prepare.md) to prepare the datasets as structure as :
```
PQTrack
├── data
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── annotations
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   |   ├── MOT17-02-DPM
|   |   |   |   ├── det
|   │   │   │   ├── gt
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
|   |   ├── test
|   |   |   ├── MOT17-01-DPM
|   |   |   |   ├── det
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
│   │
│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip
│   │
```

## Training 
Train PQTrack with 8 GPUs:
```
sh tools/dist_train.sh projects/configs/xxx 8 path_to_exp
```

## Testing
Test on MOT17-half:
```
sh tools/dist_test.sh  projects/configs/xxxxxx 8 --eval bbox
```

# Cite

