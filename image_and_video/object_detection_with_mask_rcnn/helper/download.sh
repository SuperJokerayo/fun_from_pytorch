#!/bin/bash
wget -e https_proxy=127.0.0.1:7890 https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
wget -e https_proxy=127.0.0.1:7890 https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
wget -e https_proxy=127.0.0.1:7890 https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
wget -e https_proxy=127.0.0.1:7890 https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
wget -e https_proxy=127.0.0.1:7890 https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py