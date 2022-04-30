# flake8: noqa
import os
import os.path as osp
os_path=os.getcwd()
import sys
sys.path.append(os_path)
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
