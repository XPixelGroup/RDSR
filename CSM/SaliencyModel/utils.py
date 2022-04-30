import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import cv2


def cv2_to_pil(img):
    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def vis_saliency(map, zoomin=4):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    cmap = plt.get_cmap('seismic')
    map_color = (255 * cmap(map * 0.5 + 0.5)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    Img = Img.resize((s1 * zoomin, s2 * zoomin), Image.NEAREST)
    return Img.convert('RGB')

def vis_saliency_nozoomin(map):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    cmap = plt.get_cmap('seismic')
    map_color = (255 * cmap(map * 0.5 + 0.5)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    return Img.convert('RGB')


def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil


def grad_abs_norm(grad):
    """
    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad.sum(axis=0))
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max
    return grad_norm


def grad_norm(grad):
    """
    :param grad: numpy array
    :return:
    """
    grad_2d = grad.sum(axis=0)
    grad_max = max(grad_2d.max(), abs(grad_2d.min()))
    grad_norm = grad_2d / grad_max
    return grad_norm


def grad_abs_norm_singlechannel(grad):
    """
    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad)
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max
    return grad_norm
