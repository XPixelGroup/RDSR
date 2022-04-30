import numpy as np
import torch
import cv2
from ModelZoo.utils import _add_batch_one
import math


def attribution_objective(attr_func):
    def calculate_objective(image):
        return attr_func(image)
    return calculate_objective


def Path_gradient(tensor_image, model, attr_objective):

    img_tensor = tensor_image
    img_tensor.requires_grad_(True)
    first, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, \
    middle, last, result = model(_add_batch_one(img_tensor))
    target = attr_objective(result)

    first.retain_grad()
    b1.retain_grad()
    b2.retain_grad()
    b3.retain_grad()
    b4.retain_grad()
    b5.retain_grad()
    b6.retain_grad()
    b7.retain_grad()
    b8.retain_grad()
    b9.retain_grad()
    b10.retain_grad()
    b11.retain_grad()
    b12.retain_grad()
    b13.retain_grad()
    b14.retain_grad()
    b15.retain_grad()
    b16.retain_grad()
    middle.retain_grad()
    last.retain_grad()
    result.retain_grad()

    target.backward()

    first_grad = first.grad
    b1_grad = b1.grad
    b2_grad = b2.grad
    b3_grad = b3.grad
    b4_grad = b4.grad
    b5_grad = b5.grad
    b6_grad = b6.grad
    b7_grad = b7.grad
    b8_grad = b8.grad
    b9_grad = b9.grad
    b10_grad = b10.grad
    b11_grad = b11.grad
    b12_grad = b12.grad
    b13_grad = b13.grad
    b14_grad = b14.grad
    b15_grad = b15.grad
    b16_grad = b16.grad
    middle_grad = middle.grad
    last_grad = last.grad
    final_grad = result.grad

    grad = img_tensor.grad.cpu().numpy()
    first_grad = first_grad.cpu().numpy()
    b1_grad = b1_grad.cpu().numpy()
    b2_grad = b2_grad.cpu().numpy()
    b3_grad = b3_grad.cpu().numpy()
    b4_grad = b4_grad.cpu().numpy()
    b5_grad = b5_grad.cpu().numpy()
    b6_grad = b6_grad.cpu().numpy()
    b7_grad = b7_grad.cpu().numpy()
    b8_grad = b8_grad.cpu().numpy()
    b9_grad = b9_grad.cpu().numpy()
    b10_grad = b10_grad.cpu().numpy()
    b11_grad = b11_grad.cpu().numpy()
    b12_grad = b12_grad.cpu().numpy()
    b13_grad = b13_grad.cpu().numpy()
    b14_grad = b14_grad.cpu().numpy()
    b15_grad = b15_grad.cpu().numpy()
    b16_grad = b16_grad.cpu().numpy()
    middle_grad = middle_grad.cpu().numpy()
    last_grad = last_grad.cpu().numpy()
    final_grad = final_grad.cpu().numpy()

    if np.any(np.isnan(grad)):
        grad[np.isnan(grad)] = 0.0
    if np.any(np.isinf(grad)):
        grad[np.isinf(grad)] = 0

    return first_grad[0], b1_grad[0], b2_grad[0], b3_grad[0], b4_grad[0], b5_grad[0], b6_grad[0], \
           b7_grad[0], b8_grad[0], b9_grad[0], b10_grad[0], b11_grad[0], b12_grad[0], b13_grad[0], b14_grad[0], \
           b15_grad[0], b16_grad[0], middle_grad[0], last_grad[0], final_grad[0]


def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]

