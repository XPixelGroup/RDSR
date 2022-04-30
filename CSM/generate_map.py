import math
import metric
from PIL import Image
import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from ModelZoo import get_model, load_model_now
from ModelZoo.utils import Tensor2PIL, PIL2Tensor, _add_batch_one
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2
from SaliencyModel.utils import (vis_saliency, vis_saliency_nozoomin,
                                 grad_abs_norm, grad_norm, grad_abs_norm_singlechannel)
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from feature_map import load_model_feature

#关键在于 model and point
source_dir='datasets/Set5/LRbicx4'
gt_dir = 'datasets/Set5/GTmod12'

model_list=[
['RealSRResNetx4_details_8class_abs','ModelZoo/pretrained_models/RealSRResNetx4_details.pth'],
['RealSRResNetx4_details_droplast_channel07_8class_abs','ModelZoo/pretrained_models/RealSRResNetx4_details_droplast_channel07.pth'],
]

models=[]
for model in model_list:
    models.append([model[0],load_model_now(model[0],model[1]),model[1]])

def main():

    for filename in sorted(os.listdir(source_dir)):

        img_lr = Image.open(os.path.join(source_dir, filename))
        tensor_lr = PIL2Tensor(img_lr)[:3]
        img_hr = Image.open(os.path.join(gt_dir, filename))
        tensor_hr = PIL2Tensor(img_hr)[:3]

        attr_objective = attribution_objective(attr_grad)

        for model in models:
            print(model[0])
            save_dir = 'results/map_results/'+ model[0] + '/' + filename.split('.')[0]
            if os.path.exists(save_dir):
                pass
            else:
                os.makedirs(save_dir)

            for location in ['first', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11',
                             'b12', 'b13', 'b14', 'b15','b16', 'hr', 'last']:
                small_dir = save_dir + '/' + location
                if os.path.exists(small_dir):
                    pass
                else:
                    os.makedirs(small_dir)

            first_grad, b1_grad, b2_grad, b3_grad, b4_grad, b5_grad, b6_grad, b7_grad, b8_grad, \
            b9_grad, b10_grad, b11_grad, b12_grad, b13_grad, b14_grad, b15_grad, b16_grad, middle_grad, \
            last_grad, final_grad = Path_gradient(tensor_lr,model[1], attr_objective)

            abs_normed_first_grad = grad_abs_norm_singlechannel(first_grad)
            abs_normed_b1_grad = grad_abs_norm_singlechannel(b1_grad)
            abs_normed_b2_grad = grad_abs_norm_singlechannel(b2_grad)
            abs_normed_b3_grad = grad_abs_norm_singlechannel(b3_grad)
            abs_normed_b4_grad = grad_abs_norm_singlechannel(b4_grad)
            abs_normed_b5_grad = grad_abs_norm_singlechannel(b5_grad)
            abs_normed_b6_grad = grad_abs_norm_singlechannel(b6_grad)
            abs_normed_b7_grad = grad_abs_norm_singlechannel(b7_grad)
            abs_normed_b8_grad = grad_abs_norm_singlechannel(b8_grad)
            abs_normed_b9_grad = grad_abs_norm_singlechannel(b9_grad)
            abs_normed_b10_grad = grad_abs_norm_singlechannel(b10_grad)
            abs_normed_b11_grad = grad_abs_norm_singlechannel(b11_grad)
            abs_normed_b12_grad = grad_abs_norm_singlechannel(b12_grad)
            abs_normed_b13_grad = grad_abs_norm_singlechannel(b13_grad)
            abs_normed_b14_grad = grad_abs_norm_singlechannel(b14_grad)
            abs_normed_b15_grad = grad_abs_norm_singlechannel(b15_grad)
            abs_normed_b16_grad = grad_abs_norm_singlechannel(b16_grad)
            abs_normed_middle_grad = grad_abs_norm_singlechannel(middle_grad)
            abs_normed_last_grad = grad_abs_norm_singlechannel(last_grad)
            abs_normed_final_grad = grad_abs_norm_singlechannel(final_grad)

            first_net, b1_net, b2_net, b3_net, b4_net, b5_net, b6_net, b7_net, \
            b8_net, b9_net, b10_net, b11_net, b12_net, b13_net, b14_net, b15_net, \
            b16_net, hr_net, last_net = load_model_feature(model[2])

            abs_normed_first_grad_sum = (abs_normed_first_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b1_grad_sum = (abs_normed_b1_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b2_grad_sum = (abs_normed_b2_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b3_grad_sum = (abs_normed_b3_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b4_grad_sum = (abs_normed_b4_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b5_grad_sum = (abs_normed_b5_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b6_grad_sum = (abs_normed_b6_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b7_grad_sum = (abs_normed_b7_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b8_grad_sum = (abs_normed_b8_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b9_grad_sum = (abs_normed_b9_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b10_grad_sum = (abs_normed_b10_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b11_grad_sum = (abs_normed_b11_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b12_grad_sum = (abs_normed_b12_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b13_grad_sum = (abs_normed_b13_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b14_grad_sum = (abs_normed_b14_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b15_grad_sum = (abs_normed_b15_grad.sum(axis=1)).sum(axis=1)
            abs_normed_b16_grad_sum = (abs_normed_b16_grad.sum(axis=1)).sum(axis=1)
            abs_normed_hr_grad_sum = (abs_normed_middle_grad.sum(axis=1)).sum(axis=1)
            abs_normed_last_grad_sum = (abs_normed_last_grad.sum(axis=1)).sum(axis=1)

            feature_dir = 'results/feature_map/' + model[0] + '/' + filename.split('.')[0]
            if os.path.exists(feature_dir):
                pass
            else:
                os.makedirs(feature_dir)

            for location in ['first', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11',
                             'b12', 'b13', 'b14', 'b15', 'b16', 'hr', 'last']:
                feature_dir_small = feature_dir + '/' + location
                if os.path.exists(feature_dir_small):
                    pass
                else:
                    os.makedirs(feature_dir_small)

            feature_dir_log = 'results/feature_log/' + model[0] + '/' + filename.split('.')[0]
            if os.path.exists(feature_dir_log):
                pass
            else:
                os.makedirs(feature_dir_log)

            for position in ['first', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11',
                             'b12', 'b13', 'b14', 'b15', 'b16', 'hr', 'last']:

                model_new = eval(position + '_net')
                sr, feature_list, pic_list, weight_list_0, weight_list_1, weight_list_2 = model_new(_add_batch_one(tensor_lr))
                channel_num = 0
                f = open(os.path.join(feature_dir_log + '/' + position + '_attribution.txt'), "a+")

                x_list = list(range(0,64))
                psnr_list = np.zeros(64)
                for feature, pic in zip(feature_list, pic_list):
                    map = 255 * (feature.cpu().detach().numpy())
                    pic_psnr = pic[0]

                    pic_psnr = Tensor2PIL(torch.clamp(pic_psnr, min=0., max=1.))
                    pic_psnr = pil_to_cv2(pic_psnr).astype(np.float64)
                    gt_psnr = Tensor2PIL(torch.clamp(tensor_hr, min=0., max=1.))
                    gt_psnr = pil_to_cv2(gt_psnr).astype(np.float64)
                    psnr = metric.calculate_psnr(pic_psnr, gt_psnr)
                    psnr_list[channel_num] = psnr

                    cv2.imwrite(feature_dir + '/' + position +  '/' + position + '_channel_' + str(channel_num) + '.png', map)
                    f.write('PSNR_' + position + '_channel_' + str(channel_num) + ': ' +
                            str(psnr) + '\r\n')
                    channel_num += 1
                f.close

                plt_dir = 'results/draw_pic/' + model[0] + '/' + filename.split('.')[0]
                if os.path.exists(plt_dir):
                    pass
                else:
                    os.makedirs(plt_dir)

                for c in ['attribution', 'psnr', 'comparison']:
                    plt_dir_small = plt_dir + '/' + c
                    if os.path.exists(plt_dir_small):
                        pass
                    else:
                        os.makedirs(plt_dir_small)

                plt.figure()
                plt.plot(x_list, -psnr_list, label="psnr_list", linestyle="--")
                plt.savefig(plt_dir + '/psnr/' + model[0] + '_' + position + '_psnr.png')
                plt.close()

                plt.figure()
                y_list = eval('abs_normed_' + position + '_grad_sum')
                plt.plot(x_list, y_list, label="attribution", linestyle="-.")
                plt.savefig(plt_dir + '/attribution/' + model[0] + '_' + position + '_attribution.png')
                plt.close()

                plt.figure()
                plt.plot(x_list, 30 * psnr_list, label="psnr_list", linestyle="--")
                y_list = eval('abs_normed_' + position + '_grad_sum')
                plt.plot(x_list, y_list, label="attribution", linestyle="-.")
                plt.savefig(plt_dir + '/comparison/' + model[0] + '_' + position + '.png')
                plt.close()


            log_dir = 'results/log_results/' + model[0] + '/' + filename.split('.')[0]
            if os.path.exists(log_dir):
                pass
            else:
                os.makedirs(log_dir)

            for i in range(abs_normed_first_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'first_attribution.txt'), "a+")
                f.write('first_channel_' + str(i) + ': ' +
                        str(abs_normed_first_grad_sum[i]) + '\r\n')
                first_grad_img = vis_saliency_nozoomin(abs_normed_first_grad[i])
                first_grad_img.save(save_dir + '/first' + '/first_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b1_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b1_attribution.txt'), "a+")
                f.write('b1_channel_' + str(i) + ': ' +
                        str(abs_normed_b1_grad_sum[i]) + '\r\n')
                b1_grad_img = vis_saliency_nozoomin(abs_normed_b1_grad[i])
                b1_grad_img.save(save_dir + '/b1' + '/b1_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b2_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b2_attribution.txt'), "a+")
                f.write('b2_channel_' + str(i) + ': ' +
                        str(abs_normed_b2_grad_sum[i]) + '\r\n')
                b2_grad_img = vis_saliency_nozoomin(abs_normed_b2_grad[i])
                b2_grad_img.save(save_dir + '/b2' + '/b2_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b3_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b3_attribution.txt'), "a+")
                f.write('b3_channel_' + str(i) + ': ' +
                        str(abs_normed_b3_grad_sum[i]) + '\r\n')
                b3_grad_img = vis_saliency_nozoomin(abs_normed_b3_grad[i])
                b3_grad_img.save(save_dir + '/b3' + '/b3_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b4_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b4_attribution.txt'), "a+")
                f.write('b4_channel_' + str(i) + ': ' +
                        str(abs_normed_b4_grad_sum[i]) + '\r\n')
                b4_grad_img = vis_saliency_nozoomin(abs_normed_b4_grad[i])
                b4_grad_img.save(save_dir + '/b4' + '/b4_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b5_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b5_attribution.txt'), "a+")
                f.write('b5_channel_' + str(i) + ': ' +
                        str(abs_normed_b5_grad_sum[i]) + '\r\n')
                b5_grad_img = vis_saliency_nozoomin(abs_normed_b5_grad[i])
                b5_grad_img.save(save_dir + '/b5' + '/b5_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b6_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b6_attribution.txt'), "a+")
                f.write('b6_channel_' + str(i) + ': ' +
                        str(abs_normed_b6_grad_sum[i]) + '\r\n')
                b6_grad_img = vis_saliency_nozoomin(abs_normed_b6_grad[i])
                b6_grad_img.save(save_dir + '/b6' + '/b6_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b7_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b7_attribution.txt'), "a+")
                f.write('b7_channel_' + str(i) + ': ' +
                        str(abs_normed_b7_grad_sum[i]) + '\r\n')
                b7_grad_img = vis_saliency_nozoomin(abs_normed_b7_grad[i])
                b7_grad_img.save(save_dir + '/b7' + '/b7_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b8_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b8_attribution.txt'), "a+")
                f.write('b8_channel_' + str(i) + ': ' +
                        str(abs_normed_b8_grad_sum[i]) + '\r\n')
                b8_grad_img = vis_saliency_nozoomin(abs_normed_b8_grad[i])
                b8_grad_img.save(save_dir + '/b8' + '/b8_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b9_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b9_attribution.txt'), "a+")
                f.write('b9_channel_' + str(i) + ': ' +
                        str(abs_normed_b9_grad_sum[i]) + '\r\n')
                b9_grad_img = vis_saliency_nozoomin(abs_normed_b9_grad[i])
                b9_grad_img.save(save_dir + '/b9' + '/b9_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b10_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b10_attribution.txt'), "a+")
                f.write('b10_channel_' + str(i) + ': ' +
                        str(abs_normed_b10_grad_sum[i]) + '\r\n')
                b10_grad_img = vis_saliency_nozoomin(abs_normed_b10_grad[i])
                b10_grad_img.save(save_dir + '/b10' + '/b10_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b11_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b11_attribution.txt'), "a+")
                f.write('b11_channel_' + str(i) + ': ' +
                        str(abs_normed_b11_grad_sum[i]) + '\r\n')
                b11_grad_img = vis_saliency_nozoomin(abs_normed_b11_grad[i])
                b11_grad_img.save(save_dir + '/b11' + '/b11_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b12_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b12_attribution.txt'), "a+")
                f.write('b12_channel_' + str(i) + ': ' +
                        str(abs_normed_b12_grad_sum[i]) + '\r\n')
                b12_grad_img = vis_saliency_nozoomin(abs_normed_b12_grad[i])
                b12_grad_img.save(save_dir + '/b12' + '/b12_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b13_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b13_attribution.txt'), "a+")
                f.write('b13_channel_' + str(i) + ': ' +
                        str(abs_normed_b13_grad_sum[i]) + '\r\n')
                b13_grad_img = vis_saliency_nozoomin(abs_normed_b13_grad[i])
                b13_grad_img.save(save_dir + '/b13' + '/b13_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b14_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b14_attribution.txt'), "a+")
                f.write('b14_channel_' + str(i) + ': ' +
                        str(abs_normed_b14_grad_sum[i]) + '\r\n')
                b14_grad_img = vis_saliency_nozoomin(abs_normed_b14_grad[i])
                b14_grad_img.save(save_dir + '/b14' + '/b14_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b15_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b15_attribution.txt'), "a+")
                f.write('b15_channel_' + str(i) + ': ' +
                        str(abs_normed_b15_grad_sum[i]) + '\r\n')
                b15_grad_img = vis_saliency_nozoomin(abs_normed_b15_grad[i])
                b15_grad_img.save(save_dir + '/b15' + '/b15_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_b16_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'b16_attribution.txt'), "a+")
                f.write('b16_channel_' + str(i) + ': ' +
                        str(abs_normed_b16_grad_sum[i]) + '\r\n')
                b16_grad_img = vis_saliency_nozoomin(abs_normed_b16_grad[i])
                b16_grad_img.save(save_dir + '/b16' + '/b16_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_middle_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'hr_attribution.txt'), "a+")
                f.write('hr_channel_' + str(i) + ': ' +
                        str(abs_normed_hr_grad_sum[i]) + '\r\n')
                middle_grad_img = vis_saliency_nozoomin(abs_normed_middle_grad[i])
                middle_grad_img.save(save_dir + '/hr' + '/hr_channel_' + str(i) + '.png')
                f.close()
            for i in range(abs_normed_last_grad.shape[0]):
                f = open(os.path.join(log_dir + '/' + 'last_attribution.txt'), "a+")
                f.write('hr_channel_' + str(i) + ': ' +
                        str(abs_normed_last_grad_sum[i]) + '\r\n')
                last_grad_img = vis_saliency_nozoomin(abs_normed_last_grad[i])
                last_grad_img.save(save_dir + '/last' + '/last_channel_' + str(i) + '.png')
                f.close()


if __name__ == '__main__':
    main()



