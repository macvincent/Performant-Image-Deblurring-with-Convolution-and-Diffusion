import os
import cv2
import glob
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_metrics():
    file_path = "results/test_HI_Diff_GoPro_PIDCD/visualization/GoPro/"
    gt_path = "datasets/GoPro/test/target/"

    path_list = glob.glob(file_path + '*.jpg') + glob.glob(file_path + '*.png')
    gt_list = glob.glob(gt_path + '*.jpg') + glob.glob(gt_path + '*.png')
    img_num = len(path_list)

    total_psnr = 0
    total_ssim = 0
    if img_num > 0: 
        for j in tqdm(range(1, img_num)): 
           image_name = path_list[j]
           gt_name = gt_list[j]
           input = cv2.imread(image_name)
           gt = cv2.imread(gt_name)
        #    breakpoint()
           ssim_val = ssim(input, gt, channel_axis=-1, data_range=255)
           psnr_val = psnr(input, gt)
           total_ssim = total_ssim + ssim_val
           total_psnr = total_psnr + psnr_val
    qm_psnr = total_psnr / img_num
    qm_ssim = total_ssim / img_num

    print(f'For GoPro dataset PSNR: {qm_psnr} SSIM: {qm_ssim}\n')

if __name__ == "__main__":
  calculate_metrics()