import sys
sys.path.append("../projector")
import train
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import math
import os
import timeit
import matplotlib.pyplot as plt
import add_noise
import solver_dnn as solver
import solver_wavelet_l1
#import solver_A as solver_A
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from pil import fromimage, toimage, imresize, imread, imsave
import argparse


def save_results(folder, infos, ori_img, x, z, u):
    filename = '%s/infos.mat' % folder
    sp.io.savemat(filename, infos)
    filename = '%s/x.jpg' % folder
    imsave(filename, imresize((reshape_img(np.clip(x, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
    filename = '%s/z.jpg' % folder
    imsave(filename, imresize((reshape_img(np.clip(z, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
    filename = '%s/u.jpg' % folder
    imsave(filename, imresize((reshape_img(np.clip(u, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
    
    z1 = reshape_img(np.clip(z, 0.0, 1.0)) 
    ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
    psnr = 10*np.log10( 1.0 /np.linalg.norm(z1-ori_img1)**2*np.prod(z1.shape))   
    img = Image.fromarray( imresize(np.uint8(z1*255), 4.0, interp='nearest' ) )
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='tnr.ttf', size=50)
    draw.text((135, 200), "%.2f"%psnr, (255,255,255), font=font)
    filename = '%s/z_psnr.jpg' % folder
    img.save(filename)

    fig = plt.figure()
    plt.semilogy(infos['obj_lss'][np.where(infos['obj_lss'] > 0)], 'b-', marker='o', label=r"$\frac{1}{2}\|y-Az\|_2^2$")
    plt.semilogy(infos['x_zs'][np.where(infos['x_zs'] > 0)], 'r-', marker='x', label=r"rmse of $x-z$")
    plt.xlabel('Iteration')
    plt.legend()
    fig.savefig('%s/admm.jpg' % folder)
    plt.close(fig)
    return psnr

def save_all_results(folder, list_of_infos, list_of_labels):
    cm = plt.get_cmap('gist_rainbow')
    markers = ['o', '^', 's', 'p', 'P', '*', 'H', 'X', 'D']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, infos in enumerate(list_of_infos):
        lines = ax.semilogy(infos['obj_lss'][np.where(infos['obj_lss'] > 0)], '-', marker=markers[np.mod(i, len(markers))], markevery=8, label="%s" % list_of_labels[i])
        lines[0].set_color(cm((i / len(list_of_infos))))
    plt.title('Squared Error of ADMM')
    plt.xlabel('Iteration')
    plt.ylabel(r"$\frac{1}{2}\Vert y-Az\Vert_2^2$")
    plt.legend()
    fig.savefig('%s/admm_ls_all.jpg' % folder)
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, infos in enumerate(list_of_infos):
        lines = ax.semilogy(infos['x_zs'][np.where(infos['x_zs'] > 0)], '-', marker=markers[np.mod(i, len(markers))], markevery=8, label="%s" % list_of_labels[i])
        lines[0].set_color(cm((i / len(list_of_infos))))
    plt.title('Convergence of ADMM')
    plt.xlabel('Iteration')
    plt.ylabel(r"RMSE of $x-z$")
    plt.legend()
    fig.savefig('%s/admm_x_z_all.jpg' % folder)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, infos in enumerate(list_of_infos):
        lines = ax.plot(infos['psnrs'][np.where(infos['psnrs'] > 0)], '-', marker=markers[np.mod(i, len(markers))], markevery=8, label="%s" % list_of_labels[i])
        lines[0].set_color(cm((i / len(list_of_infos))))
    plt.title('Performance of ADMM')
    plt.xlabel('Iteration')
    plt.ylabel(r"PSNR")
    plt.legend()
    fig.savefig('%s/admm_psnr_all.jpg' % folder)
    plt.close(fig)
    
def save_idxs_results(folder, list_of_infos, list_of_labels):
    cm = plt.get_cmap('gist_rainbow')
    #markers = ['o', '^', 's', 'p', 'P', '*', 'H', 'X', 'D']
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, _infos in enumerate(list_of_infos):
        for j, infos in enumerate(_infos):
            if i == 0:
                lines = ax.semilogy(infos['obj_lss'][np.where(infos['obj_lss'] > 0)], '-', alpha=0.1,#marker=markers[np.mod(i, len(markers))], markevery=8, 
                                    label="%s" % list_of_labels[j])
            else:
                lines = ax.semilogy(infos['obj_lss'][np.where(infos['obj_lss'] > 0)], '-', alpha=0.1)
            lines[0].set_color(cm((j / len(_infos))))
    plt.title('Squared Error of ADMM')
    plt.xlabel('Iteration')
    plt.ylabel(r"$\frac{1}{2}\Vert y-Az\Vert_2^2$")
    plt.legend()
    fig.savefig('%s/admm_ls_idxs.jpg' % folder)
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, _infos in enumerate(list_of_infos):
        for j, infos in enumerate(_infos):
            if i == 0:
                lines = ax.semilogy(infos['x_zs'][np.where(infos['x_zs'] > 0)], '-', alpha=0.1,#marker=markers[np.mod(i, len(markers))], markevery=8, 
                                    label="%s" % list_of_labels[j])
            else:
                lines = ax.semilogy(infos['x_zs'][np.where(infos['x_zs'] > 0)], '-', alpha=0.1)
            lines[0].set_color(cm((j / len(_infos))))
    plt.title('Convergence of ADMM')
    plt.xlabel('Iteration')
    plt.ylabel(r"RMSE of $x-z$")
    plt.legend()
    fig.savefig('%s/admm_x_z_idxs.jpg' % folder)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, _infos in enumerate(list_of_infos):
        for j, infos in enumerate(_infos):
            if i == 0:
                lines = ax.plot(infos['psnrs'][np.where(infos['psnrs'] > 0)], '-', alpha=0.1,#marker=markers[np.mod(i, len(markers))], markevery=8, 
                                label="%s" % list_of_labels[j])
            else:
                lines = ax.plot(infos['psnrs'][np.where(infos['psnrs'] > 0)], '-', alpha=0.1)
            lines[0].set_color(cm((j / len(_infos))))
    plt.title('Performance of ADMM')
    plt.xlabel('Iteration')
    plt.ylabel(r"PSNR")
    plt.legend()
    fig.savefig('%s/admm_psnr_idxs.jpg' % folder)
    plt.close(fig)


parser = argparse.ArgumentParser()
parser.add_argument('--data_set', default='imagenet', help='which dataset to use, imagenet or celeb')
parser.add_argument('--n_test_images', type=int, default=5, help='number of images used for testing')
parser.add_argument('--pretrained_model_file_baseline', default=None, help='Pretrained weights for OneNet baseline')
parser.add_argument('--pretrained_model_file_diff_admm', default=None, help='Pretrained weights for our Differentiable ADMM')
parser.add_argument('--run_wavelet_l1', type=int, default=True, help='Whether to run Wavelet l1 Sparsity baseline')

opt = parser.parse_args()
print(opt)

data_set = opt.data_set
n_test_images = opt.n_test_images
pretrained_model_file_baseline = opt.pretrained_model_file_baseline
pretrained_model_file_diff_admm = opt.pretrained_model_file_diff_admm
run_wavelet_l1 = opt.run_wavelet_l1

if data_set not in ['imagenet', 'celeb']:
    raise NotImplementedError
if data_set == 'imagenet':
    print('Loading ImageNet data set...')
    import load_imagenet as load_dataset
    if pretrained_model_file_baseline is None:
        pretrained_model_file_baseline = None
        run_baseline = False
    if pretrained_model_file_diff_admm is None:
        pretrained_model_file_diff_admm = "model/20210220133545_57288_srph25-desktop_imsize64_ratio0.010000_dis0.005000_latent0.000100_img0.001000_de1.000000_derate1.000000_dp0_gd1_softpos0.850000_wdcy_0.000000_seed0/model/model_iter-23999"
        run_diff_admm = True
elif data_set == 'celeb':
    print('Loading MS-Celeb-1M data set...')
    import load_celeb as load_dataset
    if pretrained_model_file_baseline is None:
        pretrained_model_file_baseline = None
        run_baseline = False
    if pretrained_model_file_diff_admm is None:
        pretrained_model_file_diff_admm = "model/20210221082343_111020_srph25-desktop_imsize64_ratio0.010000_dis0.005000_latent0.000100_img0.001000_de1.000000_derate1.000000_dp0_gd1_softpos0.850000_wdcy_0.000000_seed0/model/model_iter-33999"
        run_diff_admm = True


# index of test images
idxs = np.array(list(range(n_test_images)))

# result folder
clean_paper_results = 'clean_paper_results' 

# filename of the trained model. If using virtual batch normalization, 
# the popmean and popvariance need to be updated first via update_popmean.py!

psnr_inpaint_denoise = []
psnr_inpaint_center = []
psnr_inpaint_block = []
psnr_superres = []
infos_inpaint_denoise = []
infos_inpaint_center = []
infos_inpaint_block = []
infos_superres = []
for idx in idxs :
    print('idx = %d --------' % idx)

    np.random.seed(idx)

    img_size = (64,64,3)

    show_img_progress = False # whether the plot intermediate results (may slow down the process)

    def load_image(filepath):
        img = imread(filepath)
        img = imresize(img, [64,64]).astype(float) / 255.0
        if len(img.shape) < 3:
            img = np.tile(img, [1,1,3])
        return img
    
    
    def solve_inpaint_denoise(ori_img, get_denoiser, reshape_img_fun, drop_prob=0.3,
                                 noise_mean=0, noise_std=0.1,
                                 alpha=0.3, lambda_wavelet_l1=0.1, max_iter=100, solver_tol=1e-2):
        import inpaint as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, mask, A) = problem.setup(x_shape, drop_prob=drop_prob)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if show_img_progress:
            fig = plt.figure('inpaint_denoise')
            plt.gcf().clear()
            fig.canvas.set_window_title('inpaint_denoise')
            plt.subplot(1,2,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,2,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.pause(0.00001)

        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'mask': mask, 'drop_prob': drop_prob, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_wavelet_l1': lambda_wavelet_l1}

        # save the problem
        base_folder = '%s/inpaint-denoise_ratio%.2f_std%.2f' % (result_folder, drop_prob, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

        infos_all = []
        labels_all = []
        if run_baseline:
            # ours
            folder = '%s/baseline_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=False)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol, ori_img=ori_img)
            psnr_baseline = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Original OneNet')
        else:
            psnr_baseline = np.nan
        if run_diff_admm:
            # ours
            folder = '%s/diff_admm_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=True)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter_diff_admm, solver_tol=solver_tol, ori_img=ori_img)
            psnr_diff_admm = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Differentiable Unrolled OneNet')
        else:
            psnr_diff_admm = np.nan
        if run_wavelet_l1:
            # wavelet l1
            folder = '%s/wavelet_l1_lambda%f_alpha%f' % (base_folder, lambda_wavelet_l1, alpha_wavelet_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_wavelet_l1.solve(y, A_fun, AT_fun, lambda_wavelet_l1, reshape_img_fun, folder,
                                                       show_img_progress=show_img_progress, alpha=alpha_wavelet_l1,
                                                       max_iter=max_iter_wavelet_l1, solver_tol=solver_tol_wavelet_l1, ori_img=ori_img)
            psnr_wavelet_l1 = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('wavelet_l1')
        else:
            psnr_wavelet_l1 = np.nan

        save_all_results(base_folder, infos_all, labels_all)
        infos_inpaint_denoise.append(infos_all)
        psnr_inpaint_denoise.append([psnr_baseline, psnr_diff_admm, psnr_wavelet_l1])
        if idx == idxs[-1]:
            save_idxs_results(clean_paper_results, infos_inpaint_denoise, labels_all)

    def solve_inpaint_center(ori_img, get_denoiser, reshape_img_fun, box_size=1,
                            noise_mean=0, noise_std=0.,
                            alpha=0.3, lambda_wavelet_l1=0.1, max_iter=100, solver_tol=1e-2):
        import inpaint_center as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, mask, A) = problem.setup(x_shape, box_size=box_size)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if show_img_progress:
            fig = plt.figure('inpaint_center')
            plt.gcf().clear()
            fig.canvas.set_window_title('inpaint_center')
            plt.subplot(1,2,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,2,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.pause(0.00001)

        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'mask': mask, 'box_size': box_size, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_wavelet_l1': lambda_wavelet_l1}

        # save the problem
        base_folder = '%s/inpaintcenter_bs%d_std%.2f' % (result_folder, box_size, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

        infos_all = []
        labels_all = []
        if run_baseline:
            # ours
            folder = '%s/baseline_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=False)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol, ori_img=ori_img)
            psnr_baseline = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Original OneNet')
        else:
            psnr_baseline = np.nan
        if run_diff_admm:
            # ours
            folder = '%s/diff_admm_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=True)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter_diff_admm, solver_tol=solver_tol, ori_img=ori_img)
            psnr_diff_admm = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Differentiable Unrolled OneNet')
        else:
            psnr_diff_admm = np.nan
        if run_wavelet_l1:
            # wavelet l1
            folder = '%s/wavelet_l1_lambda%f_alpha%f' % (base_folder, lambda_wavelet_l1, alpha_wavelet_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_wavelet_l1.solve(y, A_fun, AT_fun, lambda_wavelet_l1, reshape_img_fun, folder,
                                                       show_img_progress=show_img_progress, alpha=alpha_wavelet_l1,
                                                       max_iter=max_iter_wavelet_l1, solver_tol=solver_tol_wavelet_l1, ori_img=ori_img)
            psnr_wavelet_l1 = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('wavelet_l1')
        else:
            psnr_wavelet_l1 = np.nan

        save_all_results(base_folder, infos_all, labels_all)
        infos_inpaint_center.append(infos_all)
        psnr_inpaint_center.append([psnr_baseline, psnr_diff_admm, psnr_wavelet_l1])
        if idx == idxs[-1]:
            save_idxs_results(clean_paper_results, infos_inpaint_center, labels_all)

    def solve_inpaint_block(ori_img, get_denoiser, reshape_img_fun, box_size=1, total_box=1,
                            noise_mean=0, noise_std=0.,
                            alpha=0.3, lambda_wavelet_l1=0.1, max_iter=100, solver_tol=1e-2):
        import inpaint_block as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, mask, A) = problem.setup(x_shape, box_size=box_size, total_box=total_box)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if show_img_progress:
            fig = plt.figure('inpaint_block')
            plt.gcf().clear()
            fig.canvas.set_window_title('inpaint_block')
            plt.subplot(1,2,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,2,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.pause(0.00001)


        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'mask': mask, 'box_size': box_size,
                'total_box': total_box, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_wavelet_l1': lambda_wavelet_l1}


        # save the problem
        base_folder = '%s/inpaint_bs%d_tb%d_std%.2f' % (result_folder, box_size, total_box, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

        infos_all = []
        labels_all = []
        if run_baseline:
            # ours
            folder = '%s/baseline_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=False)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol, ori_img=ori_img)
            psnr_baseline = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Original OneNet')
        else:
            psnr_baseline = np.nan
        if run_diff_admm:
            # ours
            folder = '%s/diff_admm_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=True)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter_diff_admm, solver_tol=solver_tol, ori_img=ori_img)
            psnr_diff_admm = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Differentiable Unrolled OneNet')
        else:
            psnr_diff_admm = np.nan
        if run_wavelet_l1:
            # wavelet l1
            folder = '%s/wavelet_l1_lambda%f_alpha%f' % (base_folder, lambda_wavelet_l1, alpha_wavelet_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_wavelet_l1.solve(y, A_fun, AT_fun, lambda_wavelet_l1, reshape_img_fun, folder,
                                                       show_img_progress=show_img_progress, alpha=alpha_wavelet_l1,
                                                       max_iter=max_iter_wavelet_l1, solver_tol=solver_tol_wavelet_l1, ori_img=ori_img)
            psnr_wavelet_l1 = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('wavelet_l1')
        else:
            psnr_wavelet_l1 = np.nan

        save_all_results(base_folder, infos_all, labels_all)
        infos_inpaint_block.append(infos_all)
        psnr_inpaint_block.append([psnr_baseline, psnr_diff_admm, psnr_wavelet_l1])
        if idx == idxs[-1]:
            save_idxs_results(clean_paper_results, infos_inpaint_block, labels_all)

    def solve_superres(ori_img, get_denoiser, reshape_img_fun, resize_ratio=0.5,
                       noise_mean=0, noise_std=0.,
                       alpha=0.3, lambda_wavelet_l1=0.1, max_iter=100, solver_tol=1e-2):
        import superres as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, A) = problem.setup(x_shape, resize_ratio=resize_ratio)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        bicubic_img = imresize(y[0], [ori_img.shape[1], ori_img.shape[2]], interp='bicubic')
        if show_img_progress:
            fig = plt.figure('superres')
            plt.gcf().clear()
            fig.canvas.set_window_title('superres')
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.subplot(1,3,3)
            plt.imshow(np.clip(bicubic_img,0,255), interpolation='nearest')
            plt.title('bicubic')
            plt.pause(0.00001)

        bicubic_img = bicubic_img.astype(float) / 255.0
        l2_dis = np.mean(np.square(ori_img[0] - bicubic_img))

        print('bicubic err = %f' % (l2_dis))


        info = {'ori_img': ori_img, 'y': y, 'bicubic': bicubic_img, 'noise': noise, 'resize_ratio': resize_ratio,
                'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_wavelet_l1': lambda_wavelet_l1}

        # save the problem
        base_folder = '%s/superres_ratio%.2f_std%.2f' % (result_folder, resize_ratio, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        imsave(filename, imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/bicubic_img.jpg' % base_folder
        imsave(filename, imresize((bicubic_img*255).astype(np.uint8), 4.0, interp='nearest'))

        infos_all = []
        labels_all = []
        if run_baseline:
            # ours
            folder = '%s/baseline_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=False)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol, ori_img=ori_img)
            psnr_baseline = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Original OneNet')
        else:
            psnr_baseline = np.nan
        if run_diff_admm:
            # ours
            folder = '%s/diff_admm_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            denoiser = get_denoiser(use_diff_admm=True)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter_diff_admm, solver_tol=solver_tol, ori_img=ori_img)
            psnr_diff_admm = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('Differentiable Unrolled OneNet')
        else:
            psnr_diff_admm = np.nan
        if run_wavelet_l1:
            # wavelet l1
            folder = '%s/wavelet_l1_lambda%f_alpha%f' % (base_folder, lambda_wavelet_l1, alpha_wavelet_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_wavelet_l1.solve(y, A_fun, AT_fun, lambda_wavelet_l1, reshape_img_fun, folder,
                                                       show_img_progress=show_img_progress, alpha=alpha_wavelet_l1,
                                                       max_iter=max_iter_wavelet_l1, solver_tol=solver_tol_wavelet_l1, ori_img=ori_img)
            psnr_wavelet_l1 = save_results(folder, infos, ori_img, x, z, u)
            infos_all.append(infos)
            labels_all.append('wavelet_l1')
        else:
            psnr_wavelet_l1 = np.nan
            
        save_all_results(base_folder, infos_all, labels_all)
        infos_superres.append(infos_all)
        psnr_superres.append([psnr_baseline, psnr_diff_admm, psnr_wavelet_l1])
        if idx == idxs[-1]:
            save_idxs_results(clean_paper_results, infos_superres, labels_all)

            

    def reshape_img(img):
        return img[0]

    # load the dataset

    print('loading data...')
    testset_filelist = load_dataset.load_testset_path_list()
    total_test = len(testset_filelist)
    print('total test = %d' % total_test)

    # We create a session to use the graph and restore the variables
    def get_denoiser(use_diff_admm=False):
        print(use_diff_admm)
        global _SESSION
        global _GRAPH_LEARNING_PHASES
        tf.reset_default_graph()
        global _GRAPH_UID_DICTS
        _GRAPH_UID_DICTS = {}
        _SESSION = None
        _GRAPH_LEARNING_PHASES = {}
        # setup the variables in the session
        n_reference = 0
        batch_size = n_reference + 1
        images_tf = tf.placeholder( tf.float32, [batch_size, img_size[0], img_size[1], img_size[2]], name="images")
        is_train = False
        if use_diff_admm is False:
            use_instance_norm = True
            use_elu_like = False
            use_custom_image_resize = False
            proj, latent = train.build_projection_model(images_tf, is_train, n_reference, use_bias=True, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, reuse=None)
        else:
            use_instance_norm = True
            use_elu_like = True
            use_custom_image_resize = True
            with tf.variable_scope('fp32_storage'):
                proj, latent = train.build_projection_model(images_tf, is_train, n_reference, use_bias=True, use_instance_norm=use_instance_norm, use_elu_like=use_elu_like, use_custom_image_resize=use_custom_image_resize, reuse=None)

        with tf.variable_scope("PROJ") as scope:
            scope.reuse_variables()

        print('loading model...')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, pretrained_model_file if use_diff_admm is False else pretrained_model_file_diff_admm)
        #print(sess.run(tf.global_variables()))
        print('finished reload.')

        # define denoiser
        def denoise(x):
            x_shape = x.shape
            x = np.reshape(x, [1, img_size[0], img_size[1], img_size[2]], order='F')
            x = (x - 0.5) * 2.0

            y = sess.run(proj, feed_dict={images_tf: x})
            y = (y / 2.0) + 0.5
            return np.reshape(y, x_shape)
        return denoise
    

    print(len(testset_filelist), idx)
    img = load_image(testset_filelist[idx])

    ori_img = np.reshape(img, [1, img_size[0],img_size[1],img_size[2]], order='F')

    result_folder = '%s/%d' % (clean_paper_results,idx)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)


    filename = '%s/ori_img.jpg' % result_folder
    imsave(filename, imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))


    ##############################################################################################
    ##### super resolution
    print('super resolution')

    #set parameters
    alpha = 0.5 # 1.0
    max_iter = 300
    max_iter_diff_admm = 8
    solver_tol = 2e-3

    alpha_wavelet_l1 = 0.3
    lambda_wavelet_l1 = 0.05
    max_iter_wavelet_l1 = 100#0
    solver_tol_wavelet_l1 = 1e-4

    resize_ratio = 0.5
    noise_std = 0.0

    results = solve_superres(ori_img, get_denoiser, reshape_img, resize_ratio=resize_ratio,
                             noise_std=noise_std,
                             alpha=alpha, lambda_wavelet_l1=lambda_wavelet_l1, max_iter=max_iter, solver_tol=solver_tol)

    ############################################################################################
    #### denoising

    print('denoising')

    # set parameter
    alpha = 0.3
    max_iter = 300
    max_iter_diff_admm = 8
    solver_tol = 3e-3

    alpha_wavelet_l1 = 0.3
    lambda_wavelet_l1 = 0.05
    max_iter_wavelet_l1 = 100#0
    solver_tol_wavelet_l1 = 1e-4

    drop_prob = 0.5
    noise_std = 0.1

    results = solve_inpaint_denoise(ori_img, get_denoiser, reshape_img, drop_prob=drop_prob,
                                    noise_mean=0, noise_std=noise_std,
                                    alpha=alpha, lambda_wavelet_l1=lambda_wavelet_l1, max_iter=max_iter, solver_tol=solver_tol)

    ##########################################################################################
    ## inpaint block

    print('inpaint block')

    # set parameter
    alpha = 0.3
    max_iter = 300
    max_iter_diff_admm = 8
    solver_tol = 1e-4

    alpha_wavelet_l1 = 0.3
    lambda_wavelet_l1 = 0.03
    max_iter_wavelet_l1 = 100#0
    solver_tol_wavelet_l1 = 1e-4

    box_size = int(0.1 * ori_img.shape[1])
    noise_std = 0.0
    total_box = 10

    results = solve_inpaint_block(ori_img, get_denoiser, reshape_img, box_size=box_size, total_box=total_box,
                                  noise_std=noise_std,
                                  alpha=alpha, lambda_wavelet_l1=lambda_wavelet_l1, max_iter=max_iter, solver_tol=solver_tol)
    
    ############################################################################################
    ### inpaint center
    print('inpaint center')
   
    alpha = 0.2 
    max_iter = 300
    max_iter_diff_admm = 8
    solver_tol = 1e-5
   
    alpha_wavelet_l1 = 0.3
    lambda_wavelet_l1 = 0.05
    max_iter_wavelet_l1 = 100#0
    solver_tol_wavelet_l1 = 1e-4
   
    box_size = int(0.3 * ori_img.shape[1])
    noise_std = 0.0

    results = solve_inpaint_center(ori_img, get_denoiser, reshape_img, box_size=box_size,
                                   noise_std=noise_std,
                                   alpha=alpha, lambda_wavelet_l1=lambda_wavelet_l1, max_iter=max_iter, solver_tol=solver_tol)

    ############################################################################################
    ### inpaint center
    if run_baseline or run_diff_admm:
        tf.reset_default_graph()

psnr_inpaint_denoise = np.array(psnr_inpaint_denoise)
psnr_inpaint_center = np.array(psnr_inpaint_center)
psnr_inpaint_block = np.array(psnr_inpaint_block)
psnr_superres = np.array(psnr_superres)
for j, method in enumerate(['OneNet (Baseline)', 'Differentiable ADMM (ours)', 'Wavelet l_1 Sparsity']):
    print(method, ':')
    print('    PSNR Inpaint:', np.mean(psnr_inpaint_denoise[:, j]), '+-', np.std(psnr_inpaint_denoise[:, j]))
    print('    PSNR Inpaint Center:', np.mean(psnr_inpaint_center[:, j]), '+-', np.std(psnr_inpaint_center[:, j]))
    print('    PSNR Inpaint Scattered:', np.mean(psnr_inpaint_block[:, j]), '+-', np.std(psnr_inpaint_block[:, j]))
    print('    PSNR Super Resolution:', np.mean(psnr_superres[:, j]), '+-', np.std(psnr_superres[:, j]))

