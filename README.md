# Differentiable Unrolled Alternating Direction Method of Multipliers for OneNet
## Python source code for reproducing the experiments described in the paper
[Paper (.pdf)](https://bmvc2019.org/wp-content/uploads/papers/0717-paper.pdf)\
\
Code is mostly self-explanatory via file, variable and function names; but more complex lines are commented.\
Designed to require minimal setup overhead.\
Note: This implementation is based on the [original OneNet project by Rick Chang](https://github.com/rick-chang/OneNet).\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Huge kudos to him and his co-authors.\
Note: I have added support for Instance Normalization instead of the very old Batch Normalization variant used in the\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;original source.\
Note: The code is still a bit messy. I may conduct further refactoring over time.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For a better Keras implementation with MIT license, see [video version](https://github.com/srph25/videoonenet).\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can simply use _data['frames']=1 there to fall back to the image case addressed here.

### Installing dependencies
**Installing Python 3.7.9 on Ubuntu 20.04.2 LTS:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```
**Installing CUDA 10.0:**
```bash
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo bash cuda_10.0.130_410.48_linux --override
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
**Installing cuDNN 7.6.5:**
```bash
wget http://people.cs.uchicago.edu/~kauffman/nvidia/cudnn/cudnn-10.0-linux-x64-v7.6.5.32.tgz
# if link is broken, login and download from nvidia:
# https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz
tar -xvzf cudnn-10.0-linux-x64-v7.6.5.32.tgz
sudo cp -r cuda/include/* /usr/local/cuda-10.0/include/
sudo cp -r cuda/lib64/* /usr/local/cuda-10.0/lib64/
```
**Installing Python packages with pip:**
```bash
python3.7 -m pip install h5py==2.10.0 ipython==7.16.1 keras==2.2.4 matplotlib==3.3.2 numpy==1.19.2 pillow==8.1.0 pywavelets==1.1.1 sacred==0.8.2 scikit-learn==0.23.2 scipy==1.5.2 tensorflow-gpu==1.14.0 tqdm==4.56.0
```
**Downloading and preprocessing ImageNet and MS-Celeb-1M data sets:**
```bash
# MS-Celeb-1M
# download FaceImageCroppedWithAlignment.tsv :
# https://academictorrents.com/details/9e67eb7cc23c9417f39778a8e06cca5e26196a97
python3.7 -m IPython preprocess_celeb.py
python3.7 -m IPython load_celeb.py

# ImageNet
# download imagenet_object_localization_patched2019.tar.gz :
# https://www.kaggle.com/c/imagenet-object-localization-challenge/data
tar -xvzf imagenet_object_localization_patched2019.tar.gz
python3.7 -m IPython load_imagenet.py
```

### Running the code
Reproduction should be as easy as executing this in the root folder (after installing all dependencies):
```bash
# MS-Celeb-1M
bash train_celeb_diff_unrolled_onenet.py
python3.7 -m IPython test.py --data_set celeb --n_test_images 500 --pretrained_model_file_diff_admm model/your_celeb_diff_admm

# ImageNet
bash train_imagenet_diff_unrolled_onenet.py
python3.7 -m IPython test.py --data_set imagenet --n_test_images 500 --pretrained_model_file_diff_admm model/your_imagenet_diff_admm
```

where `your_celeb_diff_admm` and `your_imagenet_diff_admm` are your results subdirectories created under the `model` directory during training.

For single GPU (8 Gb): use `--batch_size 5` and 100000 training iterations.\
For dual GPU (2*8 Gb): use `--batch_size 10 --gpus 2` and 50000 training iterations.

These should yield very similar numbers as in the table of our paper.


### Directory and file structure:
train_\*.sh : configurations used for training\
train.py : training script for the original OneNet baseline and our Differentiable Unrolled ADMM\
test.py : evaluation and image drawing script for all models (OneNet, Diff. Unr. ADMM, Wavelet l1 Sparsity)\
preprocess_\*.py : data set preparation scripts\
load_\*.py : data set loader scripts\
solver_\*.py : ADMM with various priors\
layers\*.py : custom TensorFlow layers


### Citation:
```latex
@inproceedings{milacski2019differentiable,
  title={Differentiable Unrolled Alternating Direction Method of Multipliers for OneNet.},
  author={Milacski, Zolt{\'a}n {\'A}d{\'a}m and P{\'o}czos, Barnab{\'a}s and Lorincz, Andr{\'a}s},
  booktitle={BMVC},
  pages={140},
  year={2019}
}
```


### Contact:
In case of any questions, feel free to create an issue here on GitHub, or mail me at [srph25@gmail.com](mailto:srph25@gmail.com).

