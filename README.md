# TomoGAN

Paper: [arXiv:1902.07582](https://arxiv.org/abs/1902.07582); [arXiv:1910.04081](https://arxiv.org/abs/1910.04081) Relevant [presentation](https://lzhengchun.github.io/file/pse-ai-townhall-TomoGAN-Zhengchun-Liu.pdf) 

An image quality enhancement model based on generative adversarial networks originally developed for synchrotron X-Ray tomography. It has also been used for other case like streaming tomography, the joint ptycho-tomography problem([arXiv:2009.09498](https://arxiv.org/abs/2009.09498)). We also developed a workflow to deploy TomoGAN on Coral edge TPU ([arXiv:1911.05878](https://arxiv.org/abs/1911.05878)) which can process an image with 1k x 1k pixels in 500ms. 

I also implemented the generator model based on pure Intel DNNL (MKL-DNN) and NVIDIA cudnn seperately for inference, using C++, so that you can easily integrate it in your C++ project. I will open source them soon.

To give it a try:

* download sample dataset (only need to download _dataset/demo-dataset-real.h5_) and the ImageNet pretrained vgg model from [Here](https://anl.box.com/s/h6koi0hhwqrj1c9tt82tldzo45tl3x15)
* install dependencies, Tensorflow(preferably 1.13 for TF1 version, 2.2 for TF2 version)
* run with python ./main-gan.py -gpu=0 -expName=test -dsfn=dataset/demo-dataset-real.h5

Train a model with your own dataset

* one of the code-free way is to prepare a h5 file with four datasets as the sample dataset demo-dataset-real.h5. for datasets in the h5 should be named as: _train_ns_ and *train_gt* for training; *test_ns* and *test_gt* for validation, each dataset should have a dimension of (N, H, W) where N is the number of images.
* Best practice: always good to make sure that there is no outliers (e.g., due to bad pixels in detector) in your data.

__Note__: with the sample dataset (128 images for training) provided here, you may not reproduce our results as it reported in  [arXiv:1910.04081](https://arxiv.org/abs/1910.04081) because we used a much larger dataset that has 2048 images. 
You may download the X-ray projection from [TomoBank](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.spheres.html) and make a reconstruction using all the projections as ground truth and a subsampled projections (every 32) as (low dose) noisy input. The dataset we provided, both norml dose (all projections) and low-dose (1/32 subsampled), were reconstructed using SIRT algorithms with 100 iterastions. 

## Citation
If you use this code for your research, please cite our paper(s):

```
@inproceedings{liu2019deep,
    title={Deep Learning Accelerated Light Source Experiments},
    author={Zhengchun Liu and Tekin Bicer and Rajkumar Kettimuthu and Ian Foster},
    year={2019},
    booktitle={2019 IEEE/ACM Third Workshop on Deep Learning on Supercomputers (DLS)}
}

@article{liu2020tomogan,
  title={TomoGAN: low-dose synchrotron x-ray tomography with generative adversarial networks: discussion},
  author={Liu, Zhengchun and Bicer, Tekin and Kettimuthu, Rajkumar and Gursoy, Doga and De Carlo, Francesco and Foster, Ian},
  journal={JOSA A},
  volume={37},
  number={3},
  pages={422--434},
  year={2020},
  publisher={Optical Society of America}
}

```

# Example
As an example / teaser, a pair of noisy input and its denoised output are shown as follows,

An example __input__ (a low-dose X-ray tomographic image ):

![Noisy Image](repo-image/ns-w016-i10-r25-s0364.png)

The corresponding __output__ (i.e., denoised using TomoGAN):

![Denoisied Image](repo-image/dn-w016-i10-r25-s0364.png)

Video clips

[![](http://img.youtube.com/vi/PcHsOK4qwlk/0.jpg)](http://www.youtube.com/watch?v=PcHsOK4qwlk "Comparison ")