# DiffusionNet - a geometric autoencoder

Tensorflow implementation of

[Diffusion Nets](http://www.sciencedirect.com/science/article/pii/S1063520317300957), 
G. Mishne, U. Shaham, A. Cloninger and I. Cohen, 
Applied and Computational Harmonic Analysis, Aug. 2017.

![decoder](https://github.com/gmishne/DiffusionNet/blob/master/bubble.png "DN decoding from embedding to surface")

Files:
- diffusion_net_pretrain.ipynb - jupyter notebook demo of using DiffusionNet for 3D curve
- diffusion_net_pretrain-layer1.py - python script for Diffusion Net with 1 hidden layer architecture, evaluting various values of the cost parameters
- diffusion_net_pretrain-layer2.py - python script for Diffusion Net with 2 hidden layers architecture, evaluting various values of the cost parameters
- anomaly.ipynb - jupyter notebook demo of using DiffusionNet for anomaly detection in images
- Diffusion.py - python implementation of diffusion maps 
- autoencoder.py - tensorflow implementation of sparse autoencoders for pre-training 

---
Output of diffusion_net_pretrain-layer2.py
```
Initial encoder loss 1.05e+00
eta=0
Final encoder loss 2.85e-02
Full autoencoder denoising loss 6.15e-02
eta=1
Final encoder loss 2.49e-02
Full autoencoder denoising loss 5.75e-02
eta=10
Final encoder loss 1.95e-02
Full autoencoder denoising loss 5.11e-02
eta=100
Final encoder loss 1.57e-02
Full autoencoder denoising loss 5.08e-02
eta=1000
Final encoder loss 3.89e-02
Full autoencoder denoising loss 6.52e-02
eta=100000.0
Final encoder loss 1.37e+00
Full autoencoder denoising loss 1.22e+00
```

![encoder2](https://github.com/gmishne/DiffusionNet/blob/master/DN_enc_2layer.png "DN encoder 2 hidden layers")
![autoencoder2](https://github.com/gmishne/DiffusionNet/blob/master/DN_stack_2layer.png "DN autoencoder 2 hidden layers")

---

Output of diffusion_net_pretrain-layer1.py
```
Initial encoder loss 1.07e+00
eta=0
Final encoder loss 4.34e-02
Full autoencoder denoising loss 8.01e-02
eta=1
Final encoder loss 4.20e-02
Full autoencoder denoising loss 7.78e-02
eta=10
Final encoder loss 3.53e-02
Full autoencoder denoising loss 7.07e-02
eta=100
Final encoder loss 3.96e-02
Full autoencoder denoising loss 7.22e-02
eta=1000
Final encoder loss 9.59e-02
Full autoencoder denoising loss 1.33e-01
eta=100000
Final encoder loss 1.40e+00
Full autoencoder denoising loss 1.21e+00
```
![encoder1](https://github.com/gmishne/DiffusionNet/blob/master/DN_enc_1layer.png "DN encoder 1 hidden layer")
![autoencoder1](https://github.com/gmishne/DiffusionNet/blob/master/DN_stack_1layer.png "DN autoencoder 1 hidden layer")

