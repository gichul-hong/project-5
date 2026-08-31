# 삼성 DS Project description Day 3

## Slide 1
- 삼성 DS2과정 프로젝트
- Image Restoration Challenge
- Jongho Lee, Ph.D
- Laboratory for Imaging Science and Technology
- Department of Electrical and Computer Engineering
- Seoul National University

---

## Slide 2
- Physics informed neural network

---

## Slide 3

---

## Slide 4

---

## Slide 5

---

## Slide 6

---

## Slide 7
- Why do we care?

---

## Slide 8
- Generalization issues in deep learning

---

## Slide 9
- Suscep map
- Local field map
- Deconvolution
- =
- Convolution
- Dipole
- Local field distr
- -1
- =
- Dipole
- Suscep distr
- Susceptibility map is generated from local field map by dipole deconvolution
- Fourier transform suggests dipole deconvolution requires division by zeros
- Dipole deconvolution

---

## Slide 10
- Multiple head orientation data
- Remove zeros using three or more orientation data (COSMOS)
- Single head orientation data
- k-space threshold (TKD), regularization using magnitude edge information (MEDI) and others (iLSQR, HEIDI, ….)
- Tian Liu, MRM 2009
- Sam Wharton, MRM 2010
- Ludovic de Rochefort MRM 2008
- Karin Shmueli, MRM 2009
- Tian Liu, MRM 2011
- Christian Langkammer, NeuroImage 2012
- Wei Liu, NeuroImage 2015
- Gold standard
- Not practical
- Dipole deconvolution

---

## Slide 11
- Dipole deconvolution using deep learning
- Neural network shown success in inverse problem when expensive gold standard label is available	e.g., parallel imaging with full sample data, low dose CT with full dose data
- In QSM: for training, expensive COSMOS as label paired with single orientation local field as input
- for inference, single orientation local field as input, generating COSMOS-quality QSM map
- McCann, IEEE SPM 2017
- Wang, Nature MI 2020
- high cost data
- QSMnet        Yoon, NeuroImage 2018
- DeepQSM      Bollmann, NeuroImage 2019

---

## Slide 12
- QSMnet
- COSMOS
- TKD
- MEDI
- Yoon, NeuroImage 2018
- 3D U-net
- Streaking free maps
- Computational efficiency (few secs)
- COSMOS-trained or simulation-trained
- Dipole deconvolution using deep learning
- Jung, NMR BioMed 2020

---

## Slide 13
- Dipole deconvolution using deep learning
- QSMGAN: GAN improves quality
- QSM outside of brain with fat
- AutoQSM: No need for background removal
- or brain mask
- Wei, NeuroImage 2019
- Chen, NeuroImage 2020
- Hanspach, MRM2022
- Recovering SWI-filtered phase data
- Kames, MRM 2022
- SWI-filtered phase            Recovered                     Original

---

## Slide 14
- Dipole deconvolution using deep learning
- FINE: Self transfer learning with physical model loss
- VaNDI: Combine nonlinear dipole inversion
- with variational network
- MoDL-QSM: Physics model enforced with
- network as regularizer
- Generalization to untrained input
- Zhang, Neuroimage 2020
- Polak,
- NMR Biomed
- 2020
- xQSM: Octave conv and noise-regularization
- Generalization for resolution
- Gao, NMR Biomed 2021
- Feng, Neuroimage 2021

---

## Slide 15
- Issues of “generalization”
- Issues of “performance comparison”
- Issues of “performance”
- Issues with deep learning dipole deconvolution

---

## Slide 16
- Issues of “generalization”
- Issues of “performance comparison”
- Issues of “performance”
- Issues with deep learning dipole deconvolution

---

## Slide 17
- • QSMnet
- • QSMnet+
- Training data distribution of QSMnet
- Susceptibility value in lesion
- Conventional QSM (ppm)
- QSMnet (ppm)
- Number of voxels
- Susceptibility value (ppm)
- Hemorrhage
- Jung, NeuroImage 2020
- Generalization: Susceptibility range
- QSMnet was trained with limited susceptibility range (-0.4 to 0.6 ppm)
- Hemorrhage with high susceptibility (>1 ppm) was underestimated in QSMnet
- Augmentation using simulated data improves hemorrhage estimation in QSMnet+

---

## Slide 18
- Jung, NMR BioMed 2020
- Lower resolution
- Higher resolution
- Generalization: Resolution

---

## Slide 19
- Oh, Med Img Anal 2022
- Generalization: Resolution
- Gao, NMR Biomed 2021
- Unsuprevised learning with adaptive instance normalization

---

## Slide 20
- Generalization: Resolution
- Sooyeon Ji
- QSM workshop 2022
- 1 x 1 x 3 mm3 data inference for network trained using 1.5 mm iso

---

## Slide 21
> **Notes**: Then, the linearity was improved in the new version of QSMnet, named as QSMnet+
This work was presented on last Monday with the number of abstract 317.

- Performance of deep learning QSM changes if test data has different characteristics from training data
- Høy, ISMRM2019
- Jung, NMR BioMed 2020
- Generalization

---

## Slide 22
- Generalization: Solutions using network architectures
- FINE: Self transfer learning with physical model loss
- VaNDI: Combine nonlinear dipole inversion with
- variational network
- MoDL-QSM: Physics model enforced with
- network as regularizer
- Generalization to untrained input
- Feng, Neuroimage 2021
- Zhang, Neuroimage 2020
- Polak, NMR Biomed 2020
- Networks need to be explored for generalization capability

---

## Slide 23
- Issues of “generalization”
- Issues of “performance comparison”
- Issues of “performance”
- Issues with deep learning dipole deconvolution

---

## Slide 24
> **Notes**: First, as we explained previous slide,

The training data of QSMnet consists of healthy subjects.
The training data distribution of QSMnet is plotted as shown here.
The x-axis is the susceptibility value of the voxel in training images, and y-axis is the number of voxels.

Since the network never observed the large susceptibility values such as hemorrhage lesions, or calcification during training stage,
we don’t know the network can perform properly in this range.

- Performance comparison
- No large size "public" dataset available for QSM
- Comparison via "private" data of different characteristics (resolution, background removal ...)
- Chungseok Oh
- Comparison with pre-trained net not fair due to data characteristic difference affecting performance
- Comparison with re-trained net not fair because retrain net is no longer the same as original net

---

## Slide 25
> **Notes**: First, as we explained previous slide,

The training data of QSMnet consists of healthy subjects.
The training data distribution of QSMnet is plotted as shown here.
The x-axis is the susceptibility value of the voxel in training images, and y-axis is the number of voxels.

Since the network never observed the large susceptibility values such as hemorrhage lesions, or calcification during training stage,
we don’t know the network can perform properly in this range.

- Performance comparison - reproducibility
- Deep neural network has shown limited reproducibility
- For fair comparison, use of public dataset and sharing pretrained network is recommended
- Renard, Sci Rep 2020        Alahmari, IEEE Acess 2020
- Oh, ISMRM 2022
- QSMnet and variational network
- Metrics: SSIM, RSME
- For training, CUDA convolution bench marking and determinism, mini-batch order, and network weight initiation affect results
- For inference, mostly reproduciblity

---

## Slide 26
- Issues of “generalization”
- Issues of “performance comparison”
- Issues of “performance”
- Issues with deep learning dipole deconvolution

---

## Slide 27
- Performance
- Meaning of better performance?
- Still problem of generalization? (7T? unbalanced training for high iron concentration?)
- Can we design network for improved performance for deep gray matter iron quantification?
- Yao, NI clinical available online
- Deep learning shown high quality results but conventional recon outperformed in challenge 2.0
- QSM Challenge, MRM 2021
- Redefine performance matrix (not only SSIM but "functional" performance)

---

## Slide 28
- Generalization vs. Harmonization

---

## Slide 29
- Generalization vs. Harmonization
- Siemens
- Philips
- Generalization:
- Develop neural network that can process various conditions of input
- (e.g., different vendor images)
- Harmonization:
- Process input image to have similar characteristics to training data
- Cons and pros:
- Need more data or new design
- May generalize for unknown domain
- Takes longer time for training
- *No need to worry if network is used for the same characteristic dataset as train data
- How:
- Preprocessing using neural network
- How:
- Data augmentation, network architecture
- Cons and pros:
- Need to design new network (require data)
- Takes longer time for inference

---

## Slide 30
- Siemens
- Philips

---

## Slide 31
- Image harmonization: Style transfer
- 0.43 x 0.43 x 1.5 mm3
- If paired data set available: Supervised learning
- Only unpaired data available: Style transfer via cycleGAN
- Combine both for limited paired data
- Isola, CVPR 2017
- CycleGAN
- Siemens
- (unpaired)
- Philips style image
- Philips
- Philips style
- or not
- fake
- real
- Siemens
- (pair A)
- Philips style image
- Siemens
- (pair A)
- Philips
- (pair A)
- Siemens
- (pair A)
- Philips style
- or not
- fake
- real
- CycleGan with paired data

---

## Slide 32
- Image harmonization: Style transfer
- 0.43 x 0.43 x 1.5 mm3
- Joonhyeok Yoon
- ISMRM 2022

---
