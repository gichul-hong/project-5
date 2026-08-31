# 삼성 DS Project description Day 2

## Slide 1
- 삼성 DS2과정 프로젝트
- Image Restoration Challenge
- Jongho Lee, Ph.D
- Laboratory for Imaging Science and Technology
- Department of Electrical and Computer Engineering
- Seoul National University

---

## Slide 2
- 2
- 제1회
- 삼성DS2과정 Digital Image Processing Challenge
- Semiconductor
- Image Processing
- Challenge
- 삼성DS2과정생
- To be announced

---

## Slide 3
- 3
- Aim of our challenge

---

## Slide 4
- Basic model for degradation
- Degradation
- PSF: h(x,y)
- (e.g blur)
- Restoration
- filter: r(x,y)
- 4

---

## Slide 5
- blur
- What happens when we also have degradation?
- 5

---

## Slide 6
- What happens when we also have degradation?
- S = guess area
- Manual guess
- 6

---

## Slide 7
- Inverse filtering
- 7

---

## Slide 8
- Wiener filter: minimum mean-square error filtering
- Wiener filter
- Tunable parameter
- 8

---

## Slide 9
- 9
- Aim of our challenge

---

## Slide 10
> **Notes**: QSM 의 리컨은 MRI로부터 측정한 Phase map 에서 시작합니다. Phase map 은 Susceptibility distribution 과 dipole 간의 convolution 의 결과로 나오게 되어 Susceptibility distribution 은 deconvolution 의 과정으로 구할 수 있습니다. 이 문제를 Fourier space로 가져오면 convolution 이 곱셈이 되기 때문에 Susceptibility distribution 은 phase map 의 fourier transform 을 dipole 패턴의 fourier transform 으로 나눔으로써 구할 수 있습니다. 다이폴 패턴의 fourier transform 은 zero-cone 형태를 갖는데, 보시는 바와 같이 zero-cone 의 표면에서는 0의 값을 가지게 됩니다. 따라서 이 지점에서 나눗셈은 정의가 되지 않기 때문에 QSM 을 reconstruction 하는 과정은 ill-posed problem 이 됩니다.

- 10
- Suscep map
- Local field map
- Deconvolution
- =
- Convolution
- Kernel
- Local field distr
- -1
- =
- Kernel
- Suscep distr
- Seminconductor image is generated from input image by deconvolution
- Our convolution model (PSF/IRF/Kernel)

---

## Slide 11
> **Notes**: QSM 의 리컨은 MRI로부터 측정한 Phase map 에서 시작합니다. Phase map 은 Susceptibility distribution 과 dipole 간의 convolution 의 결과로 나오게 되어 Susceptibility distribution 은 deconvolution 의 과정으로 구할 수 있습니다. 이 문제를 Fourier space로 가져오면 convolution 이 곱셈이 되기 때문에 Susceptibility distribution 은 phase map 의 fourier transform 을 dipole 패턴의 fourier transform 으로 나눔으로써 구할 수 있습니다. 다이폴 패턴의 fourier transform 은 zero-cone 형태를 갖는데, 보시는 바와 같이 zero-cone 의 표면에서는 0의 값을 가지게 됩니다. 따라서 이 지점에서 나눗셈은 정의가 되지 않기 때문에 QSM 을 reconstruction 하는 과정은 ill-posed problem 이 됩니다.

- 11
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
- Dipole
- Suscep distr
- Fourier Transform
- [final image]
- Multiplication
- Fourier Transform
- [Input image]
- Fourier Transform
- [Input image]
- Fourier Transform
- [Final]
- Fourier transform suggests deconvolution requires division by zeros
- Our convolution model (PSF/IRF/Kernel)

---

## Slide 12
- 12
- Estimation or inversion

---

## Slide 13
- 13
- Overdetermined linear equations

---

## Slide 14
- 14
- Multi-objective least-squares

---

## Slide 15
- 15
- Underdetermined linear equations

---

## Slide 16
- 16
- Least-norm solution

---

## Slide 17
> **Notes**: QSM 의 리컨은 MRI로부터 측정한 Phase map 에서 시작합니다. Phase map 은 Susceptibility distribution 과 dipole 간의 convolution 의 결과로 나오게 되어 Susceptibility distribution 은 deconvolution 의 과정으로 구할 수 있습니다. 이 문제를 Fourier space로 가져오면 convolution 이 곱셈이 되기 때문에 Susceptibility distribution 은 phase map 의 fourier transform 을 dipole 패턴의 fourier transform 으로 나눔으로써 구할 수 있습니다. 다이폴 패턴의 fourier transform 은 zero-cone 형태를 갖는데, 보시는 바와 같이 zero-cone 의 표면에서는 0의 값을 가지게 됩니다. 따라서 이 지점에서 나눗셈은 정의가 되지 않기 때문에 QSM 을 reconstruction 하는 과정은 ill-posed problem 이 됩니다.

- 17
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
- Dipole
- Suscep distr
- Fourier Transform
- [final image]
- Multiplication
- Fourier Transform
- [Input image]
- Fourier Transform
- [Input image]
- Fourier Transform
- [Final]
- Fourier transform suggests deconvolution requires division by zeros
- Our convolution model (PSF/IRF/Kernel)

---

## Slide 18
> **Notes**: Dipole deconvolution 문제를 해결하기 위해 몇가지 방법들이 제안되어 왔습니다. 그 중 한가지 방법인 COSMOS방법은 B0 방향에 대해 여러 방향으로 촬영을 한 데이터를 이용합니다. 예를들면, 그대로 왼쪽 오른쪽 앞 뒤 방향으로 얻을 수가 있습니다. COSMOS 방법은 gold-standard 로 생각될 수 있지만, 결과를 얻기 위해 여러 번 스캔해야한다는 점에서 실용적이지 못합니다. 더 실용적인 방법으로는 한 방향에 대해서 한번의 촬영을 하고 Regulization 방법을 이용하는 겁니다. 그러한 방법으로는 TKD 와 MEDI등이 있는데, 여전히 streaking artifacts 나 smoothing 이 일어나고는 합니다.

- Multiple orientation data
- Remove zeros using three or more orientation data
- Single orientation data
- k-space threshold (TKD), regularization using L2 norm etc…
- Potential solutions

---

## Slide 19
> **Notes**: 그래서 이러한 방법을 극복하는 method 로써 저희는 QSM recon 에 딥러닝을 접목시켰고, QSMnet 이라고 부르기로 했습니다. 이 방법의 목표는 single orientation data로부터 COSMOS 의 quality 를 갖는 결과를 얻어내는 것입니다. 그렇기 때문에 multi-orientation data 에 대해서도 consistent 한 QSM 결과가 도출되어야 하고, 이를 하기 위해 저희는 multi-orientation local field 트레이닝 데이터와 상응하는 COSMOS map 을 label로 하여 트레이닝을 하였습니다.

- Single input
- Deep neural network
- (Your choice)
- Ground truth
- Neural network
- Potential solutions

---

## Slide 20
- How do you implement?
- +  regularization term

---

## Slide 21
- How do you implement?
- +  regularization term
- Perform 2D FT, kernel multiplication, 2D FT-1 then columnize matrix

---

## Slide 22
- How do you implement? How do you make skinny matrix?
- +  regularization term
- Perform 2D FT, kernel multiplication, 2D FT-1 then columnize matrix

---
