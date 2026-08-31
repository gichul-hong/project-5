# 🔬 삼성 DS Semiconductor Image Restoration Challenge 종합 가이드 & 개발 명세서

> **본 문서는 `ref/강의자료` (Day 1~3 PPTX), 실습 코드(Jupyter Notebook), 챌린지 평가 기준을 AI Agent 및 개발자가 토큰 낭비 없이 즉시 참조하여 규정을 준수하면서 고성능 모델을 개발할 수 있도록 작성된 공식 프로젝트 명세서입니다.**

---

## 📌 1. 공식 챌린지 룰 & 평가 규정 (Evaluation & Requirements)

### 1.1 데이터셋 구성 및 제약 조건 (Dataset Constraints)
* **Train / Validation Dataset**:
  * **Clean Images**: 총 **7,368장** 제공
  * **Forward Corruption Model**: 2D dipole convolution (제공된 시뮬레이터 모델) + Noise model (노이즈 특성은 미제공)
* **⚠️ 데이터 합성 생성 제한 규칙 (Data Generation Limits - 필수 준수!)**:
  * Clean Image 1장당 **최대 6개**의 Dipole Kernel 방향(Orientations)까지만 생성 가능
  * 각 방향(Orientation)당 **최대 2개**의 Noisy 이미지만 생성 가능
  * ❌ **Clean Image 1장당 총 12개를 초과하는 Corrupted 이미지를 생성할 수 없음** ($6 \text{ orientations} \times 2 \text{ noises} = \text{최대 12개}$)
* **Test Dataset**:
  * **Corrupted Images만 100장 제공** (Sample당 1장, Clean Ground-Truth 라벨 미제공)
  * Colab 환경에서 암호화된 평가자(Encrypted Evaluator)가 포함된 Test 코드로 실행 및 채점

---

### 1.2 공식 평가 메트릭 (Evaluation Metrics)
* **정량 평가 지표**: Test 데이터셋에 대한 복원 품질 평가
  1. **PSNR (Peak Signal-to-Noise Ratio)** $\uparrow$ (높을수록 우수)
  2. **SSIM (Structural Similarity Index)** $\uparrow$ (높을수록 우수)
* **소수점 표기**: 소수점 둘째 자리에서 반올림하여 기록

---

### 1.3 가산점 및 발표 평가 기준 (Bonus & Presentation)
* **💡 부분 가산점 (Partial Bonus)**:
  * **Label-free (Self-supervised / Unsupervised / Noise2Void / Neighbor2Neighbor 등)** 파이프라인 적용 시 부분 가산점 부여
* **발표 평가 요건 (Presentation Requirements)**:
  1. **전체 파이프라인의 명확한 설명** (Overall pipeline structure)
  2. **복원 전/후 예시 결과 비교 시각화** (Before / After restoration examples)
  3. **특정 방법론을 선택한 이유와 타당성 입증** (Justification for method choice)

---

## 🛠️ 2. 허용되는 접근법 (Allowed Approaches)

참가자는 아래의 모든 접근법을 자유롭게 탐색하고 결합할 수 있습니다:
1. **Classical Image Processing**: 전통적 필터링, 정규화 역필터, Wiener 필터, TV(Total Variation) 정규화
2. **End-to-End Deep Learning**: Corrupted $\rightarrow$ Clean 복원 단일 신경망
3. **Two-Stage Approach**: 1단계 Denoising + 2단계 Deconvolution 분리 파이프라인
4. **Self-Supervised / Label-Free Strategies (가산점 대상)**: 실제 환경에 부합하는 라벨 프리 학습 전략
5. **Physics-Informed Approach (PINN)**: 물리적 결함 모델(2D Dipole Convolution)을 Loss 또는 Network 구조에 직접 결합

---

## 📐 3. 영상 열화 및 물리 모델 (Forward Degradation Model)

### 3.1 기본 열화 수식
$$g(x,y) = h(x,y) * f(x,y) + \eta(x,y)$$
- $f(x,y)$: 원본 클린 이미지 (Ground Truth, 7,368장)
- $h(x,y)$: 2D Dipole Convolution Kernel (PSF / IRF)
- $\eta(x,y)$: 미지의 부가 노이즈 (Gaussian, Rician, Uniform, Salt & Pepper 등)
- $g(x,y)$: 관측된 열화 이미지 (Test 100장)

### 3.2 Fourier 공간에서의 역문제(Inverse Problem) 특성
$$\mathcal{F}\{g\} = \mathcal{F}\{h\} \cdot \mathcal{F}\{f\} + \mathcal{F}\{\eta\}$$
* Dipole Kernel의 Fourier 변환은 Zero-cone 표면에서 0의 값을 가집니다.
* 단순 $1 / \mathcal{F}\{h\}$ 나눗셈은 **0으로 나누기(Division by zero)** 및 노이즈 급격 증폭을 유발하는 **Ill-posed Problem**입니다.

---

## 🧪 4. 베이스라인 실습 코드 파이프라인 (`ref/code_denoising/`)

### 4.1 기본 DnCNN 베이스라인 아키텍처
```python
import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, in_channels=1, out_channels=1):
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        ]
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(n_channels, out_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise # Residual learning (예측된 노이즈 제거)
```

### 4.2 기본 학습 파라미터 (Example Config)
- **Batch Size**: Train 16, Validation 1
- **Optimizer**: Adam ($lr=1e-4$, decay rate $0.88$)
- **Loss**: L2 / L1 Loss
- **Noise Simulation Ranges**:
  - Gaussian: $\sigma \in [0.0, 0.1]$
  - Rician: $\sigma \in [0.0, 0.15]$
  - Uniform: $\sigma \in [0.0, 0.2]$
  - Salt & Pepper: $\sigma \in [0.0, 0.2]$

---

## 🚀 5. Agent 개발 전략 및 고득점 체크리스트

1. **데이터 생성 규칙 준수 검증**:
   - Clean 1장당 Orientation $\le 6$, Noise $\le 2$, 총 $\le 12$장 생성 규칙을 초과하지 않는 데이터로더 구현.
2. **복합 손실 함수 설계**:
   - $\mathcal{L}_{total} = \mathcal{L}_{L1} + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{Edge} \mathcal{L}_{Sobel}$
3. **SOTA 복원 아키텍처 적용**:
   - NAFNet (Nonlinear Activation Free Network), Restormer, U-Net with Residual Blocks
4. **Self-Supervised / Label-Free 모듈 추가**:
   - 가산점 획득을 위한 Self-Supervised / Zero-Shot 복원 전략 구현 및 비교 분석
5. **정량 & 정성 비교 시각화 자동화**:
   - Test 100장에 대한 PSNR/SSIM 평가 및 Before/After 비교 이미지 저장 스크립트 작성

---

## 📂 6. 파싱된 원본 자료 상세 색인

- [삼성 DS Project description Day 1.md (Slide 24~30 평가 및 규칙 상세)](file:///C:/hong/project-5/ref/parsed_markdown/삼성%20DS%20Project%20description%20Day%201.md#L200-L279)
- [삼성 DS Project description Day 2.md (Deconvolution 수식 및 역필터 이론)](file:///C:/hong/project-5/ref/parsed_markdown/삼성%20DS%20Project%20description%20Day%202.md)
- [삼성 DS Project description Day 3.md (PINN, 딥러닝 일반화, QSMnet 구조)](file:///C:/hong/project-5/ref/parsed_markdown/삼성%20DS%20Project%20description%20Day%203.md)
- [train_denoising_example.md (Colab 학습 파이프라인 전체)](file:///C:/hong/project-5/ref/parsed_markdown/train_denoising_example.md)
- [test_denoising.md (Encrypted Evaluator 및 Baseline 비교)](file:///C:/hong/project-5/ref/parsed_markdown/test_denoising.md)
