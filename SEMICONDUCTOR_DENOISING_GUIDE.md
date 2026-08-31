# 🔬 반도체 열화 이미지 복원 (Semiconductor Denoising) 프로젝트 종합 가이드

> **본 문서는 `ref/` 디렉토리 내의 강의 자료(Day 1~3 PPTX), 실습 코드(Jupyter Notebook), 챌린지 평가 기준을 AI Agent 및 개발자가 토큰 낭비 없이 즉시 참조하여 모델을 설계·학습·평가할 수 있도록 요약·정리한 통합 개발 명세서입니다.**

---

## 1. 프로젝트 개요 & 챌린지 목표

- **대회명**: 삼성 DS 과정 Digital Image Processing Challenge (Semiconductor Image Restoration)
- **주요 과제**: 미세 반도체 공정 이미지(SEM/현미경 등)에서 발생하는 다양한 형태의 열화(Noise / Blur / Ill-posed Degradation)를 제거하고 원본 이미지를 고품질로 복원(Restoration / Denoising).
- **최종 평가 메트릭**:
  1. **PSNR (Peak Signal-to-Noise Ratio)** $\uparrow$ (높을수록 우수)
  2. **SSIM (Structural Similarity Index)** $\uparrow$ (높을수록 우수, 1에 가까움)
  3. **평가 방식**: Supervised(지도학습) 또는 Self-supervised(자기지도학습) 모델 적용

---

## 2. 노이즈 및 열화 모델 (Degradation Model)

### 2.1 영상 열화 기본 수식
$$g(x,y) = h(x,y) * f(x,y) + \eta(x,y)$$
- $f(x,y)$: 원본 클린 이미지 (Ground Truth)
- $h(x,y)$: PSF (Point Spread Function / Blur Kernel)
- $\eta(x,y)$: 부가 노이즈 (Additive Noise)
- $g(x,y)$: 열화된 관측 이미지 (Degraded Image)

### 2.2 발생 가능한 노이즈 분포 (Day 1 강의 내용)
1. **Gaussian Noise**: 전자 장비의 열 잡음(Thermal Noise) 형태 $\rightarrow$ 정규분포 ($\sigma \in [0.0, 0.1]$)
2. **Rician Noise**: 저신호 영역에서 왜곡이 발생하는 노이즈 ($\sigma \in [0.0, 0.15]$)
3. **Uniform Noise**: 균일 분포 노이즈 ($\sigma \in [0.0, 0.2]$)
4. **Salt & Pepper Noise**: 극단값(0 또는 1)이 점 형태로 나타나는 충동성 노이즈 ($\sigma \in [0.0, 0.2]$)

---

## 3. 핵심 복원 접근법

### 3.1 전통적 필터링 (Conventional Methods)
- **Mean / Median Filter**: 국소 영역 평균/중앙값 기반 노이즈 제거 (엣지가 뭉개지는 한계)
- **Inverse Filter**: $F(u,v) \approx G(u,v) / H(u,v)$ ($H(u,v) \approx 0$ 부근에서 노이즈 폭발 문제)
- **Wiener Filter**: 최소 평균 제곱 오차(MMSE) 기반 필터링 (파라미터 튜닝 필요)

### 3.2 딥러닝 기반 복원 (Deep Learning / PINN)
- **DnCNN (Deep Convolutional Neural Network for Denoising)**
  - Residual Learning: 입력 이미지에서 **노이즈 성분 $\eta(x,y)$만을 예측**하도록 학습
  - 수식: $\hat{f}(x,y) = g(x,y) - \mathcal{R}(g(x,y))$ (여기서 $\mathcal{R}$은 신경망)
  - 구조: `Conv + ReLU` $\rightarrow$ 다중 `Conv + BatchNorm + ReLU` $\rightarrow$ `Conv`
- **U-Net / Multi-scale Architectures**:
  - 인코더-디코더 구조 + Skip Connection으로 고주파수 세부 디테일 보존
- **최신 복원 모델 확장 아이디어**:
  - NAFNet, Restormer, SCUNet 등 최신 Denoising SOTA 아키텍처 도입 가능
  - Loss 함수 조합: $\mathcal{L}_{total} = \mathcal{L}_{L1} + \lambda_{1} \mathcal{L}_{SSIM} + \lambda_{2} \mathcal{L}_{Perceptual}$

---

## 4. 실습 코드 파이프라인 분석 (`ref/code_denoising/`)

### 4.1 데이터 로더 및 노이즈 주입 파이프라인
```python
# train_denoising_example.ipynb 발췌 요약
NOISE_RANGES = {
    "gaussian": (0.0, 0.1),
    "rician": (0.0, 0.15),
    "uniform": (0.0, 0.2),
    "salt_and_pepper": (0.0, 0.2),
}

# 학습 시 각 배치마다 4종 노이즈 중 무작위 선택하여 Ground Truth 이미지에 합성 노이즈를 얹어 Pair 생성
# 입력 데이터: *.npy 형식의 [0, 1] 범위 정규화된 2D 이미지
```

### 4.2 기본 베이스라인 모델: DnCNN
```python
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
        return x - noise # Residual learning: 원본 복원
```

### 4.3 하이퍼파라미터 기본값
- **Optimizer**: Adam ($lr = 1e-4$, decay rate $0.88$)
- **Loss**: L2 loss 또는 L1 loss
- **Batch Size**: 16 (Train), 1 (Val)
- **Input Dimension**: Single channel grayscale ($1 \times H \times W$)

---

## 5. 파싱된 원본 자료 맵 (Parsed Files Map)

상세 내용이 필요할 때 아래 마크다운 파일들을 개별 열람할 수 있습니다:

| 파일 경로 | 원본 파일 | 내용 요약 |
|---|---|---|
| [ref/parsed_markdown/삼성 DS Project description Day 1.md](file:///C:/hong/project-5/ref/parsed_markdown/삼성%20DS%20Project%20description%20Day%201.md) | Day 1 PPTX | 열화 모델, 노이즈 종류 및 통계적 특성, 기초 필터 |
| [ref/parsed_markdown/삼성 DS Project description Day 2.md](file:///C:/hong/project-5/ref/parsed_markdown/삼성%20DS%20Project%20description%20Day%202.md) | Day 2 PPTX | Deconvolution, Ill-posed Inverse Problem, Wiener Filter, 정규화 기법 |
| [ref/parsed_markdown/삼성 DS Project description Day 3.md](file:///C:/hong/project-5/ref/parsed_markdown/삼성%20DS%20Project%20description%20Day%203.md) | Day 3 PPTX | PINN (Physics-Informed), 딥러닝 기반 디컨볼루션, 일반화(Generalization) 성능 개선 |
| [ref/parsed_markdown/train_denoising_example.md](file:///C:/hong/project-5/ref/parsed_markdown/train_denoising_example.md) | train_denoising_example.ipynb | 전체 학습 파이프라인 코드 (데이터 로더, 모델, 트레이닝 루프) |
| [ref/parsed_markdown/test_denoising.md](file:///C:/hong/project-5/ref/parsed_markdown/test_denoising.md) | test_denoising.ipynb | 테스트/검증 평가 및 Baseline(Mean, Median) 성능 비교 코드 |

---

## 6. 모델 성능 향상을 위한 권장 개발 전략 (Agent 개발 지침)

1. **아키텍처 고도화**:
   - 단순 DnCNN(17 layers)에서 **U-Net 기반 Residual Denoising**, **NAFNet(Nonlinear Activation Free)**, **Restormer** 등으로 확장.
2. **손실 함수(Loss) 개선**:
   - $L_2$ Loss 단독 사용 시 발생하는 Blurry 효과를 방지하기 위해 **$L_1 + \text{SSIM Loss} + \text{Edge/Sobel Loss}$** 조합 적용.
3. **데이터 증강 (Data Augmentation)**:
   - Random Crop, Flip, Rotation ($90^\circ, 180^\circ, 270^\circ$).
   - 복합 노이즈(Gaussian + Salt & Pepper 혼합)에 대한 강건성 확보.
4. **Self-Supervised / Blind Denoising 고려**:
   - Noise2Noise, Noise2Void, Neighbor2Neighbor 기법 적용 검토.
