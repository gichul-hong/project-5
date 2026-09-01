# gemini-deconvolution iter3 리뷰 및 종합 기술 문서

- **대상 노트북**: `gemini-deconvolution/iter3.ipynb`
- **핵심 목표**: **Clean Label 을 전혀 사용하지 않는 Self-Supervised / Physics-Informed Deconvolution 파이프라인 구축**
- **비교 기준**: 
  - `gemini-deconvolution/iter1.ipynb` (Baseline U-Net 25.59 dB / Wiener 42.25 dB)
  - `gemini-deconvolution/iter2.ipynb` (Supervised PhysicsResidualUnet 78.01 dB / GCV 83.90 dB)

---

## 1. 이론적 배경 및 필요성

### 1.1 왜 Deconvolution에서 Self-Supervised Learning인가?
반도체 영상 복원 과제의 물리적 열화 모델은 다음과 같습니다:
$$g(x,y) = h(x,y) * f(x,y) + \eta(x,y)$$
- $f$: 원본 Clean 반도체 영상 (Ground Truth)
- $h$: 2D Dipole Convolution Kernel (물리 모델)
- $\eta$: 미지의 노이즈 (Gaussian, Rician, Uniform, Salt & Pepper)
- $g$: 관측된 측정 영상 (Measure)

실제 산업 및 측정 환경에서는 **Clean 정답 영상 $f$를 획득하기 어렵거나 불가능**합니다.
따라서 오직 관측된 $g$와 물리적 순방향 연산자 $\mathcal{H}$만을 이용하여 원본 $f$를 복원하는 **Label-Free / Self-Supervised Deconvolution** 접근법이 강력한 가산점 및 학술적 정당성을 갖습니다.

---

## 2. 손실 함수 설계 (`SelfSupDeconvLoss`)

Clean Ground Truth $f$ 없이 오직 물리 연산자 $\mathcal{H}$ 와 관측치 $g$ 만으로 네트워크 $f_\theta$ 를 최적화합니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \alpha \mathcal{L}_{\text{spectral}} + \lambda_{\text{tv}} \mathcal{L}_{\text{tv}}$$

### 2.1 세부 손실 항목

1. **Forward Physics Consistency Loss ($\mathcal{L}_{\text{data}}$)**:
   $$\mathcal{L}_{\text{data}} = \| \mathcal{H}(\hat{f}) - g \|_1 = \left\| \mathcal{F}^{-1}\{ \mathcal{F}\{\hat{f}\} \cdot D(k) \} - g \right\|_1$$
   - 모델이 추정한 복원 영상 $\hat{f}$에 Dipole forward kernel을 다시 걸었을 때, 입력으로 주어진 관측치 $g$와 일치하도록 강제합니다.
   - $L_1$ norm을 사용하여 이상치 및 노이즈에 강건하게 수렴하도록 유도합니다.

2. **Dual-Domain Spectral Consistency Loss ($\mathcal{L}_{\text{spectral}}$)**:
   $$\mathcal{L}_{\text{spectral}} = \| \hat{F}(k) \cdot D(k) - G(k) \|_2^2$$
   - Fourier k-space 주파수 도메인에서 물리 연산 일관성을 추가로 보존하여 주파수 스펙트럼의 왜곡을 방지합니다.

3. **Total Variation (TV) Regularization ($\mathcal{L}_{\text{tv}}$)**:
   $$\mathcal{L}_{\text{tv}} = \frac{1}{HW} \left( \|\nabla_x \hat{f}\|_1 + \|\nabla_y \hat{f}\|_1 \right)$$
   - Dipole kernel의 치명적 결함인 **Magic Angle ($54.7^\circ$) Zero-cone 표면의 결손 주파수**로 인한 스트리킹 아티팩트 및 고주파 진동을 완화합니다.

---

## 3. 핵심 아키텍처 및 데이터 흐름

```
[입력 측정값 (Measure g)]
       │
       ├───► [Analytic Multi-Inversion Module]
       │        ├─ GCV Wiener Inversion
       │        ├─ TKD (clip 5.0) Inversion
       │        ├─ Fixed Wiener Inversions (K=1e-4, 1e-2)
       │        └─ GCV grad-Tikhonov Inversion
       │
       ▼ (6-channel Physics Feature Map)
[PhysicsResidualUnet]
       │
       ▼ (Residual Estimation Δx)
[Restored Image: x̂ = GCV_Base + Δx]
       │
       ├───► [Dipole Forward Operator ℋ] ──► ℋ(x̂) vs g  (Data Loss)
       │                                     F̂(k)·D(k) vs G(k) (Spectral Loss)
       │
       └───► Total Variation Regularization: TV(x̂)
```

---

## 4. 완전 무라벨 모델 선택 (`select_by="selfsup"`)

- 학습 중 최적 체크포인트를 선정할 때도 Clean 라벨(Oracle PSNR)에 의존하지 않고, Validation 셋의 **Self-Supervised Loss $\mathcal{L}_{\text{selfsup}}$**가 가장 낮은 모델을 Best로 자동 저장합니다.
- `history.json`에 `val_psnr`(Oracle)과 `val_selfsup`를 동시에 기록하여, **"라벨 없이 선택했을 때 Oracle 대비 손실(dB)이 얼마인가"**를 정량적으로 증명할 수 있습니다.

---

## 5. 실행 및 성능 비교 가이드

| 실험군 | 학습 방식 | 입력 $\to$ Target | 예상 PSNR | 특징 |
|---|:---:|:---:|:---:|---|
| **iter1** | Supervised | Measure $\to$ Clean | ~25.6 dB | Baseline U-Net (순수 딥러닝 초기 모델) |
| **iter2** | Supervised (Physics) | Measure $\to$ Clean | **78.0 dB / 83.9 dB** | GCV 해석적 초기해 + Residual 지도학습 |
| **iter3** | **Self-Supervised** | **Measure $\to$ Forward Consistency** | **~75-80 dB** | **Clean 라벨 0장 사용, 완전 무라벨/자기지도** |

---

## 6. 결론 및 기대 효과

`gemini-deconvolution/iter3.ipynb`는 단순한 비지도 학습 시도를 넘어, **물리 법칙(Dipole Convolution Forward Operator)과 딥러닝을 결합하여 정답 라벨 없이도 이상적인 역문제를 풀어내는 차세대 복원 프레임워크**입니다.
대회 발표 및 평가 시 강력한 기술적 차별점을 제공합니다.
