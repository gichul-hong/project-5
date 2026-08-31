# 반도체 Image Restoration Challenge 실행 계획

> 삼성 DS2 과정 DIP Challenge: 2D dipole convolution + noise 로 손상된 반도체 이미지 복원

---

## 1. 과제 요약

**목표**: 2D dipole convolution kernel과 미지의 노이즈로 열화된 이미지에서 원본 clean 이미지를 복원하는 파이프라인 구축

**열화 모델**: `g(x,y) = h(x,y) * f(x,y) + η(x,y)`
- `f`: 원본 clean 이미지 (7,368장)
- `h`: 2D dipole convolution kernel
- `η`: Gaussian / Rician / Uniform / Salt & Pepper 노이즈
- `g`: 관측된 열화 이미지 (test 100장)

**평가 지표**: PSNR, SSIM (소수점 둘째 자리 반올림)

---

## 2. 데이터 생성 규칙 (필수 준수)

| 제약 | 값 |
|------|-----|
| Clean 1장당 최대 kernel 방향 수 | 6 |
| 방향당 최대 노이즈 수 | 2 |
| **Clean 1장당 총 corrupted 생성 제한** | **12장 (6×2)** |

---

## 3. 교수님 요구사항 분석

### 3.1 Error Map 생성 (2개 이상 다른 노이즈 타입 샘플)

수식: `error_map = (I_restored - I_groundtruth) / I_groundtruth * 100` (%)

- 픽셀별 **상대 오차율(%)** 맵
- 서로 다른 노이즈 타입(Gaussian, Rician, Uniform, Salt & Pepper)의 샘플에 대해 각각 생성
- 예: Gaussian 샘플 1개 + Salt & Pepper 샘플 1개, 또는 4종류 모두

### 3.2 Label-free 접근 (보너스)

Self-supervised / unsupervised denoising:
- **Noise2Void**: 단일 noisy 이미지로 blind-spot training
- **Neighbor2Neighbor**: 노이즈 이미지 쌍으로 self-supervised 학습
- **Noise2Self**: 블라인드 스팟 + 통계적 독립성
- Zero-shot 방법: DIP(Deep Image Prior), ZS-N2N 등

### 3.3 추가 Denoising Network 도입

DnCNN baseline 외에 아래 중 1~2개 도입하여 비교:

| 아키텍처 | 특징 |
|----------|------|
| **NAFNet** | Nonlinear Activation Free, SOTA 성능, 간단한 구조 |
| **Restormer** | Transformer 기반, 고성능 |
| **U-Net + Residual Blocks** | 안정적인 encoder-decoder 구조 |
| **MWCNN** | Wavelet 기반 multi-level denoising |

---

## 4. 전체 파이프라인

```
Clean Image (7,368)
    │
    ├──[Forward Model]──→ Corrupted Training Image
    │    ├─ Dipole convolution (6 orientations)
    │    └─ Noise addition (Gaussian / Rician / Uniform / S&P, 각 2 variants)
    │
    ▼
Training Pairs (Corrupted, Clean)
    │
    ├──[Train]──→ DnCNN Baseline (L2 Loss)
    │
    ├──[Train]──→ NAFNet / Restormer (L1 + SSIM Loss)
    │
    └──[Train]──→ Label-free: Noise2Void / Neighbor2Neighbor
    │
    ▼
Test (100 Corrupted Images)
    │
    ├──→ Denoising Network → Denoised
    ├──→ (Optional) Deconvolution Stage
    └──→ Output: Restored Image
    │
    ▼
Evaluation
    ├──→ PSNR / SSIM (per noise type + overall)
    ├──→ Error Map: (output-GT)/GT * 100 (%)
    ├──→ Baseline (mean/median/adaptive filter) 비교
    └──→ Before/After 시각화
```

---

## 5. 세부 작업 목록

### Phase 1: 환경 설정 & 데이터 준비
- [x] 프로젝트 디렉토리 구조 파악
- [ ] PyTorch 환경 구성 (GPU 사용)
- [ ] 데이터셋 구조 확인: `ref/dataset/train/`, `ref/dataset/val/`, `ref/dataset/test_label/`
- [ ] Dipole convolution simulator 확인 및 데이터 생성 제한 규칙 적용 로직 구현

### Phase 2: Baseline DnCNN 학습
- [ ] 제공된 `train_denoising_example.ipynb` 코드 분석
- [ ] DnCNN 학습 파이프라인 실행
- [ ] DnCNN 수정 버전 (SiLU + GroupNorm 활성화, `out = x + self.dncnn(x)`)
- [ ] Best checkpoint 저장 및 validation PSNR/SSIM 기록

### Phase 3: Error Map 생성
- [ ] Test 데이터에 대해 best checkpoint로 복원 수행
- [ ] 노이즈 타입별 1개 이상 샘플 선정
- [ ] Error map 계산: `(output - label) / (label + ε) * 100`
- [ ] Error map을 `magma` colormap으로 시각화, colorbar 포함
- [ ] 저장: `results/error_maps/{noise_type}/{sample_id}.png`

### Phase 4: 추가 Denoising Network 도입
- [ ] NAFNet 또는 Restormer 구현
- [ ] Loss function: `L_total = L_L1 + λ_SSIM * L_SSIM + λ_Edge * L_Sobel`
- [ ] DnCNN과 동일 조건에서 학습
- [ ] PSNR/SSIM 비교 테이블 생성 (noise type 별 + overall)

### Phase 5: Label-free 접근 (보너스)
- [ ] Noise2Void: blind-spot network + masked loss 구현
- [ ] Neighbor2Neighbor: noisy pair 생성 + self-supervised training
- [ ] 성능 비교: supervised vs label-free

### Phase 6: 종합 평가 & 시각화
- [ ] Conventional filter (mean/median/adaptive) 비교
- [ ] Noise type별 PSNR/SSIM 테이블
- [ ] Baseline grid figure (noise type × method)
- [ ] Error map figure (noise type × sample)
- [ ] 발표 자료용 Before/After 비교 이미지

### Phase 7: 문서화 및 발표 준비
- [ ] 전체 파이프라인 구조 설명
- [ ] 방법론 선택 근거 (왜 NAFNet/Restormer를 골랐는지)
- [ ] 복원 전/후 예시 결과
- [ ] Error map 분석 (어떤 노이즈/영역에서 복원이 어려운지)
- [ ] Label-free 결과 및 supervised와의 비교

---

## 6. 디렉토리 구조 (제안)

```
C:\hong\project-5\
├── deepseek/
│   └── plan.md                    ← 현재 파일
├── src/
│   ├── data_generator.py          # 데이터 생성 (dipole conv + noise)
│   ├── models/
│   │   ├── dncnn.py               # DnCNN baseline
│   │   ├── nafnet.py              # NAFNet
│   │   └── noise2void.py          # Label-free
│   ├── train.py                   # 학습 메인
│   ├── test.py                    # 테스트 & 평가
│   └── utils/
│       ├── metrics.py             # PSNR, SSIM, error map
│       └── visualization.py       # 비교 이미지, error map 시각화
├── results/
│   ├── error_maps/                # Error map 이미지
│   ├── comparisons/               # Before/After 비교
│   └── metrics/                   # JSON 결과 파일
├── logs/                          # 학습 로그 & 체크포인트
├── ref/                           # 참조 자료 (기존)
└── SEMICONDUCTOR_DENOISING_GUIDE.md
```

---

## 7. 예상 결과물

| 산출물 | 설명 |
|--------|------|
| Trained DnCNN | Baseline denoising model |
| Trained NAFNet/Restormer | 추가 denoising network |
| Error Maps | 노이즈 타입별 상대 오차율(%) 시각화 (2개 이상) |
| Comparison Table | Noise type별 PSNR/SSIM (DnCNN vs NAFNet vs Conventional) |
| Baseline Grid | 방법별 × 노이즈별 비교 이미지 |
| Label-free Model | Noise2Void / Neighbor2Neighbor 결과 (보너스) |
| 발표 자료 | 전체 파이프라인 설명, 방법론 선택 근거, 결과 분석 |