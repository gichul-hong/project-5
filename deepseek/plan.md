# 반도체 Image Restoration Challenge 실행 계획 v2

> 삼성 DS2 과정 DIP Challenge: Multi-orientation 2D dipole convolution + noise 로 손상된 반도체 이미지 복원
>
> 마지막 수정: 2026-09-02

---

## 1. 과제 요약

**목표**: Multi-orientation 2D dipole convolution kernel과 미지의 노이즈로 열화된 이미지에서 원본 clean 이미지를 복원하는 파이프라인 구축

**열화 모델**: `g(x,y) = h(x,y) * f(x,y) + η(x,y)`

| 기호 | 의미 |
|---|---|
| `f(x,y)` | 원본 clean 이미지 (7,368장) |
| `h(x,y)` | 2D dipole convolution kernel (B0 방향에 의존) |
| `η(x,y)` | 미지의 부가 노이즈 (Gaussian / Rician / Uniform / Salt & Pepper) |
| `g(x,y)` | 관측된 열화 이미지 |

**평가 지표**: PSNR, SSIM (소수점 둘째 자리 반올림)

---

## 2. 데이터 생성 규칙 (필수 준수)

| 제약 | 값 |
|------|-----|
| Clean 1장당 최대 dipole kernel 방향 수 | 6 |
| 방향당 최대 노이즈 수 | 2 |
| **Clean 1장당 총 corrupted 생성 제한** | **12장 (6 orientations × 2 noises)** |

이 규칙은 COSMOS (Calculation Of Susceptibility through Multiple Orientation Sampling) 접근을 위해 설계된 것이다. 최대 6개 방향의 측정치를 pseudo-inverse로 결합하여 dipole blur를 제거하고, 2개 noise variant는 Noise2Noise pair로 활용할 수 있다.

---

## 3. MOSAIC.ipynb 분석 — COSMOS의 핵심 원리

### 3.1 Dipole Kernel & Ill-posedness

```python
# ref/code_multiorientation/MOSAIC.ipynb:38-53
D(k) = 1/3 - (kx*B0_x + ky*B0_y)² / (kx² + ky² + 1e-8)
```

- 단일 B0 방향에서는 k-space의 **magic angle cone (~54.7°)** 에서 `D(k) ≈ 0`
- `|D|`가 0에 가까운 주파수는 복원 불가 (division by zero, noise 증폭)
- Wiener filter는 `K` 파라미터로 이 영역을 정규화하지만, 해당 주파수 성분은 영구 손실

### 3.2 COSMOS Multi-Orientation Inversion

```
각 k-space 픽셀 (i,j) 마다:
  y = [measure_1(i,j), ..., measure_N(i,j)]^T   ← N×1 측정 벡터
  A = [kernel_1(i,j), ..., kernel_N(i,j)]^T     ← N×1 dipole 행렬
  x̂(i,j) = pinv(A) · y                           ← least-squares pseudo-inverse
```

- **여러 B0 방향**의 kernel은 magic angle cone의 위치가 모두 다르다 (방향에 따라 회전)
- N개 방향의 측정치를 쌓으면 over-determined least-squares system이 됨
- N≥5 방향이면 전 k-space에서 거의 완벽한 복원 가능 (MOSAIC 실측)
- 챌린지 규칙의 "최대 6 orientation"은 여기서 비롯된 제약

### 3.3 Noisy Condition — Tikhonov Regularization

```python
# MOSAIC.ipynb:268-280
x̂(i,j) = at_y / (at_a + λ)   ← scalar Tikhonov per k-space pixel
```

- 단순 pseudo-inverse는 노이즈를 급격히 증폭
- Tikhonov 정규화 (λ=0.01)로 노이즈 증폭을 효과적으로 억제
- λ는 GCV (Generalized Cross-Validation)로 label 없이 자동 선택 가능 (gemini iter2에서 검증됨)

### 3.4 COSMOS vs 기존 접근법 비교

| 접근 | Blur 제거 | Noise 제거 | 한계 |
|---|---|---|---|
| **COSMOS + Tikhonov** | 완벽 (multi-orientation) | 부분 (λ로 감쇠) | N≤6 제약, λ 선택 필요 |
| **Wiener (1 orientation)** | 부분 (cone 손실) | 부분 | cone에서 복구 불가능한 정보 손실 |
| **PhysResUNet (gemini)** | 학습 (1 orientation) | 학습 | CNN이 전역 푸리에 역연산을 근사해야 함 |
| **DnCNN (claude)** | 불가 (noise only 모델) | 우수 | blur 전혀 처리 못함 |

---

## 4. 전체 파이프라인 (수정)

```
Clean Image f (7,368장)
    │
    ├──[Forward Model]──→ Train Data (12장/clean)
    │    ├─ Dipole convolution: 6개 B0 방향 (0°, 60°, 120°, 180°, 240°, 300°)
    │    └─ Noise addition: 4종 noise type 중 각 방향당 2개 variant
    │       → 총 6×2 = 12장 per clean
    │
    ▼
Training Pipeline
    │
    ├──[Stage 1: COSMOS Deconvolution]
    │    ├─ Measure 6방향 → COSMOS + Tikhonov → deblurred image
    │    └─ λ 선택: GCV (label-free) 또는 oracle sweep
    │
    ├──[Stage 2: Denoising Network]
    │    ├─ Input: deblurred (residual noise 남아있음)
    │    ├─ Target: clean (supervised) 또는 noisy_pair (N2N)
    │    ├─ Model: DnCNN (SiLU + GroupNorm, residual)
    │    └─ Loss: L1/L2 (supervised) 또는 L1 (N2N)
    │
    └──[Optional: End-to-End Fine-tuning]
         ├─ COSMOS + Denoiser를 하나의 그래프로 연결
         └─ Joint fine-tuning (physics consistency loss + denoising loss)
    │
    ▼
Inference (test 100장)
    │
    ├──→ COSMOS deconv (6 orientations)
    ├──→ Denoiser → Restored f̂
    └──→ Evaluation
    │
    ▼
Evaluation
    ├──→ PSNR / SSIM (per noise type + overall)
    ├──→ Error Map: (f̂ - f_gt) / (f_gt + ε) × 100 (%)
    ├──→ Baseline (Wiener, median, COSMOS-only) 비교
    └──→ Before/After 시각화
```

### 4.1 핵심 변경 사항 (v1 → v2)

| 항목 | v1 | v2 | 근거 |
|---|---|---|---|
| Deconv 접근 | CNN single-orientation | COSMOS multi-orientation | MOSAIC 분석. 6방향 규칙은 COSMOS를 위해 설계됨. Analytic inversion이 CNN보다 월등 |
| Deconv-Denoise 순서 | Denoise → Deconv | **Deconv → Denoise** | COSMOS가 먼저 blur를 제거한 뒤 residual noise만 DnCNN으로 처리 |
| Forward model | `g = f + η` (claude track) 또는 `g = h*f` (gemini track) | **`g = h_dir * f + η` (dir=1..6, full combined)** | 실제 degradation 모델. 기존 두 track은 각각 절반만 풀고 있었음 |
| 12장/clean 활용 | 방향별 noise 2장 (N2N) | 동일 + **N2N + COSMOS inversion input으로 이중 활용** | 6방향은 COSMOS inversion에 묶어서 사용, 각 방향 2장 noise variant는 N2N pair로 |
| N2N 확장 | noise-only pair | **noise variant pair → COSMOS 후 denoiser N2N** | COSMOS 통과 후 잔여 noise 간 pair로 사용 |

---

## 5. 세부 Phase별 작업 목록

### Phase 1: 통합 Forward Model 구축 — 최우선

> 입력: clean image / 출력: 12장 corrupted (6 direction × 2 noise variant), 1장 COSMOS-recon

| # | 작업 | 참조 | 설명 |
|---|---|---|---|
| 1.1 | Dipole kernel simulator 구현 | `MOSAIC.ipynb:38-53` | B0 방향 벡터 → k-space dipole kernel |
| 1.2 | 6개 방향 정의 | — | 0°, 60°, 120°, 180°, 240°, 300° 균등 분할. 최적 방향 수는 Phase 2에서 실험 |
| 1.3 | Multi-orientation forward pass | `MOSAIC.ipynb:57-86` | `measure_k = fft(f) * kernel_dir → ifft` |
| 1.4 | Noise simulator 4종 통합 | `claude/iter1.ipynb` `NoiseSimulator` | Gaussian, Rician, Uniform, S&P. `claude/verify_noise.py`로 검증 완료 |
| 1.5 | **12장/clean variant pool 생성** | `claude/iter3.ipynb` `variant_group_specs()` | `(파일명, dir_idx, noise_idx)` → 결정론적 변형. 규칙 완전 준수 |
| 1.6 | COSMOS + Tikhonov 역연산 | `MOSAIC.ipynb:268-280` | k-space per-pixel least-squares. λ 선택: GCV |
| 1.7 | 열화 순서 검증 | — | `noise ∘ dipole` vs `dipole + noise`. 실제 과제 순서에 맞춰 결정 |

**산출물**: `C:\hong\project-5\deepseek\src\data_generator.py`, `cosmos_inverter.py`

### Phase 2: Baseline Benchmark — Combined Degradation 측정

> 목표: combined degradation에서의 성능 상한/하한을 먼저 측정

| # | 작업 | 설명 |
|---|---|---|
| 2.1 | **Degraded input PSNR/SSIM** | 6방향 measure 중 best vs clean, 평균 vs clean |
| 2.2 | **COSMOS-only (noiseless)** | 6방향 COSMOS pseudo-inverse. 상한 측정 (거의 완벽 재구성 예상) |
| 2.3 | **COSMOS + Tikhonov (noisy)** | λ sweep (1e-6 ~ 1e-1) 후 GCV 선택 vs oracle PSNR 곡선 |
| 2.4 | **Single-dir Wiener vs COSMOS** | 1방향 Wiener가 얼마나 cone에서 손해보는지 정량화 |
| 2.5 | **Conventional denoising on COSMOS output** | median/adaptive filter → PSNR. 딥러닝 이전 conventional 상한 |
| 2.6 | **기존 모델 평가** | `claude/iter2.ipynb` DnCNN을 combined degradation에 직접 넣어 성능 측정 (참고용) |
| 2.7 | 방향 수 ablation | 2/3/4/5/6 orientation COSMOS 성능 비교. 몇 방향이면 충분한지 |

**산출물**: `deepseek/results/baseline_metrics.json`, λ-PSNR graph

### Phase 3: COSMOS + DnCNN Denoising (Supervised)

> 목표: COSMOS deconv → DnCNN denoise 2-stage supervised pipeline. 가장 빠른 high-score 경로.

| # | 작업 | 참조 | 설명 |
|---|---|---|---|
| 3.1 | Training data 준비 | Phase 1 | COSMOS-recon (6방향) → denoiser 입력, clean → target |
| 3.2 | DnCNN supervised 학습 | `claude/iter2.ipynb` | 60 epoch, cosine LR, L1+L2 복합 손실, L1 validation 한정 |
| 3.3 | 층화 validation | `claude/iter4.ipynb` | noise 4종 × σ 3 level × 43 L1 이미지 = 516 조합 전수 평가 |
| 3.4 | Noise type별 성능 분해 | `claude/iter2_review.md` 3.1절 | gaussian/rician/uniform/S&P 각 25장 PSNR/SSIM |
| 3.5 | **SOTA 아키텍처 실험 (교수님 요구사항)** |
| 3.5a | **NAFNet** 구현 | `SEMICONDUCTOR_DENOISING_GUIDE.md` 5.3절 | Nonlinear Activation Free. SOTA 성능, 구조 단순 |
| 3.5b | **Restormer** 구현 | — | Transformer 기반. COSMOS-recon 특성에 더 적합할 가능성 |
| 3.5c | DnCNN vs NAFNet vs Restormer 비교 | — | 같은 데이터/설정에서 PSNR/SSIM, 학습 시간 비교 테이블 |

**산출물**: `deepseek/src/models/dncnn.py`, `nafnet.py`, `restormer.py`, `deepseek/src/train_supervised.py`

### Phase 4: Self-Organizing Map (SOM) 기반 취약 지점 분석

> 목표: COSMOS를 통해 복원된 이미지 내에서 DnCNN이 실패하는 영역의 패턴을 자동 발견하여 복원 전략을 차별화

| # | 작업 | 설명 |
|---|---|---|
| 4.1 | Error map 수집 | COSMOS-recon + DnCNN 결과와 GT의 per-pixel error map 수집 (train set) |
| 4.2 | SOM 클러스터링 | (local gradient, local mean, COSMOS residual, noise type) → 2D SOM grid로 클러스터링 |
| 4.3 | **Hard pixel / Normal pixel 분류** | SOM cluster별 mean error 기준으로 low/high error pixel 클래스 정의 |
| 4.4 | Hard pixel targeted training | Hard pixel에 higher weight loss 적용 → 선택적 denoising 강화 |
| 4.5 | 개선 정량화 | SOM map-based targeting 적용 전/후 PSNR 비교 |

**산출물**: `deepseek/results/som_error_map.png`, `deepseek/src/som_analysis.py`

### Phase 5: Label-Free Approach (가산점)

> 목표: clean label을 사용하지 않는 파이프라인. 챌린지 규칙과 N2N이 자연스럽게 호환.

| # | 작업 | 참조 | 설명 |
|---|---|---|---|
| 5.1 | **Noise2Noise on COSMOS output** | `claude/iter3.ipynb` | 12장 풀에서 같은 (clean, dir)의 noise variant끼리 pair → `COSMOS(A) → COSMOS(B)` |
| 5.2 | N2N 수렴 분석 | `claude/iter3_review.md` 5절 | 이론 하한 E&#124;noise&#124; 대비 train loss 측정 |
| 5.3 | Noise type별 분해 | `claude/iter3_review.md` 4절 | zero-mean 여부에 따른 N2N 유효성 검증 (gaussian/uniform=OK, rician/S&P=손실) |
| 5.4 | Noise2Void | — | COSMOS 출력만으로 blind-spot training. Pair도 불필요 |
| 5.5 | Self-supervised 모델 선택 | `claude/iter3_review.md` 6절 | val selfsup loss로 best ckpt 선택 → oracle과의 상관 분석 |
| 5.6 | Supervised vs Label-free 종합 비교표 | — | Noise type × Method matrix. 발표용 |

**산출물**: `deepseek/src/train_n2n.py`, `deepseek/results/label_free_comparison.json`

### Phase 6: Conventional Filter & Analytic Pipeline 비교

| # | 작업 | 설명 |
|---|---|---|
| 6.1 | COSMOS (6방향) + median filter (3×3, 5×5) | Conventional denoising baseline |
| 6.2 | COSMOS + adaptive Wiener filter | spatially-varying noise 추정 후 adaptive filtering |
| 6.3 | COSMOS + Tikhonov only (no denoiser) | λ를 noise level에 맞춰 최적화한 analytic-only 상한 |
| 6.4 | 1-dir Wiener + median vs COSMOS + DnCNN | "왜 multi-orientation + DL인가" 발표 근거 |

### Phase 7: Error Map 생성 (교수님 요구사항)

| # | 작업 | 설명 |
|---|---|---|
| 7.1 | Error map 계산 | `(f̂ - f_gt) / (f_gt + ε) × 100` (상대 오차율 %) |
| 7.2 | 노이즈 타입 4종 × 샘플 최소 2개 | Gaussian 1 + Rician 1 + Uniform 1 + S&P 1, 각 2샘플 이상 |
| 7.3 | 시각화 | magma colormap, colorbar 포함, 256×256 |
| 7.4 | Error map 분석 | 어떤 노이즈 타입/영상 구조에서 복원이 어려운지 패턴 분석 |

**산출물**: `deepseek/results/error_maps/{noise_type}/{sample_id}.png`

### Phase 8: 종합 평가 & 시각화

| # | 작업 | 설명 |
|---|---|---|
| 8.1 | Noise type × Method PSNR/SSIM 최종 테이블 | DnCNN vs NAFNet vs Restormer vs N2N vs N2V vs COSMOS-only vs Wiener+median |
| 8.2 | Baseline grid figure | 각 noise type별 best method 전/후 비교 이미지 (6×4 grid) |
| 8.3 | Learning curve | epoch vs val PSNR (noise type별 분리), train loss vs 이론 하한 |
| 8.4 | Error map dashboard | 4 noise type × 2 samples = 8개 error map 한 페이지 |
| 8.5 | Ablation summary | 방향 수 vs PSNR, λ vs PSNR (noise level별) |

### Phase 9: 발표 자료 준비

| # | 작업 | 설명 |
|---|---|---|
| 9.1 | **파이프라인 구조도** | Clean → 6-dir forward → COSMOS deconv → Denoiser → Restored (시각적 flow) |
| 9.2 | **방법론 선택 근거** | COSMOS (multi-orientation 정보 결합) + DnCNN/NAFNet (residual noise 제거). 각 단계를 왜 분리했는지 |
| 9.3 | **복원 전/후 예시** | 입력 (most blurred direction) vs COSMOS-only vs COSMOS+DnCNN vs GT |
| 9.4 | **Error map 분석** | Hard pixel (error > 5%) 분포, edge vs flat region 성능 비교 |
| 9.5 | **Label-free 결과 비교** | Supervised vs N2N vs N2V. 격차의 이론적 설명 (zero-mean 노이즈 조건) |
| 9.6 | **Conventional 대비 딥러닝 이득** | "왜 딥러닝인가" — COSMOS-only 대비 denoiser 추가로 얻는 dB 정량화 |

---

## 6. 과거 Iteration에서의 주요 교훈

### 6.1 Claude Track (Denoising Only)

| Iter | 핵심 발견 | 계획 반영 |
|---|---|---|
| iter1 | exp decay LR은 epoch 증가와 양립 불가 → cosine annealing 필요 | Phase 3.2 |
| iter1 | val이 test와 분포 다름 (L8 과다) → val을 L1으로 한정 | Phase 3.3 |
| iter2 | 60 epoch + cosine LR만으로 +3.83 dB (30.51 → 34.34) | base config로 채택 |
| iter2 | **on-the-fly 60 epoch = 규칙 5배 위반** → 고정 12장 pool 필수 | Phase 1.5 |
| iter3 | N2N이 supervised 대비 −0.98 dB (label-free) | Phase 5 기대치 설정 |
| iter3 | **규칙 준수 + label-free 가 같은 설계로 동시 해결** | 아키텍처 확정 |
| iter3 | val 노이즈 추첨이 iteration 간 비교를 불가능하게 함 → 층화 필요 | Phase 3.3 |
| iter4 | 층화 validation: 43장 × 4 noise × 3 σ level = 516 조합 전수 | Phase 3.3 |

### 6.2 Gemini Track (Deconvolution Only)

| Iter | 핵심 발견 | 계획 반영 |
|---|---|---|
| iter1 | **Raw CNN이 전역 푸리에 역연산을 학습하지 못함** (U-Net 25.6 dB vs Wiener 42.3 dB) | deconv를 analytic (COSMOS)로 분리 |
| iter2 | GCV로 K 자동 선택 → 80 dB. Physics channels + residual 구조의 승리 | COSMOS의 λ 선택에 GCV 적용 |
| iter2 | **Flip TTA +1.2 dB** (dipole kernel은 flip에 대해 등변) | inference 시 TTA 적용 |
| iter2 | 노이즈 있으면 analytic이 급락 (σ=1e-4에서 K=0 → 29 dB) | COSMOS + denoiser 조합의 정당성 |
| iter3 | Self-supervised physics consistency로 라벨 없이 78 dB 달성 | Phase 5 연관 |

### 6.3 기존 Plan의 문제점 (v1)

| 문제 | 해결 |
|---|---|
| Denoising과 deconv를 순서 없이 모호하게 배치 | **Deconv first (COSMOS) → Denoise second** 로 순서 확정 |
| 단일 orientation deconv를 가정 | Multi-orientation COSMOS로 변경. 규칙의 "최대 6방향"이 여기서 비롯된 것임을 인지 |
| claude/gemini track이 별개로 존재 | 통합: COSMOS (gemini의 physics insight) + DnCNN (claude의 denoising 성과) |
| dipole이 "미제공"이라며 보류 | `MOSAIC.ipynb`에 dipole kernel 구현이 이미 존재 |

---

## 7. 디렉토리 구조 (수정)

```
C:\hong\project-5\
├── deepseek/
│   ├── plan.md                          ← 이 파일
│   ├── src/
│   │   ├── data_generator.py            # Forward model: 6-dir dipole + noise
│   │   ├── cosmos_inverter.py           # COSMOS + Tikhonov deconv
│   │   ├── dipole_kernel.py             # Dipole kernel (MOSAIC에서 포팅)
│   │   ├── noise_simulator.py           # Noise 4종 시뮬레이터
│   │   ├── models/
│   │   │   ├── dncnn.py                 # DnCNN (SiLU + GroupNorm)
│   │   │   ├── nafnet.py                # NAFNet
│   │   │   └── restormer.py             # Restormer
│   │   ├── train_supervised.py          # Supervised training main
│   │   ├── train_n2n.py                 # Noise2Noise training
│   │   ├── train_n2v.py                 # Noise2Void training
│   │   ├── test.py                      # Inference & evaluation
│   │   └── utils/
│   │       ├── metrics.py               # PSNR, SSIM, error map
│   │       └── visualization.py         # 비교 이미지, error map 시각화
│   ├── results/
│   │   ├── error_maps/                  # Error map PNG
│   │   ├── comparisons/                 # Before/After 비교 PNG
│   │   ├── som/                         # SOM analysis outputs
│   │   └── metrics/                     # JSON metric files
│   └── logs/                            # 체크포인트 & history.json
├── claude/                              # Denoising reference (읽기 전용)
├── gemini-deconvolution/                # Deconv reference (읽기 전용)
├── ref/                                 # 참조 자료
│   ├── dataset/                         # Clean images & test data
│   ├── code_denoising/                  # 제공 예제 (denoising)
│   ├── code_deconvolution/              # 제공 예제 (deconvolution)
│   └── code_multiorientation/           # MOSAIC (COSMOS reference)
└── SEMICONDUCTOR_DENOISING_GUIDE.md
```

---

## 8. 실행 우선순위 요약

| 순서 | 우선순위 | Phase | 설명 | 예상 시간 |
|---|---|---|---|---|
| **1** | 🔴 Critical | Phase 1 | 통합 forward model + COSMOS inverter | 2-3h |
| **2** | 🔴 Critical | Phase 2.1~2.7 | Combined degradation baseline 측정 | 1h |
| **3** | 🟡 High | Phase 3.1~3.4 | COSMOS + DnCNN supervised pipeline | 4-5h |
| **4** | 🟡 High | Phase 3.5 | NAFNet / Restormer 비교 | 6-8h |
| **5** | 🟢 Medium | Phase 5 | Label-free (N2N, N2V) | 3-4h |
| **6** | 🟢 Medium | Phase 4 | SOM 기반 취약점 분석 | 2-3h |
| **7** | 🔵 Low | Phase 6 | Conventional filter 비교 | 1h |
| **8** | 🔵 Low | Phase 8 | 종합 평가 & 시각화 | 2h |
| **9** | 🔵 Low | Phase 9 | 발표 자료 | 3h |

**총 예상**: 24-30 시간

---

## 9. 불확실성 / 추가 정보 대기 항목

| # | 불확실한 사항 | 영향 | 결정 시점 |
|---|---|---|---|
| 1 | **열화 순서**: `η ⊕ (h ⊛ f)` vs `h ⊛ (f + η)` | COSMOS 정확도에 직접 영향. noise가 convolution 앞에 있으면 COSMOS 입력이 덜 왜곡됨 | 가이드라인 확인 |
| 2 | 실제 test 100장이 6방향 측정치 + noise인지 여부 | 6방향 측정이 하나의 파일에 묶여서 올지, 별도 파일인지 | 가이드라인 확인 |
| 3 | B0 방향이 실제로 몇 개, 어떤 각도로 주어지는지 | COSMOS 방향 수 최적화에 영향 | 가이드라인 확인 |
| 4 | Dipole kernel 정규화 상수 (`+1e-8`)가 실제 forward model과 동일한지 | COSMOS 정확도. 다르면 mismatch로 에러 발생 | 학습 시 검증 |
| 5 | 채점 test의 noise type·강도가 training noise와 다른지 | `iter1_review.md` 4.1절 지적대로, N2V + TTA로 방어 필요 | — |