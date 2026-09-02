# 반도체 Image Restoration Challenge 실행 계획 v3

> 삼성 DS2 과정 DIP Challenge: 2D dipole convolution + noise 로 손상된 반도체 이미지 복원
>
> 마지막 수정: 2026-09-02 (`train_final_example.ipynb` 및 test 데이터 배포 반영)

---

## 0. 실측으로 확정된 사실 (최우선 참조)

`ref/dataset` 과 `ref/code_denoising+deconv` 를 직접 대조해 검증한 결과다. **추정이 아니라 측정값이다.**

| 항목 | 확정 사실 | 검증 방법 |
|---|---|---|
| **열화 순서** | **dipole conv → noise** (`y = N(Ax)`) | example 노트북 명시 + 잔차 통계 대조 |
| **B0 방향** | **`(0, 1)` 단일 방향** | `test_deconv_only/forward_meta.json` 100 entry 전부 동일 |
| forward model 재현 | `dipole_forward(label)` vs `test_deconv_only` **max err 1.9e-07** | 5샘플 직접 계산 |
| 잔차 = noise | gaussian `std≈σ`, uniform `std≈σ/√3`, rician 양의 bias, S&P sparse | `test_deconv_noise − test_deconv_only` |
| noise 분포 | gaussian/rician/uniform/salt_and_pepper **각 25장** | `noise_meta.json` 집계 |
| 데이터 규모 | train 7,268 (L1 3,167 / L8 4,101) · val 100 (L1 43 / L8 57) · test 100 (**전부 L1**) | 파일 집계 |
| 이미지 | `256×256` `float32`, label 범위 `[0, 1]` (단 max 가 항상 1 은 아님) | 직접 로드 |
| 제공 ckpt | `Unet(chans=64, pools=4)` **13,431,233 params** | `checkpoint_baseline_best.ckpt` 파싱 |
| 제공 ckpt spec | `target="label"`, `clean_branch_prob=0.5`, 100 epoch, batch 64, lr 2e-4, L2, early stop 10 | ckpt 내 `spec` 키 |
| ckpt 확장 흔적 | `model_config` 에 `dncnn_layers=20/features=96`, `unroll_iters=5`, `wiener_bands=4`, `phys_dc_steps` | ckpt 내 `model_config` 키 |
| Wiener ≡ Tikhonov | `D` 가 실수라 `D/(D²+K)` 로 동일. 별도 방법이 아니다 | 항등식 `allclose` 검증 |
| flip TTA | `A(flip x) = flip(A x)` 오차 **1.2e-07** → 사용 가능. `rot90` 은 오차 크므로 불가 | 직접 계산 |
| `|D|` 특성 | range `[-0.667, 0.333]`, `|D|<0.01` 인 주파수 **1.59%**, min `2.27e-05` (0 이 아님) | kernel 직접 계산 |

### 0-1. v2 계획에서 폐기한 가정

| v2 가정 | 실제 | 영향 |
|---|---|---|
| COSMOS multi-orientation inversion 이 핵심 | **test 는 단일 방향 `(0,1)`** 뿐이다. `test_deconv_multi` 는 `forward_meta.json` 만 있고 npy 가 없다 | COSMOS 를 주 파이프라인에서 제외. 단일 방향 Wiener/학습 기반으로 전환 |
| "6 orientations" 규칙이 COSMOS 를 위한 설계 | 채점 대상이 단일 방향이므로 orientation 예산은 **augmentation 여지**일 뿐 | noise 실현 12개 고정으로 규칙 충족 |
| deconv 를 analytic 으로 완전히 해결 가능 | 노이즈가 있으면 analytic 은 붕괴한다 (`gemini iter1_review` 2.3) | 학습 기반 residual 보정이 필수 |

> `ref/code_multiorientation/MOSAIC.ipynb` 의 COSMOS 는 **개념 학습 자료**로 남긴다.
> 다중 방향 데이터가 실제로 배포되면 (`test_deconv_multi` 에 npy 가 채워지면) 5절 확장 항목으로 부활시킨다.

---

## 1. 과제 요약

**목표**: dipole convolution 과 미지의 노이즈로 열화된 이미지에서 원본 clean 이미지를 복원

**열화 모델**: `y = N(A x)`, `A x = F⁻¹{D(k)·F{x}}`, `D(k) = 1/3 − (k·B̂₀)²/|k|²`

- `x`: 원본 clean 이미지
- `A`: dipole convolution (B0 = (0,1) 고정)
- `N`: Gaussian / Rician / Uniform / Salt & Pepper 중 하나
- `y`: 관측 영상 (`test_deconv_noise` 100장)

**평가**: `test_deconv_noise` → 복원 → `test_label` 기준 PSNR / SSIM (소수점 둘째 자리)

**핵심 난점**: magic angle(약 54.7도) 원뿔에서 `D≈0` 이라 그 주파수 성분은 정보가 거의 소실된다.
`1/D` 는 원점 근처에서 4e4 까지 치솟아 노이즈를 폭발적으로 증폭한다 (ill-posed).

---

## 2. 데이터 생성 규칙 (필수 준수)

| 제약 | 값 |
|------|-----|
| Clean 1장당 최대 kernel 방향 수 | 6 |
| 방향당 최대 노이즈 수 | 2 |
| **Clean 1장당 총 corrupted 생성 제한** | **12장** |

**구현**: `(파일명, variant_idx)` → `crc32` seed 로 `(noise_type, sigma)` 를 결정론적으로 확정한 12개 고정 풀.
epoch 을 몇 번 돌려도 noise 실현은 12개를 넘지 않는다 (`claude/iter2_review.md` 5절의 실격 리스크 해소).

`config.augment_flip=True` 면 flip 4종 각각에 12개가 붙어 corrupted 종수는 48이 된다.
flip 은 clean 영상의 기하 변환이고 조교 example 도 동일하게 쓰지만,
엄격 해석이 필요하면 `augment_flip=False` 로 정확히 12장이 된다.

---

## 3. 구현 현황

### 3-1. `claude-final/iter1.ipynb` (작성 완료, CPU 30/30 셀 실행 검증)

Colab 용 단일 노트북. `train_final_example.ipynb` 스켈레톤의 미정의 부분
(`config`, `get_data_wrapper_loader`, `DataKey`, `SPECS`, `train_model`, `TEST_NOISE_META` 등)을 전부 구현했다.

| 구성 | 내용 |
|---|---|
| **데이터 I/O** | Drive → `/content` SSD 로 `(N,256,256) float32` packed `.npy` memmap 캐시. epoch 마다의 Drive I/O 제거 |
| **자기검증** | forward model 을 제공 데이터와 대조, noise 잔차 통계 대조, seed 재현성, Wiener≡Tikhonov, flip 등변성 |
| **validation** | 층화: `이미지 × noise 4종 × σ 3층` 전수 (43×4×3 = 516 조합). run 간 비교 가능 |
| **conventional** | mean/median/adaptive + Wiener, `K` 는 validation sweep (19 grid) |
| **모델 A** | `unet_e2e` — U-Net chans 64, measure→label. 제공 ckpt 와 동일 구조 |
| **모델 B** | `dncnn_blur` — DnCNN 20층 96feat, measure→`Ax` 학습 후 Wiener deconv (K 재sweep) |
| **모델 C** | `phys_unet` — 입력 3채널 `[measure, Wiener(K), Wiener(10K)]`, `out = base + 0.1·UNet`, 마지막 conv **zero-init** |
| **학습** | AMP, cosine LR + warmup, L1(+SSIM+data-consistency), early stop, noise type 별 val 기록 |
| **평가** | conventional 4종 + 제공 baseline + A/B/C + C·flipTTA, noise type 별 PSNR/SSIM |
| **분석** | difference map, zoom-in, **상대오차 error map(%)**, σ-PSNR 산점도, 하위/상위 5장 |

### 3-2. 모델 C 를 주력으로 삼은 근거

`gemini-deconvolution/iter1` 에서 raw measure 만 주고 U-Net 에 deconvolution 을 맡긴 결과
**25.59 dB** 로 한 줄짜리 Wiener filter(42.25 dB)보다 **16.7 dB 나빴다.**
dipole 의 역연산은 전역·비국소인데 CNN 은 국소 연산자이므로 가장 불리한 설계였다.

iter2 에서 해석해를 입력 채널로 주고 residual 만 학습시키자 **78.0 dB** 로 올랐다.
C 는 이 구조를 그대로 가져온 것이고, zero-init 덕분에 **학습 시작 시점 출력이 Wiener base 와 정확히 같아
conventional 성능이 하한으로 보장**된다.

---

## 4. 실행 순서

| 순서 | 작업 | 비고 |
|---|---|---|
| **1** | 스모크 테스트: `config.max_train_images=200`, `epochs=1` | 파이프라인 확인. 반드시 먼저 |
| **2** | 캐시 빌드 + conventional K sweep | conventional baseline 확보 |
| **3** | **C (`phys_unet`) 학습** | 주력. 가장 높은 기대값 |
| **4** | A (`unet_e2e`) 학습 | 제공 ckpt 재현 및 비교 기준 |
| **5** | B (`dncnn_blur`) 학습 + Wiener K 재sweep | 역할 분리 접근의 검증 |
| **6** | 전체 평가 + error map + 취약점 분석 | 발표 자료 대부분이 여기서 나온다 |

---

## 5. 다음 iteration 후보 (측정값을 보고 결정)

| 우선순위 | 항목 | 발동 조건 / 근거 |
|---|---|---|
| 1 | `phys_scale` 상향, `dc_weight` 하향 | C 가 conventional 을 못 넘을 때. residual 이 학습되지 않은 것 |
| 2 | Rician variance-stabilizing transform 또는 bias 보정 | Rician 이 최하위일 때. `E[\|s+n\|]>s`, 신호 0 영역에서 `σ√(π/2)` bias |
| 3 | median prefilter 를 입력 채널로 추가 | S&P 가 최하위일 때. impulse 는 detection 문제 |
| 4 | σ 조건부 입력 (FFDNet 방식) | 저노이즈 구간 상대 이득이 작을 때 |
| 5 | **Noise2Noise (label-free, 가산점)** | 12개 풀에서 같은 noise type 의 두 실현을 pair 로 사용. `claude/iter3` 에서 supervised 대비 −0.98 dB 확인 |
| 6 | Unrolled network + physics DC step | 제공 `model_config` 의 `unroll_iters=5`, `phys_dc_steps` 흔적 |
| 7 | COSMOS multi-orientation | `test_deconv_multi` 에 npy 가 배포되면 |

### 5-1. Label-free 접근의 노이즈별 유효성 (`claude/iter3_review.md` 4절 실측)

| noise | zero-mean | N2N 유효성 | supervised 대비 실측 |
|---|---|---|---|
| gaussian | O | 정확 | **+0.211 dB** |
| uniform | O | 정확 | **+0.428 dB** |
| rician | X (양의 bias) | bias 잔존 | −0.724 dB |
| salt_and_pepper | X (impulse) | L1 필요 | −3.834 dB |

zero-mean 인 2종은 supervised 와 동등하고 아닌 2종에서만 격차가 생겼다.
이론이 데이터로 그대로 확인된 결과라 발표 소재로 강하다.

---

## 6. 과거 Iteration 교훈 (계획에 이미 반영됨)

| 출처 | 교훈 | 반영 위치 |
|---|---|---|
| `claude/iter1_review.md` 2.2 | exp decay LR 은 성능이 오르는 중에 LR 을 죽인다 | cosine LR |
| `claude/iter2_review.md` 3.2 | epoch 6배 + cosine 만으로 +3.83 dB | 60 epoch 기본값 |
| `claude/iter2_review.md` 5 | on-the-fly noise 는 규칙 상한의 5배 | 고정 12개 풀 |
| `claude/iter2_review.md` 4.4 | ALL 평균으로 외삽하지 말고 noise type 별로 분해 | val 을 noise type 별로 기록 |
| `claude/iter3_review.md` 8 | 무작위 추첨 val 은 run 간 비교 불가 | 층화 validation |
| `claude/iter1_review.md` 2.4 | val 분포가 test 와 다르면 모델 선택이 어긋난다 | val 을 L1 한정 |
| `gemini/iter1_review.md` 2.1 | 국소 CNN 에 전역 역연산을 raw 로 떠넘기면 실패 | 모델 C 의 physics 입력 |
| `gemini/iter2_review.md` 3 | 해석해 base + residual + zero-init | 모델 C 구조 |
| `gemini/iter1_review.md` 5.3 | flip 은 등변, rot90 은 불가 | flip TTA only |

---

## 7. 산출물

| 산출물 | 경로 |
|---|---|
| 학습/평가 노트북 | `claude-final/iter1.ipynb` |
| 체크포인트 · history | `{ROOT}/claude-final/logs_final/{id}_{key}/` |
| test metric (장별) | `{ROOT}/claude-final/test_metrics.json` |
| 종합 요약 | `{ROOT}/claude-final/summary_iter1.json` |
| error map | `{ROOT}/claude-final/error_maps/` |
| Wiener K | `{ROOT}/claude-final/best_k.json` |

### 발표 요건 충족 현황

| 요건 | 충족 방법 |
|---|---|
| 전체 파이프라인 설명 | 노트북 2절 forward model + 6절 모델 표 |
| 복원 전/후 비교 시각화 | 8-1 (difference map), 8-2 (zoom-in) |
| 방법론 선택 근거 | 6절 — iter1 의 실패 수치(25.59 vs 42.25 dB)를 근거로 제시 |
| Error map (2종 이상) | 8-3 — noise 4종 각각 상대오차율(%) |
| Conventional 비교 | 8절 표 — mean/median/adaptive + Wiener |
| 취약점 분석 | 8-4 — noise type 별, σ 별, 하위 5장 |
| Label-free (가산점) | 5절 항목 5 (다음 iteration) |
