# iter2 리뷰

- 대상: `claude/iter2.ipynb` (Colab 전구간 실행 완료, exec 1~22)
- run dir: `.../logs_denoising/00001_train`
- 학습 시간: **1시간 17분 33초** (60 epoch) / test 포함 총 1시간 18분 27초
- 적용 범위: `iter1_review.md` 개선 항목 **1번만** (epoch↑ + cosine LR + validation L1 한정)

---

## 1. 어떻게 학습했고, 얼마나 학습했는지

### 1.1 학습 방식

| 항목 | 값 | iter1 대비 |
|---|---|---|
| 학습 목표 | supervised 회귀. corrupted → clean 직접 복원 | 동일 |
| 손실 함수 | **L2 (MSE)**, 이미지별 평균 후 배치 평균 | 동일 |
| optimizer | Adam, betas (0.9, 0.99), weight decay 없음 | 동일 |
| 초기 LR | 1e-4 | 동일 |
| **LR 스케줄** | **cosine annealing** (`lr_tol=1` 구간 유지 후 → `lr × 0.02` = 2e-6) | **exp decay 0.88 → cosine** |
| batch size | 16 (train) / 1 (valid) | 동일 |
| 입력 크기 | **256×256 전체 이미지** (crop/patch 없음) | 동일 |
| augmentation | 랜덤 H/V flip, 각 p=0.5 | 동일 |
| 노이즈 주입 | **on-the-fly**. 매 batch 마다 4종 중 균등 랜덤 선택 후 σ ~ U(범위) | 동일 |
| mixed precision | 사용 안 함 (fp32) | 동일 |

노이즈 σ 범위 (`NOISE_RANGES`):

| noise | σ 범위 |
|---|---|
| gaussian | 0.0 ~ 0.1 |
| rician | 0.0 ~ 0.15 |
| uniform | 0.0 ~ 0.2 |
| salt_and_pepper | 0.0 ~ 0.2 |

### 1.2 학습 분량

| 항목 | 값 |
|---|---|
| epoch | **60** (iter1: 10) |
| iteration / epoch | 455 (= ceil(7268 / 16)) |
| **총 optimizer step** | **27,300** (iter1: 4,550) |
| 총 이미지 forward | **436,080 장** (= 7,268 × 60) |
| epoch당 소요 | 약 77.6초 |
| 총 학습 시간 | 1h 17m 33s |

### 1.3 LR 실측 궤적

| epoch | 1 | 2 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|
| LR | 1.000e-4 | 1.000e-4 | 9.547e-5 | 7.850e-5 | 5.365e-5 | 2.805e-5 | 9.014e-6 | 2.000e-6 |

iter1의 exp decay는 epoch 10에서 이미 3.60e-5(초기값의 36%)까지 떨어졌으나, cosine은 epoch 10에서 9.55e-5(95%)를 유지한다. **의도한 대로 전반부 학습률이 보존되었다.**

### 1.4 모델 선택 및 저장

| 항목 | 값 |
|---|---|
| validation 주기 | 매 epoch |
| validation 대상 | **`L1_*.npy` 43장** (iter1: `*.npy` 100장) |
| validation 노이즈 | seed 고정 (`crc32(파일명)`) → epoch 간 재현 가능 |
| best 선택 기준 | validation PSNR |
| **선택된 best** | **epoch 56, val PSNR 33.6488** |
| checkpoint 저장 | 10 epoch 주기 (6개) + best 1개 |
| 비교 이미지 저장 | 10 epoch 주기 → 24 png (6 epoch × 4장) |

iter1은 매 epoch 저장으로 ckpt 10개 + png 40장이었다. iter2는 60 epoch이므로 주기 저장으로 바꿔 Drive 사용량을 ckpt 144MB → 17MB, png 360MB → 36MB 수준으로 억제했다.

---

## 2. 사용 모델 및 데이터 양

### 2.1 모델 — iter1과 완전히 동일 (변경 없음)

| 항목 | 값 |
|---|---|
| 아키텍처 | DnCNN (residual) |
| `channels` | 1 (grayscale) |
| `num_of_layers` | 17 |
| `kernel_size` | 3 |
| `padding` | 1 (해상도 유지) |
| `features` | 64 |
| **파라미터 수** | **593,024** |
| 정규화 | GroupNorm(4, 64) — BatchNorm 아님 |
| 활성함수 | SiLU |
| residual | `out = x + dncnn(x)` (노이즈가 아니라 잔차를 더함) |

구조:

```
Conv(1→64, 3x3) → SiLU
[ Conv(64→64, 3x3) → GroupNorm(4,64) → SiLU ]  × 16
Conv(64→1, 3x3)
+ skip connection (입력 x 를 더함)
```

**iter2는 모델을 전혀 건드리지 않았다.** 성능 변화는 순수하게 학습 방식(epoch, LR schedule)과 모델 선택 기준(validation subset)의 결과다.

### 2.2 데이터 양

**원본 clean 이미지** (모두 256×256, float32, 값 범위 [0, 1])

| split | 장수 | 사용 방식 |
|---|---|---|
| `train/` | **7,268** | L1 3,167 + L8 4,101. 전량 학습에 사용 |
| `val/` | 100 | **그중 L1 43장만 사용** (L8 57장 미사용) |
| `test_label/` | 100 | 전량 L1. 채점용 정답 |
| `test_noise_only/` | 100 | 사전 생성된 corrupted. 모델 입력 |
| **합계 (clean)** | **7,368** | 강의자료 명시값과 일치 |

**Corrupted 이미지 (학습용)**

| 항목 | 값 |
|---|---|
| 생성 방식 | **on-the-fly**. 디스크에 저장하지 않음 |
| 사전 생성 파일 | 없음 |
| **누적 생성량** | **clean 1장당 60개** (60 epoch × epoch당 1회) |
| 총 corrupted 소비량 | 436,080 장 |
| 서로 다른 (clean, flip) 조합 | 29,072 (7,268 × 4) |
| dipole orientation | **0개 (미적용)** |

**Test 입력 데이터**

`test_noise_only/test_noise_only/*.npy` 100장. `noise_meta.json`에 파일별 `noise_type`과 `sigma`가 기재되어 있고 4종이 각 25장씩 균등 분포한다. 이 값은 채점에는 쓰이지 않고 noise별 성능 분해에만 사용했다.

---

## 3. 결과

### 3.1 Test 성능 (채점 대상 100장)

| noise | n | PSNR in | **PSNR out** | SSIM in | **SSIM out** |
|---|---|---|---|---|---|
| gaussian | 25 | 27.668 | 34.343 | 0.6835 | 0.9327 |
| rician | 25 | 24.079 | 30.607 | 0.7104 | 0.9042 |
| uniform | 25 | 29.631 | 34.421 | 0.7950 | 0.9496 |
| salt_and_pepper | 25 | 17.304 | **37.979** | 0.4746 | **0.9842** |
| **ALL** | **100** | 24.671 | **34.337** | 0.6659 | **0.9427** |

### 3.2 iter1 대비

| noise | iter1 | iter2 | Δ PSNR |
|---|---|---|---|
| gaussian | 32.044 | 34.343 | +2.30 |
| rician | 28.900 | 30.607 | +1.71 |
| uniform | 31.406 | 34.421 | +3.02 |
| **salt_and_pepper** | 29.692 | **37.979** | **+8.29** |
| **ALL** | **30.510** | **34.337** | **+3.83** |

SSIM: 0.8950 → **0.9427** (+0.048)

**epoch 6배 + LR 스케줄 교체만으로 +3.83 dB.** 모델 변경 없이 얻은 결과다.

### 3.3 Conventional filter 대비

| noise | DnCNN iter2 | 최선 conventional | Δ |
|---|---|---|---|
| gaussian | 34.343 | 30.078 (adaptive 5x5) | +4.27 |
| rician | 30.607 | 27.046 (adaptive 5x5) | +3.56 |
| uniform | 34.421 | 30.521 (adaptive 5x5) | +3.90 |
| salt_and_pepper | 37.979 | 28.496 (median 3x3) | +9.48 |
| **ALL** | **34.337** | 27.202 (adaptive 5x5) | **+7.14** |

iter1 리뷰의 문제 2.3(“S&P 제외하면 adaptive filter 대비 +1.80 dB에 불과”)은 **해소되었다.** S&P 제외 평균 격차가 +1.80 → **+3.91 dB**로 벌어졌다. "왜 딥러닝인가"에 대한 정량적 근거가 확보되었다.

### 3.4 수렴 상태

ep 51~60 validation PSNR: 33.60 / 33.55 / 33.57 / 33.56 / 33.62 / 33.65 / 33.61 / 33.62 / 33.58 / 33.60
→ **표준편차 0.031 dB. 완전히 평평하다.**

train loss: ep46 6.531e-4 → ep60 6.450e-4 (14 epoch에 1.2% 감소)

**이 스케줄에서 epoch 카드는 소진되었다.** 단, LR이 설계상 2e-6까지 내려간 결과이므로 "모델 용량 한계"로 단정할 수는 없다. 120 epoch cosine으로 다시 돌리면 중간 LR 구간이 길어져 추가 이득이 가능하지만, ep46 이후 loss 정체를 보면 **+0.2~0.4 dB 정도**로 예상된다. 비용 대비 효율이 낮다.

---

## 4. 예측 사후 검증 (post-mortem)

학습 중간(ep38 시점)에 수행한 예측과 실제:

| 지표 | 예측(중심) | 예측 범위 | **실제** | 오차 |
|---|---|---|---|---|
| val_L1 plateau 평균 | 33.2 | 33.0 ~ 33.6 | 33.60 | +0.40 (범위 내) |
| val_L1 best | 33.4 | 33.2 ~ 33.7 | 33.65 | +0.25 (범위 내) |
| 최종 train loss | 6.1e-4 | 6.0 ~ 6.3e-4 | 6.45e-4 | +0.35e-4 |
| **test PSNR** | 33.4 | 33.1 ~ 33.8 | **34.337** | **+0.94 (범위 밖)** |
| **test SSIM** | 0.930 | 0.925 ~ 0.936 | **0.9427** | **+0.013 (범위 밖)** |

**validation 예측은 맞았고 test 예측은 틀렸다.**

### 4.1 왜 틀렸나

`test − val_L1` offset을 참조 checkpoint로 실측해 **+0.21 dB**로 가정했다. 실제 iter2 모델에서는 **+0.69 dB**였다.

원인은 **noise별 이득이 균등하지 않았기 때문**이다. 나는 "MSE 도메인에서 균등한 dB 이득"을 가정해 각 noise type에 약 +3 dB를 일괄 적용했다. 실제는:

| noise | 가정 Δ | 실제 Δ |
|---|---|---|
| gaussian | +3 | +2.30 |
| rician | +3 | +1.71 |
| uniform | +3 | +3.02 |
| **salt_and_pepper** | **+3** | **+8.29** |

S&P가 예상의 2.8배로 나왔다. 만약 S&P도 +2.3(gaussian 수준)이었다면 ALL은 32.84였을 것이므로, **S&P 초과 이득이 ALL에 약 +1.5 dB를 기여**했다. 내 오차 0.94 dB는 거의 전부 이 항이다.

### 4.2 S&P가 상전이한 이유

salt & pepper는 **smoothing 문제가 아니라 detection + inpainting 문제**다.

- impulse가 정확히 `img.max()` 또는 `0`이라는 **이산적 값**을 갖는다
- 따라서 네트워크가 "이 픽셀은 오염되었다"를 판별하는 검출기를 학습할 수 있고, 판별에 성공하면 주변에서 거의 완벽히 복원한다
- 성능 천장이 매우 높다 (실측 37.98 dB, SSIM 0.9842 — 거의 무손실)
- iter1(10 epoch, 4,550 step)은 이 검출기를 아직 학습하지 못했고, iter2(60 epoch, 27,300 step)는 학습했다

즉 점진적 개선이 아니라 **능력 획득(phase transition)** 이다. 이런 항은 선형 외삽으로 예측할 수 없다.

### 4.3 놓친 조기 신호

validation PSNR **표준편차**가 계속 증가하고 있었다.

| epoch | 1 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| val PSNR | 26.36 | 30.38 | 31.60 | 32.83 | 33.25 | 33.41 | 33.60 |
| **val std** | **4.03** | 4.94 | 5.85 | 6.25 | 6.58 | 6.63 | **6.77** |

(iter1은 같은 구간에서 4.4~4.5로 거의 일정했다.)

**평균이 오르면서 분산도 함께 커지는 것은 이득이 이질적이라는 신호다.** 일부 이미지(S&P)가 크게 개선되고 나머지는 완만히 개선되면 분포가 넓어진다. 이 신호를 보고 noise별로 분해해서 외삽해야 했다.

### 4.4 다음 예측에 반영할 규칙

1. **ALL 평균으로 외삽하지 말고 noise type별로 분해해서 외삽한다.** 4종은 성질이 다른 문제이고 학습 곡선도 다르다.
2. validation 지표의 **표준편차를 함께 추적한다.** 분산 증가는 이질적 이득의 조기 신호다.
3. `test − val` offset은 모델에 따라 변한다. 이번에 +0.21 → +0.69로 3배 변했다. **offset은 iteration마다 재측정해야 한다.**
4. impulse noise 계열은 상전이형 이득이므로 "아직 학습되지 않았을 가능성"을 항상 상방 리스크로 둔다.

---

## 5. 리스크: 데이터 생성 규칙 위반 가능성 (최우선 확인 필요)

챌린지 규칙 (`ref/parsed_markdown/삼성 DS Project description Day 1.md:257-260`):

```
- For each clean image, you may generate up to 6 dipole kernel orientations.
- For each orientation, you may generate up to 2 noisy images.
- Do not exceed 12 corrupted images per clean image (6 orientations × 2 noises)
```

**iter2는 on-the-fly 노이즈 생성을 60 epoch 반복했으므로, clean 1장당 60개의 서로 다른 corrupted 이미지를 소비했다.** 상한 12의 5배다.

| | corrupted / clean | 규칙(12) 대비 |
|---|---|---|
| 제공된 예제 (10 epoch) | 10 | 이내 |
| iter1 (10 epoch) | 10 | 이내 |
| **iter2 (60 epoch)** | **60** | **5배 초과** |

제공된 예제의 기본값이 `train_epoch = 10`인 것은 우연이 아닐 가능성이 있다. on-the-fly 생성 방식에서 epoch 수가 곧 clean당 corrupted 생성량이 되므로, 10 epoch은 12 상한 안에 들어가는 값이다.

### 해석의 여지

규칙이 "데이터셋 합성"을 규제하는 것인지, "학습 중 augmentation"까지 포함하는 것인지 명확하지 않다. 그러나 **on-the-fly 방식은 실질적으로 60개의 corrupted 변형을 만들어 쓴 것과 동일**하므로, 엄격한 해석에서는 위반이다. 그리고 iter2의 성능 향상(+3.83 dB) 중 상당 부분이 이 추가 데이터 다양성에서 왔을 가능성이 있다.

### 대응 방안 (규칙 준수 + 장기 학습 양립)

**clean 1장당 corrupted를 12개 이하로 사전 생성하고, 모든 epoch이 이 고정 풀에서만 샘플링하도록 한다.**

```python
# DataWrapper.__init__ 에서 clean 당 노이즈 스펙 12개를 미리 확정한다
# (파일명 기반 seed 로 재현 가능하게, 디스크 저장 없이도 무방)
self.variants = {}   # name -> [(noise_type, sigma), ...] 길이 <= 12

# __getitem__ 에서는 그 12개 중 하나만 고른다
spec = self.variants[_name][idx_variant]
noisy = NoiseSimulator(*spec)(label)
```

이렇게 하면 100 epoch을 돌려도 clean당 소비량은 12로 고정된다. 다만 데이터 다양성이 60 → 12로 줄어드는 만큼 **PSNR이 하락할 것으로 예상된다.** 하락폭을 측정하는 것이 iter3의 첫 작업이 되어야 한다.

> 이 항목은 성능 문제가 아니라 **실격 리스크**다. 규칙 해석을 조교/교수에게 확인하는 것이 가장 빠른 해결책이다.

---

## 6. 남은 약점

### 6.1 rician이 최하위 (30.607)

2위 gaussian(34.343)과 **3.74 dB** 차이다. 원인:

- Rician은 **신호 의존적**이며 **양의 bias**를 갖는다: `E[|s + n|] > s`. 신호가 0인 영역에서 `E = σ√(π/2) ≈ 1.25σ`
- σ 범위가 0~0.15로 gaussian(0~0.1)보다 넓다
- residual learning `out = x + f(x)`는 zero-mean 가법 노이즈에 유리한 구조인데, Rician은 그 가정을 위반한다

대응 후보: variance-stabilizing transform (Rician → 근사 Gaussian 변환 후 복원), σ 조건부 입력, 또는 bias 보정항.

### 6.2 σ 조건부 입력 부재

uniform은 입력 29.631 → 출력 34.421로 +4.79 dB인데, gaussian은 27.668 → 34.343으로 +6.68 dB다. 저노이즈 입력에서 상대 이득이 작다. 노이즈 강도를 모델에 알려주지 않기 때문이다 (FFDNet 방식 미적용).

### 6.3 SSIM이 손실 함수에 없다

SSIM 0.9427까지 왔으나 손실은 여전히 L2뿐이다. SSIM도 채점 대상이므로 복합 손실로 추가 이득 여지가 있다.

### 6.4 dipole convolution 미적용 (변함없음)

iter2의 34.337 dB는 **blur가 없는 `g = f + η` 문제에서의 점수**다. 실제 과제는 `g = h*f + η`이며 `ref/`에 dipole simulator가 존재하지 않는다. 이 격차는 iter2에서도 해소되지 않았다.

---

## 7. iter3 권장 우선순위

| # | 항목 | 근거 | 예상 효과 |
|---|---|---|---|
| **0** | **규칙 준수 확인 + clean당 corrupted ≤12 고정 풀 구현** | 5절. 실격 리스크 | PSNR 하락 예상. 하락폭 측정이 목적 |
| 1 | rician 전용 개선 (VST 또는 bias 보정) | 6.1. 최하위 노이즈, 3.74 dB 격차 | rician +1~2 dB |
| 2 | σ 조건부 입력 (FFDNet 방식, in_channels 2) | 6.2 | 전 구간 +0.3~0.8 dB |
| 3 | L1 + SSIM 복합 손실 | 6.3 | SSIM +0.005~0.015 |
| 4 | Noise2Noise / Noise2Void (label-free 가산점) | `iter1_review.md` 4절 | 점수 아닌 가산점 |
| 5 | dipole forward model + measurement consistency | 6.4 | 과제의 실제 난제 |

**epoch 추가 증량은 권장하지 않는다.** 3.4절에서 확인한 대로 이 스케줄에서는 수렴했고, 120 epoch으로 늘려도 +0.2~0.4 dB로 예상되며, 무엇보다 5절의 규칙 리스크를 악화시킨다.

---

## 8. 부록: iter2 코드 변경 내역

| # | 변경 | iter1 | iter2 | 위치 |
|---|---|---|---|---|
| A | 학습 길이 | 10 epoch | 60 epoch | `GeneralConfig.train_epoch` |
| B | LR schedule | exp decay 0.88 | cosine (`lr → lr×0.02`) | `GeneralConfig.lr_schedule`, `lr_min_ratio`, `get_learning_rate()` |
| C | validation 대상 | `*.npy` (100장) | `L1_*.npy` (43장) | `GeneralConfig.valid_data_type`, `Trainer._set_data()` |
| D | ckpt 저장 주기 | 매 epoch | 10 epoch + best | `GeneralConfig.ckpt_interval`, `Trainer._train()` |
| D | png 저장 주기 | 매 epoch | 10 epoch + 마지막 | `GeneralConfig.fig_interval`, `Trainer._valid()` |

`get_learning_rate()`는 `schedule="exp"`로 하위 호환을 유지한다 (검증: ep9에서 3.596e-5, iter1과 동일).

모델 정의, 손실 함수, 노이즈 시뮬레이터, augmentation, batch size, optimizer는 **전혀 변경하지 않았다.**
