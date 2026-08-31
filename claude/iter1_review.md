# iter1 리뷰 및 다음 단계

- 대상: `claude/train_denoising_iter1.ipynb` (Colab 전구간 실행 완료, exec 1~22)
- 실행 환경: Colab GPU, `ROOT = /content/drive/MyDrive/DS2026/20260831-pjt5-이종호/실습`
- 총 학습 시간: **17분 12초** (10 epoch)
- 비교 기준: `ref/log_denoising_example/00012_train` (제공된 참조 학습 결과)

---

## 1. 결론

**파이프라인은 정상 완주했으나 성능 개선은 0이다.** iter1은 제공된 참조 결과의 재현(reproduction)이며 iteration이 아니다.

| noise | iter1 PSNR | 참조 PSNR | Δ | iter1 SSIM | 참조 SSIM |
|---|---|---|---|---|---|
| gaussian | 32.282 | 32.044 | +0.24 | 0.8963 | 0.9032 |
| rician | 28.727 | 28.900 | −0.17 | 0.8632 | 0.8621 |
| uniform | 32.060 | 31.407 | +0.65 | 0.9279 | 0.9167 |
| salt_and_pepper | 29.298 | 29.692 | −0.39 | 0.8915 | 0.8978 |
| **ALL** | **30.592** | **30.511** | **+0.08** | **0.8947** | **0.8950** |

이미지별 PSNR 표준편차가 4.78 dB이므로 noise type별(n=25) 표준오차는 약 0.96 dB이다. 위 Δ는 전부 통계적 잡음 범위 안에 있다.

파이프라인이 올바르게 동작한다는 확인으로서는 의미가 있다. `NoiseSimulator` 구현, Colab 경로, `test_noise_only` 중첩 경로, best checkpoint 선택, conventional filter 비교, 그림 저장이 모두 정상 작동함이 입증되었다.

---

## 2. 문제점

### 2.1 학습이 명백히 부족하다

Validation PSNR 추이:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| val PSNR | 26.72 | 27.10 | 28.87 | 29.07 | 28.99 | 30.10 | 30.09 | **30.60** | 30.27 | 30.57 |

마지막 epoch까지 상승 추세가 유지된다. best가 epoch 8인 것은 수렴이 아니라 **상승 중인 곡선의 요동**이다.

그런데 총 학습 시간이 17분 12초(정상 상태 약 78초/epoch)다. Colab 세션 한 번에 60 epoch(약 80분), 100 epoch(약 130분)도 충분히 가능하다. `train_epoch = 10`은 제공된 예제의 기본값일 뿐이고 그대로 쓸 근거가 없다.

### 2.2 LR 스케줄이 학습을 조기에 죽인다

`lr_decay = 0.88`, `lr_tol = 1`이므로 LR은 `lr * 0.88^(epoch-1)`로 감쇠한다.

| epoch | 1 | 4 | 7 | 10 |
|---|---|---|---|---|
| LR | 1.00e-04 | 7.74e-05 | 5.28e-05 | 3.60e-05 |

epoch 10에서 이미 초기값의 36%다. **성능이 아직 오르는 중에 LR을 먼저 줄이고 있다.**

이 지수 감쇠는 총 epoch 수와 무관하게 동작하므로, epoch만 늘리면 `0.88^59 ≈ 6e-4`배까지 떨어져 후반이 전부 낭비된다. epoch을 늘리려면 스케줄을 **총 epoch에 맞춰 감쇠하는 cosine annealing**으로 바꿔야 한다.

### 2.3 593k 파라미터 네트워크가 5x5 adaptive filter를 거의 못 이긴다

| noise | DnCNN | Adaptive 5x5 | Δ |
|---|---|---|---|
| gaussian | 32.282 | 30.078 | +2.20 |
| rician | 28.727 | 27.046 | +1.68 |
| uniform | 32.060 | 30.521 | +1.54 |
| salt_and_pepper | 29.298 | 21.164 | **+8.13** |
| **ALL** | **30.592** | **27.202** | **+3.39** |

**salt & pepper를 제외하면 평균 +1.80 dB에 불과하다.** 딥러닝의 실질 기여가 impulse noise 처리에 편중되어 있다. 발표에서 "왜 딥러닝인가"를 정당화하려면 이 격차를 벌려야 한다.

특히 uniform은 입력이 이미 29.63인데 출력이 32.06으로 **+2.43 dB**다. 저노이즈 입력에서 이득이 작다는 것은 네트워크가 노이즈 강도와 무관하게 획일적으로 동작한다는 신호다. σ 조건부 입력(FFDNet 방식)이 필요하다.

### 2.4 Validation 분포가 test와 다르다

파일명 prefix 분포:

| split | L1 | L8 |
|---|---|---|
| train | 3,167 | 4,101 |
| **val** | **43** | **57** |
| **test_label** | **100** | **0** |

L1과 L8은 통계가 유의하게 다르다 (각 60장 샘플):

| subset | mean | std | gradient 평균 | 어두운 픽셀 비율(<0.1) |
|---|---|---|---|---|
| L1 | 0.476 | 0.229 | **0.0862** | 0.061 |
| L8 | 0.412 | 0.281 | **0.0629** | 0.163 |

L1은 고주파 성분이 L8보다 **37% 많은** 어려운 subset이고, **test는 전부 L1**이다. 그런데 best checkpoint를 **57%가 L8인 validation**의 PSNR로 선택하고 있다. 모델 선택 기준이 실제 채점 분포와 어긋난다.

수정은 `LoaderConfig.data_type`을 validation에서만 `"L1_*.npy"`로 바꾸면 된다. 단, validation이 100 → 43장으로 줄어 선택의 표준오차가 0.48 → 0.73 dB로 커지는 trade-off가 있다.

### 2.5 dipole convolution이 여전히 없다

iter1은 `g = f + η`만 다룬다. 실제 과제는 `g = h*f + η`이다 (`ref/parsed_markdown/삼성 DS Project description Day 1.md:216-217`).

이는 iter1의 결함이라기보다 미해결 과제이지만, **PSNR 30.6은 blur가 없는 문제에서의 점수**라는 점을 인지해야 한다. 실제 채점 test에 dipole blur가 걸려 있으면 이 모델은 무너진다. 또한 `ref/`에 dipole simulator 파일이 존재하지 않는다 (강의자료는 "model is given"이라고만 기술).

---

## 3. 개선 항목 (ROI 순)

| # | 항목 | 코드 변경량 | 예상 시간 | 목적 |
|---|---|---|---|---|
| **1** | **epoch 60 + cosine LR + validation L1 한정** | 설정 + 함수 1개 | 약 1.5h | supervised 기준선 확보 |
| 2 | Noise2Noise (L1 손실) | 로더 5줄 | 약 3h | label-free 가산점 |
| 3 | Noise2Void (test 100장 학습) | 마스킹 구현 | 약 2h | "진짜 label-free" 시연 |
| 4 | Test-Time Adaptation (supervised + N2V) | 손실 합산 | 약 3h | 미지 노이즈 일반화 방어 |
| 5 | σ 추정치를 입력 채널로 추가 (FFDNet) | 모델 in_channels 2 | 약 3h | 저노이즈 구간 이득 회복 |
| 6 | L1 + SSIM 복합 손실 | 손실 함수 | 약 3h | SSIM도 채점 대상인데 현재 L2만 사용 |
| 7 | 랜덤 crop 128 + 90° 회전 | augment 함수 | 약 3h | 유효 데이터량 증가 |
| 8 | dipole forward model + measurement consistency | 신규 구현 | — | 과제의 실제 난제 |

> **적용 현황**: 항목 1만 `claude/iter2.ipynb`에 적용됨. 상세는 해당 노트북 첫 markdown 셀 참조.

---

## 4. Semi-supervised / Unsupervised 접근법

### 4.1 전제

이 과제는 forward model을 직접 합성할 수 있으므로 **label-free가 supervised를 PSNR로 이길 가능성은 거의 없다.** 강의자료도 "**partial** bonus consideration"이라고만 기술한다 (`Day 1.md:274`).

따라서 목표를 둘로 분리해야 한다.

- **A. 가산점용**: label-free 파이프라인을 별도 브랜치로 구현하고, supervised와의 격차를 *왜 그런지* 정량 분석
- **B. 실제 점수용**: `Day 1.md:218`이 "noise characteristic is **not** given"이라고 명시한다. 즉 **채점 test의 노이즈가 시뮬레이션한 4종과 다를 수 있다.** 이 리스크를 막는 유일한 수단이 준지도 학습(test-time adaptation)이다. 이것이 가산점보다 중요하다.

### 4.2 Noise2Noise — 가장 잘 맞고, 규칙이 이미 허락한다

챌린지 규칙 "각 방향당 최대 2개의 Noisy 이미지"(`Day 1.md:259`)는 정확히 N2N 설정이다. 같은 (clean, orientation)에 독립적인 노이즈를 두 번 걸어 `noisy_a → noisy_b`로 학습한다. **clean 이미지를 target으로 한 번도 쓰지 않으므로 label-free다.**

이론: `argmin E‖f(x_a) − x_b‖²`는 `E[x_b | x_a]`로 수렴하고, 노이즈가 zero-mean이면 이것이 clean이다.

노이즈별 유효성:

| noise | zero-mean | L2 손실 | L1 손실 |
|---|---|---|---|
| gaussian | O | 정확 | 약간 손실 |
| uniform | O | 정확 | 약간 손실 |
| rician | **X (양의 bias)** | bias 잔존 | 개선 |
| salt_and_pepper | **X** | outlier에 끌려감 | **정확 (median)** |

→ **L1 손실을 써야 한다.** L1은 조건부 median으로 수렴하므로 impulse noise에 강건하고 Rician bias도 완화한다.

구현: `DataWrapper.__getitem__`에서 label 대신 노이즈를 두 번 뽑아 반환하면 끝이며, 나머지 파이프라인은 손댈 필요가 없다.

```python
# noisy_a 를 입력, noisy_b 를 target 으로 (clean 미사용)
noisy_a = self.noise_simulator(label)
noisy_b = self.noise_simulator(label)
return noisy_b, noisy_a, _name   # (target, input, name)
```

예상: gaussian/uniform은 supervised 대비 −0.3~0.7 dB, rician/S&P는 더 벌어진다. **그 격차 자체가 위 표로 설명되는 좋은 발표 소재다.**

### 4.3 Noise2Void (blind-spot) — "진짜 label-free" 주장이 가장 강력

test noisy 100장**만으로** 학습할 수 있다. clean도, pair도 필요 없다. 가산점 관점에서 가장 임팩트 있는 시연이다.

원리: 입력에서 일부 픽셀을 마스킹(주변 픽셀 값으로 교체)하고 그 위치에서만 손실을 계산한다. 네트워크가 자기 자신을 복사하는 자명해를 막는다.

이 문제에서의 명확한 실패 모드 두 가지 (정량화하면 좋은 분석이 된다):

- **Rician bias**: blind-spot은 `E[noisy | 주변]`을 추정하는데 Rician은 `E[|s + n|] > s`다. 신호가 0인 영역에서 `E = σ√(π/2) ≈ 1.25σ`이고, σ = 0.15면 **0.19의 계통 오차**가 [0,1] 이미지에 남는다. 어두운 영역이 들뜬다.
- **salt & pepper**: 마스킹된 위치의 MSE가 impulse에 지배된다. L1 손실이 필수다.

또한 N2V는 **blur를 제거하지 못한다.** blur는 신호의 공간 상관이므로 blind-spot 가정 위반은 아니지만, 그냥 손대지 않고 통과시킨다. dipole 단계에서는 무력하다.

### 4.4 Neighbor2Neighbor — 이 데이터에서는 권장하지 않음

N2B는 인접 픽셀 sub-sampling으로 pair를 만든다. 반도체 레이아웃은 **주기적 고주파 패턴**이라 2배 sub-sampling에서 aliasing이 발생한다. L1 subset의 gradient 에너지가 0.086인 점을 보면 실제로 위험하다. N2N이 이미 가능한 상황에서 열등한 대체재를 쓸 이유가 없다.

> `deepseek/plan.md:56`은 Neighbor2Neighbor를 비중 있게 배치했으나, 이 데이터 특성상 우선순위를 낮춰야 한다.

### 4.5 Semi-supervised: Test-Time Adaptation — 실제 점수에 가장 중요

`Day 1.md:218`의 "noise characteristic is not given"에 대한 유일한 방어책이다.

```
L = L_supervised(합성 pair) + λ · L_selfsup(test noisy 100장)
```

- `L_supervised`: 현재 하고 있는 것
- `L_selfsup`: test 100장에 대한 N2V blind-spot 손실 (라벨 불필요)

두 손실을 함께 최적화하면 시뮬레이션한 4종과 실제 노이즈가 달라도 모델이 실제 분포로 끌려간다. λ는 0.1~0.5부터 시작한다.

구현 난이도는 N2V를 만든 뒤라면 낮다(로더 두 개, 손실 합산). **일반화 리스크 대비 효과는 4.2~4.4보다 크다.**

### 4.6 Label-free deconvolution (physics-informed) — dipole 확보 후의 본게임

`h`를 알면 clean label 없이 deconvolution을 학습할 수 있다.

```
L = ‖ h ⊛ f_θ(g) − g ‖²  +  λ · TV( f_θ(g) )
```

측정 일치성(measurement consistency) 항만으로 label-free이며, `h`의 zero-cone 때문에 생기는 null-space는 TV 정규화가 채운다 (`Day 2.md:78`, `:122`). Day 3의 PINN 내용과 정확히 맞물리고, "왜 이 방법인가"에 대한 답이 물리 모델에서 직접 도출된다.

### 4.7 권장 실행 순서

| 순서 | 작업 | 비용 | 목적 |
|---|---|---|---|
| 1 | epoch↑ + cosine LR + val L1 필터 | 코드 5줄 / 1.5h | supervised 기준선 확보 |
| 2 | Noise2Noise (L1 손실) | 로더 5줄 / 3h | 가산점, 최소 노력 최대 효과 |
| 3 | Noise2Void on test 100장 | 마스킹 구현 / 2h | "진짜 label-free" 시연 |
| 4 | TTA (supervised + N2V 동시) | 손실 합산 / 3h | 미지 노이즈 일반화 방어 |
| 5 | dipole forward model + measurement consistency | 신규 구현 | 과제의 실제 난제 |

2~4는 전부 현재 파이프라인 재사용이라 신규 코드가 거의 없다.

발표 자료에는 **supervised / N2N / N2V를 noise type별로 나눈 표**를 넣고, Rician bias(`σ√(π/2)`)와 S&P 비대칭성으로 격차를 설명하는 것이 좋다. 한계를 정량적으로 짚은 분석이 PSNR 0.3 dB 개선보다 평가에서 유리하다.

---

## 5. 부록: 검증된 사실

### 5.1 노이즈 생성기 역검증

`ref/dataset/test_label`(clean) + `test_noise_only`(noisy) + `noise_meta.json`(선언된 type·σ)을 대조하여 `NoiseSimulator`의 4개 수식이 실제 생성기와 일치함을 확인했다 (`claude/verify_noise.py`).

| noise | n | 선언 σ 평균 | 추정 σ | max 오차 |
|---|---|---|---|---|
| gaussian | 25 | 0.05291 | 0.05289 | 0.00042 |
| rician | 25 | 0.07581 | 0.07594 | 0.00191 |
| uniform | 25 | 0.08568 | 0.08571 | 0.00106 |
| salt_and_pepper | 25 | 0.08357 | 0.07901 | 0.01714 |

salt & pepper만 σ 대비 93~95%인데, `torch.randint`의 **중복 좌표 샘플링**(복원추출) 때문이다. `1 − exp(−σ/2)` 예측값과 일치하므로 구현이 정확하다.

**생성기는 clipping을 하지 않는다** (gaussian noisy 실측 범위 `[−0.298, 1.359]`). 따라서 `NoiseSimulator`에도 clamp를 넣지 않았다. 넣으면 train/test 분포가 어긋난다.

### 5.2 데이터셋 실측

- 이미지: `256 × 256`, `float32`, clean 값 범위 `[0, 1]`
- train 7,268장 + val 100장 = **7,368장** (강의자료 `Day 1.md:215`와 일치)
- `test_label/` 100장에 **clean ground truth가 존재**한다. 실습 데이터셋은 로컬 채점이 가능하다.
- `test_noise_only/`의 실제 npy는 `test_noise_only/test_noise_only/`로 한 단계 더 깊다.
- noise type은 4종 각 25장씩 균등 분포.

### 5.3 원본 노트북의 결함 (iter1에서 수정됨)

| 위치 | 문제 |
|---|---|
| `NoiseSimulator` | **클래스 자체가 정의되어 있지 않았다.** `RandomNoiseSimulator.__call__`이 호출하고 `DataWrapper`가 타입 힌트로 참조하는데 본체가 없어 `NameError`로 학습이 시작조차 못 했다. |
| DataWrapper | `self.file_list[idx].split("/")[-1]` — Windows 경로(`\`)에서 basename 추출 실패 |
| DataWrapper | `noisy_type != "random"`이면 `self.noise_simulator`가 미설정 → `AttributeError` |
| Config | `test_noise_only` 중첩 경로 미반영 |
| Config | ROOT가 Colab Drive 경로로 하드코딩 |
| cell 1 | 로컬에서 `google.colab` import 시 `ImportError` |
