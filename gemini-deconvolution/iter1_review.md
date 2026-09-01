# iter1 리뷰 및 iter2 설계

- 대상: `gemini-deconvolution/iter1.ipynb` (Colab 전구간 실행 완료, exec 1~28, 2026-09-01 05:03~05:16)
- 실행 환경: Colab GPU, `ROOT = /content/drive/MyDrive/DS2026/20260831-pjt5-이종호/실습`
- 총 학습 시간: **12분 17초** (30 epoch, 약 24.6초/epoch, validation·figure 저장 포함)
- run dir: `logs_deconvolution/00002_train`
- 과제: deconvolution (`measure = dipole_kernel ⊛ label`), U-Net vs conventional 역필터

---

## 1. 결론

**파이프라인은 정상 완주했지만 iter1 은 실패한 baseline 이다.**

| 방법 | 학습 | PSNR | SSIM |
|---|---|---|---|
| Measure (input) | — | 7.892 ± 1.949 | 0.032 |
| **U-Net (best ckpt)** | 30 epoch / 12분 | **25.860 ± 3.802** | 0.881 |
| TKD (clip 5) | 없음 | 31.187 ± 5.597 | 0.934 |
| Tikhonov (λ=1e-3) | 없음 | 35.377 ± 4.726 | 0.970 |
| **Wiener (K=1e-4)** | **없음** | **42.251 ± 5.667** | **0.988** |

**학습된 0.84 M 파라미터 U-Net 이 한 줄짜리 역필터보다 16.39 dB 나쁘다.** n=100, PSNR 표준편차 3.8 dB
이므로 표준오차는 0.38 dB다. 16.39 dB 격차는 통계적 잡음이 아니라 **구조적 실패**다.

더 나쁜 해석: iter1 자신의 K sweep 을 보면 U-Net(25.86)은 **K 를 100배 잘못 고른 Wiener(K=1e-2, 24.85)** 와
같은 수준이다.

| K | 0.1 | 0.01 | 0.001 | 0.0001 |
|---|---|---|---|---|
| PSNR | 11.896 | 24.847 | 35.377 | 42.251 |

원인은 학습량이 아니라 **문제 설정**이다 (2.1 참조).

---

## 2. 문제점

### 2.1 국소 CNN 에 전역 역연산을 raw measure 로 떠넘겼다 (핵심)

dipole forward 는 k-space 에서 **대각 연산자**다.

```
measure = ifft( fft(label) * H ),   H = 1/3 - ky²/(kx²+ky²+1e-8)
```

`iter1.ipynb` cell 6 출력이 `dipole |H| min 2.27e-05 max 0.667` 이다. 분모의 `+1e-8` 때문에
**magic angle cone 에서도 H 가 정확히 0 이 아니다.** 즉 이 forward model 은 **모든 주파수에서 가역**이고,
정답은 `label = ifft( fft(measure) / H )` 라는 닫힌 해로 존재한다.

그런데 iter1 의 U-Net 은 receptive field 가 유한한 국소 연산자다. `1/H` 는 원점 근처 cone 에서
값이 4.4e+4 까지 치솟는 **전역·비국소** 연산이므로, 국소 CNN 이 raw measure 만 보고 이것을 처음부터
학습하는 것은 가장 불리한 설계다. denoising 과 달리 deconvolution 에서는 forward model 을
**모델 구조에 넣어야** 한다.

### 2.2 conventional 비교가 K=1e-4 에서 멈춰 상한을 놓쳤다

`measure` 에 노이즈가 없으므로(`measure_noise: False`, cell 14 출력) K 를 더 내리면 계속 좋아진다.
동일 forward model 로 로컬 재현한 결과 (`ref/dataset/test_label` 24장, float32, on-the-fly measure):

| K | 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8 | 1e-9 | 0 |
|---|---|---|---|---|---|---|---|---|---|
| PSNR | 25.35 | 35.12 | **41.00** | 47.27 | 54.34 | 62.27 | 71.21 | 80.70 | **107.22** |
| SSIM | 0.914 | 0.959 | 0.980 | 0.993 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |

K=1e-4 열(41.00)이 iter1 의 Colab 실측(42.25, n=100)과 일치하므로 이 표는 신뢰할 수 있다.

**즉 iter1 이 보고한 "최고" 42.25 dB 조차 conventional 방법의 상한이 아니고, 실제 상한은 107 dB 다.**
iter1 은 65 dB 를 그냥 버렸다.

### 2.3 그러나 고정 K 를 그냥 낮추면 위험하다

`Day 1` 은 "noise characteristic is **not** given" 이라고 명시한다. 노이즈가 조금이라도 있으면
K→0 은 즉시 붕괴한다 (동일 실험, 20장):

| σ | 최적 K | 최적 PSNR | K=1e-4 | K=0 |
|---|---|---|---|---|
| 0 | 0 | 107.25 | 41.06 | 107.25 |
| 1e-4 | 3e-6 | **47.07** | 40.86 | **28.86** |
| 5e-4 | 3e-5 | 38.97 | 38.65 | 14.03 |
| 1e-3 | 1e-4 | 35.80 | 35.80 | 11.20 |
| 1e-2 | 3e-3 | 24.69 | 18.55 | 5.13 |

따라서 **K 를 측정값에서 자동 추정**해야 한다. label 없이 가능한 방법이 GCV 다 (부록 5.2).

### 2.4 학습 곡선: 마지막 epoch 까지 상승 중인데 lr 을 먼저 죽였다

| epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| valid PSNR | 16.52 | 22.30 | 24.22 | 25.05 | 25.47 | 25.62 | **25.78** |
| lr | 1.00e-4 | 6.82e-5 | 3.60e-5 | 1.90e-5 | 1.00e-5 | 5.29e-6 | 2.79e-6 |

- best 가 **마지막 epoch(30)** 이다. 수렴하지 않았다.
- 그런데 lr 은 이미 초기값의 **2.8%** 다. `lr_decay=0.88` 지수감쇠는 총 epoch 수와 무관하게 떨어지므로
  epoch 만 늘리면 후반이 전부 낭비된다. **cosine annealing** 으로 바꿔야 한다.
- 학습이 12분밖에 안 걸리므로 epoch 을 3~10배 늘릴 여유는 충분하다.
- 단, epoch 만 늘려서 42 dB 에 도달하는 것은 비현실적이다. 마지막 10 epoch 의 개선폭이 **+0.31 dB** 이고
  포화 중이므로, 남은 16.4 dB 를 학습량으로 메우려면 수백~수천 epoch 이 필요하다. **구조를 바꿔야 한다.**

### 2.5 validation 분포가 test 와 다르다

| split | 총 | L1 | L8 |
|---|---|---|---|
| train | 7,268 | 3,167 | 4,101 |
| val | 100 | **43** | **57** |
| test_label | 100 | **100** | **0** |

best checkpoint 를 **57% 가 L8 인 validation** 으로 고르는데 test 는 전부 L1 이다. 모델 선택 기준이
채점 분포와 어긋난다. `LoaderConfig.data_type` 을 validation 에서만 `"L1_*.npy"` 로 바꾸면 된다
(선택 표준오차는 0.38 → 0.58 dB 로 커지는 trade-off).

### 2.6 노이즈가 0 이라 비교 자체가 degenerate 하다

`measure_noise: False` 이므로 학습·검증·테스트 measure 가 모두 무노이즈다. 이 조건에서는 forward model 을
아는 역필터가 107 dB 로 사실상 정답을 복원하므로 **"왜 딥러닝인가" 라는 질문에 답할 수 없다.**
딥러닝의 존재 이유를 보이려면 노이즈가 있는 구간에서 비교해야 한다.

### 2.7 test measure 가 제공 파일이 아니라 즉석 생성이다

cell 14 출력이 `test_measure_dir: None`, cell 42 출력이 `measure : on-the-fly (ForwardSimulator)` 다.
`dataset/test_deconv_only` 가 Drive 에 없어서 fallback 된 것이다. 즉 **학습과 테스트의 measure 생성 경로가
완전히 동일**하므로 현재 수치는 낙관적으로 편향돼 있다. 채점용 measure 파일이 배포되면 재확인해야 한다.
(디렉토리가 한 단계 중첩된 형태로 배포될 수 있어 iter1 에는 자동 하강 로직을 이미 넣었다.)

---

## 3. 개선 항목 (ROI 순) → iter2

| # | 항목 | 코드량 | 예상 효과 | 근거 |
|---|---|---|---|---|
| **1** | **정규화 세기 K 를 GCV 로 자동 선택** | 함수 2개 | **42.3 → 107 dB, 학습 0** | 2.2, 2.3, 5.2 |
| **2** | **physics 입력 6채널 + residual** (`out = base + s·UNet(x)`) | 모델/입력 | CNN 이 국소 잔차만 담당 | 2.1 |
| **3** | **마지막 conv zero-init** | 3줄 | 초기 출력 = analytic base → 회귀 불가능한 출발점 | — |
| 4 | cosine LR + warmup, epoch↑, AdamW | 함수 1개 | 수렴 개선 | 2.4 |
| 5 | Charbonnier + data-consistency 손실 | loss 1개 | cone 근처 이상치 강건성 | — |
| 6 | dihedral 8× + 노이즈 σ log-uniform 증강 | dataset | 미지 노이즈 방어 | 2.3, 2.6 |
| 7 | validation 을 `L1_*` 로 제한 | 1줄 | 선택 기준 정합 | 2.5 |
| 8 | flip 4× TTA | 함수 1개 | 소폭 (+0.1~0.3 dB) | 5.3 |
| 9 | base / net / net+TTA 자동 선택 | 셀 1개 | 무노이즈 구간 회귀 방지 | 2.6 |
| 10 | 노이즈 σ sweep 으로 "딥러닝이 이기는 구간" 정량화 | 셀 1개 | 발표 근거 | 2.6 |

> 적용 현황: 전 항목 `gemini-deconvolution/iter2.ipynb` 에 반영.

### 3.1 iter2 의 핵심 구조

```
measure ──► GCV 로 K 선택 (label 불필요)
        ├─► base   = Wiener(measure, K)          ← 주 추정치
        ├─► soft   = Wiener(measure, K·10)
        ├─► sharp  = Wiener(measure, K·0.1)
        ├─► grad   = gradTikhonov(measure, K·3)
        └─► log10(K) 상수맵                       ← 노이즈 세기 조건 입력 (FFDNet 방식)
                    │
                    ▼  6채널
              U-Net (chans 24)
                    │
   output = base + 0.1 · UNet(6채널)   ← 마지막 conv zero-init
```

무노이즈 구간에서는 GCV 가 K→1e-14 를 고르고 base 가 이미 107 dB 이므로 네트워크는 "잔차 0 을 출력하는 법"을
배우면 된다. 노이즈 구간에서는 base 가 35 dB 수준으로 떨어지므로 잔차 보정의 여지가 크다.
`log10(K)` 채널이 두 regime 을 구분하는 조건 입력 역할을 한다.

---

## 4. 발표 관점 정리

이 과제의 forward model 은 **가역이고 노이즈가 없으면 역필터가 정답**이다. 따라서
"딥러닝이 conventional 을 이긴다" 는 서술은 이 데이터에서 성립하지 않는다. 정직한 프레이밍은 다음 두 축이다.

1. **무노이즈 구간**: analytic 역필터가 상한(107 dB). 딥러닝은 불필요하고, 실제로 손해다.
   → iter1 의 25.86 dB 는 이 사실을 모른 채 학습만 돌린 결과라는 점을 그대로 보고하는 것이 좋다.
2. **노이즈 구간 (실제 채점 조건은 미공개)**: 고정 K 는 붕괴하고, GCV 는 oracle 에 근접하며,
   노이즈 증강으로 학습된 residual U-Net 이 analytic 을 넘을 수 있다.
   → iter2 의 σ sweep 셀이 교차점을 수치로 제시한다.

즉 기여는 "PSNR 을 올렸다" 가 아니라 **"어느 구간에서 어떤 방법을 써야 하는지를 정량적으로 규정했다"** 가 된다.

---

## 5. 부록: 검증된 사실

모두 iter1 과 동일한 `dipole_kernel` 구현으로 로컬 재현했다 (`ref/dataset`, torch 2.12 CPU, float32).

### 5.1 forward model 특성

- `|H| min 2.269e-05`, `max 0.667` (256×256). **cone 이 정확히 0 이 아니라 가역**이다.
- `|H| < 0.01` 인 주파수 비율 1.59%, `< 0.005` 는 0.81%.
- 무노이즈 measure 에서 `label - Wiener(K=1e-6)` 의 표준편차는 0.0031, 최대 0.0218.

### 5.2 GCV 로 고른 K vs oracle

$$\mathrm{GCV}(R) = \frac{N\lVert H\hat x - y\rVert^2}{(\mathrm{tr}(I-A))^2},\quad A = \frac{H^2}{H^2+R}$$

전부 k-space 대각이라 닫힌 형태로 계산되고 **label 이 필요 없다.**

| σ | oracle K / PSNR | GCV K / PSNR | 차이 |
|---|---|---|---|
| 0 | 1e-12 / 107.44 | 1e-14 / **107.37** | −0.07 |
| 3e-5 | 1e-7 / 53.04 | 1e-7 / **53.49** | +0.45 |
| 1e-4 | 1e-6 / 46.83 | 1e-6 / **46.56** | −0.27 |
| 1e-3 | 1e-4 / 36.35 | 6e-5 / **35.10** | −1.25 |

격자를 `1e-1 ~ 1e-14` 로 넓게 두어도 노이즈가 있으면 GCV 가 내부 최소점을 찾으므로 안전하다.

### 5.3 대칭성 (TTA 유효성)

| 변환 | `max |dipole(T x) − T dipole(x)|` | TTA 사용 |
|---|---|---|
| 좌우 flip | 1.8e-07 | **가능** |
| 상하 flip | 1.2e-07 | **가능** |
| 양방향 flip | 1.8e-07 | **가능** |
| rot90 | **6.7e-01** | 불가 (B0 축이 바뀐다) |

flip 은 kernel 이 `ky²` 에 대해 짝함수이므로 정확히 등변이다(해석적으로도 성립).
단 **학습 증강**에서는 measure 를 label 로부터 즉석 생성하므로 rot90 도 유효한 pair 가 된다 (8× dihedral 가능).

### 5.4 gradient 가중 Tikhonov

`R(k) = λ|k|²` 로 두면 고주파를 더 강하게 억제한다.

| σ | plain Wiener | grad-Tikhonov | Laplacian(`|k|⁴`) |
|---|---|---|---|
| 0 | **80.79** | 77.55 | 79.28 |
| 1e-4 | 47.12 | 48.55 | **48.79** |
| 1e-3 | 35.75 | 38.11 | **38.14** |

무노이즈에서는 손해, 노이즈가 있으면 +1.4~2.4 dB. iter2 는 채널과 baseline 양쪽에 포함했다.

### 5.5 Wiener == Tikhonov

$$\frac{1}{H}\cdot\frac{|H|^2}{|H|^2+K} = \frac{H^*}{|H|^2+K} = \frac{H}{H^2+K}$$

실수 kernel 이므로 두 식이 완전히 동일하다. iter1 에서 두 방법을 별도 열로 병기하면서 `K=λ=1e-4` 로
둔 탓에 값이 문자 그대로 같게 나왔다. 현재는 `_self_check` 에서 `allclose` 로 항등식을 검증하고
`TIKHONOV_LAMBDA=1e-3` 으로 분리해 두었다 (실행 로그의
`note: Wiener(K) == Tikhonov(lambda=K) 임을 확인` 이 이 검증 출력이다).

### 5.6 데이터셋 실측

- 이미지 `256×256`, float32, label 범위 `[0, 1]`
- train 7,268 / val 100 / test_label 100
- `test_label` 100장에 **clean ground truth 가 존재**하므로 로컬 채점이 가능하다
- prefix 분포는 2.5 표 참조

### 5.7 iter1 실행 중 수정된 결함

| 위치 | 문제 | 증상 |
|---|---|---|
| `DataWrapper.__getitem__` | `split("/")[-1]` 로 basename 추출 | Windows 에서 `_name` 이 절대경로 전체가 되어 **비교 그림이 `dataset/train`, `test_label`, `val` 안에 png 로 저장**되고 `valid_png` 는 빈 디렉토리가 됐다 |
| 1-4 PSF 데모 TKD | 복소 OTF 에 `np.clip` | 잘린 주파수의 위상이 소실. 대칭 PSF 3종에서는 무증상이지만 비대칭 PSF 검증 시 42.28 → 48.65 dB 손실 |
| `Trainer._train` | `valid_interval>1` 일 때 stale metric 으로 best 갱신 | 검증되지 않은 가중치가 `checkpoint_best.ckpt` 를 덮어쓴다 |
| `calculate_psnr` | mask 분기 `keepdim` 누락 | `(B,)`×`(B,1,1,1)` 브로드캐스팅으로 `(B,1,1,B)` |
| `wiener_deconv` / `tikhonov_deconv` | 동일 필터를 다른 방법으로 병기 | 5.5 |
| `ForwardSimulator` | 노이즈 4종이 도달 불가 코드 | `DataWrapper` 가 인자 없이 생성 → config 로 배선 |
| config | `test_deconv_only` 중첩 경로 미고려 | 배포 형태에 따라 `FileNotFoundError` |
| `train_epoch_unet` | 미니배치마다 loss 객체 생성 | epoch 1회로 이동 |
| 결과 표시 셀 | `IPython.display.Image` 가 `PIL.Image` 를 가림 | 별칭 처리 |
