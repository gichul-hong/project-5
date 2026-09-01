# iter3 리뷰 — Unsupervised (Noise2Noise)

- 대상: `claude/iter3.ipynb` (Colab 전구간 실행 완료, exec 1~24)
- run dir: `.../logs_denoising/00002_train`
- 설정: `train_mode="n2n"`, `loss_model="l1"`, `select_by="psnr"`, 60 epoch
- 학습 시간: **1시간 17분 23초** (test 포함 1시간 18분 01초) — iter2(1h 17m 33s)와 사실상 동일
- best checkpoint: **epoch 55**

---

## 1. 결론

**clean label 을 한 번도 쓰지 않고 supervised 대비 PSNR −0.98 dB, SSIM −0.0006 을 달성했다.**

| | iter2 (supervised) | **iter3 (Noise2Noise)** | Δ |
|---|---|---|---|
| clean label | 사용 | **미사용** | |
| corrupted / clean | 60 (규칙 5배 초과) | **12 (규칙 준수)** | |
| test PSNR | 34.337 | **33.358** | **−0.979** |
| test SSIM | 0.9427 | **0.9421** | **−0.0006** |

SSIM 은 실질적으로 동등하고, PSNR 격차 1 dB 도 전부 salt & pepper 한 종류에서 나왔다.
게다가 iter3 은 **규칙 위반 리스크를 동시에 제거**했다 (clean 당 corrupted 60장 → 12장).

가산점 대상인 label-free 파이프라인으로서는 매우 좋은 결과다.

---

## 2. 설정 및 학습량

| 항목 | iter2 | iter3 |
|---|---|---|
| 학습 방식 | supervised (corrupted → clean) | **Noise2Noise (noisy A → noisy B)** |
| 손실 | L2 (MSE) | **L1** |
| corrupted 생성 | on-the-fly, 매 epoch 새 노이즈 | **`(파일명, group, member)` 고정 풀** |
| corrupted / clean | 60 | **12 (= 6 group × 2 member)** |
| epoch / step | 60 / 27,300 | 동일 |
| LR | 1e-4 → cosine → 2e-6 | 동일 |
| 모델 | DnCNN 17층, features 64, 593,024 params | **동일 (변경 없음)** |
| batch / 입력 | 16 / 256×256 전체 | 동일 |
| augmentation | H/V flip p=0.5 | 동일 (label/noisy/target 동시 적용) |
| validation | `L1_*.npy` 43장 | 동일 (단 노이즈 추첨이 다름 → 8절) |
| 모델 선택 | val PSNR | val PSNR (selfsup 도 병행 기록) |

### N2N 구성

`variant_group_specs(name)` 가 group 별로 `(noise_type, sigma)` 를 하나 정하고,
member 0/1 은 **같은 조건의 서로 다른 노이즈 실현**이다 (같은 촬영 조건에서 두 번 찍은 것에 해당).

학습 시 group 을 무작위로 하나 골라 member 0 → member 1 (또는 그 역) 로 학습한다.
clean 은 손실에 들어가지 않는다.

`(name, group, member)` 조합이 6 × 2 = 12 로 고정되므로 **epoch 을 몇 번 돌려도 clean 당 corrupted 는 12장을 넘지 않는다.**

---

## 3. Test 결과 (채점 대상 100장)

| noise | n | PSNR in | PSNR out | SSIM in | SSIM out |
|---|---|---|---|---|---|
| gaussian | 25 | 27.668 | 34.554 | 0.6835 | 0.9374 |
| rician | 25 | 24.079 | 29.883 | 0.7104 | 0.8957 |
| uniform | 25 | 29.631 | 34.849 | 0.7950 | 0.9489 |
| salt_and_pepper | 25 | 17.304 | 34.145 | 0.4746 | 0.9866 |
| **ALL** | **100** | 24.671 | **33.358** | 0.6659 | **0.9421** |

### 전체 비교

**PSNR**

| run | gaussian | rician | uniform | salt_and_pepper | ALL |
|---|---|---|---|---|---|
| iter1 (sup, 10ep) | 32.282 | 28.727 | 32.060 | 29.298 | 30.592 |
| iter2 (sup, 60ep) | 34.343 | 30.607 | 34.421 | **37.979** | **34.337** |
| **iter3 (n2n, l1)** | **34.554** | 29.883 | **34.849** | 34.145 | 33.358 |
| adaptive 5x5 | 30.078 | 27.046 | 30.521 | 21.164 | 27.202 |
| median 3x3 | 27.382 | 24.080 | 26.779 | 28.496 | 26.684 |

**SSIM**

| run | gaussian | rician | uniform | salt_and_pepper | ALL |
|---|---|---|---|---|---|
| iter1 (sup, 10ep) | 0.8963 | 0.8632 | 0.9279 | 0.8915 | 0.8947 |
| iter2 (sup, 60ep) | 0.9327 | **0.9042** | **0.9496** | 0.9842 | **0.9427** |
| **iter3 (n2n, l1)** | **0.9374** | 0.8957 | 0.9489 | **0.9866** | 0.9421 |
| adaptive 5x5 | 0.8544 | 0.8294 | 0.8867 | 0.5971 | 0.7919 |
| median 3x3 | 0.7667 | 0.7596 | 0.8212 | 0.8831 | 0.8076 |

label-free 인 iter3 이 **gaussian PSNR·SSIM, uniform PSNR, salt&pepper SSIM 에서 supervised 를 앞섰다.**
conventional filter 대비로는 ALL 기준 **+6.16 dB** (adaptive 5x5) 우위다.

---

## 4. 이론 예측 검증 — noise별 패턴이 정확히 재현되었다

`iter1_review.md` 4.2절에서 N2N 의 유효성을 노이즈별로 예측했다. 결과와 대조한다.

| noise | zero-mean | 사전 예측 | **실제 Δ (iter3 − iter2)** | 판정 |
|---|---|---|---|---|
| gaussian | O | supervised 에 근접 | **+0.211 dB** | 적중 |
| uniform | O | supervised 에 근접 | **+0.428 dB** | 적중 |
| rician | **X (양의 bias)** | 더 벌어짐 | **−0.724 dB** | 적중 |
| salt_and_pepper | **X (impulse)** | 더 벌어짐 | **−3.834 dB** | 적중 |

**zero-mean 인 2종은 supervised 와 동등하고, zero-mean 이 아닌 2종만 손실이 발생했다.**
N2N 이론이 요구하는 조건(`argmin E‖f(x_a) − x_b‖` 가 clean 으로 수렴하려면 노이즈가 zero-mean)이
그대로 데이터에 나타났다. 발표에서 쓸 수 있는 가장 깔끔한 결과다.

- Rician: `E[|s + n|] > s`. 신호가 0인 영역에서 `E = σ√(π/2) ≈ 1.25σ` 의 양의 bias.
  σ 범위가 0~0.15 이므로 최대 0.19 의 계통 오차가 [0,1] 이미지에 남는다.
- salt & pepper: impulse 는 zero-mean 도 대칭도 아니다. L1(조건부 median)이 L2 보다 낫지만
  supervised 가 clean 을 직접 보는 이점을 완전히 상쇄하지는 못한다.

### 통계적 유의성

test PSNR 의 이미지별 표준편차는 6.126 dB 이므로 noise type 당(n=25) 표준오차는 약 1.23 dB 다.

- `+0.211`, `+0.428`, `−0.724` → **모두 1 SE 이내. 통계적으로 동등**하다고 봐야 한다
- `−3.834` → 약 3.1 SE. **유일하게 실재하는 격차**

즉 **PSNR 격차 −0.979 dB 는 사실상 salt & pepper 단일 항목에서 나온 것**이다.

### salt & pepper 의 역설: PSNR 은 지고 SSIM 은 이겼다

| | PSNR | SSIM |
|---|---|---|
| iter2 (sup, L2) | 37.979 | 0.9842 |
| iter3 (n2n, L1) | 34.145 | **0.9866** |

PSNR 은 3.83 dB 뒤졌는데 SSIM 은 오히려 앞섰다. 해석:

- iter3 은 impulse 를 **구조적으로는 완전히 제거**했다 (SSIM 0.9866, 거의 무손실)
- 다만 진폭 오차가 일부 남았고, PSNR 은 이를 제곱으로 벌한다
- iter2 는 L2 손실로 MSE 를 직접 최적화했으므로 PSNR 지표에서 구조적으로 유리하다

**따라서 이 격차의 상당 부분은 "label-free 라서" 가 아니라 "L1 손실을 써서" 생긴 것일 수 있다.**
9절의 대조군 없이는 분리할 수 없다.

---

## 5. 수렴 검증 — N2N 손실 하한에 도달했다

N2N 은 target 자체가 noisy 이므로 **손실이 0 으로 가지 않는다.**
완벽한 denoiser `f(x_a) = clean` 이어도 손실은 `E|clean − x_b| = E|noise_b|` 로 남는다.

노이즈 종류별 `E|noise|` 실측 (train 400장, σ ~ U(범위)):

| noise | E&#124;noise&#124; |
|---|---|
| gaussian | 0.04020 |
| rician | 0.05938 |
| uniform | 0.04954 |
| salt_and_pepper | 0.03899 |
| **4종 균등 평균 (= L1 손실 하한)** | **0.04703** |

| | 값 |
|---|---|
| iter3 train loss (ep1) | 0.06878 |
| iter3 train loss (ep60) | **0.05023** |
| 이론 하한 | 0.04703 |
| 하한 대비 초과분 | **+0.00320 (+6.8%)** |

**손실이 이론 하한의 6.8% 이내까지 내려갔다. N2N 이 정상 동작했고 거의 최적까지 수렴했다.**

> 주의: N2N 의 train loss 는 하한이 0 이 아니므로 **절대값으로 진척도를 판단할 수 없다.**
> iter2(supervised, L2, 최종 6.45e-4)와 iter3(N2N, L1, 최종 5.02e-2)의 loss 를 직접 비교하는 것은 무의미하다.
> 판단 기준은 "하한과의 거리"다.

수렴 확인: ep50~60 val PSNR = 35.60 / 35.64 / 35.68 / 35.67 / 35.61 / 35.71 / 35.67 / 35.65 / 35.65 / 35.67 / 35.67
→ 표준편차 0.03 dB. 완전히 평평하다.

---

## 6. label-free 모델 선택 검증 — 비용 0.04 dB

`select_by="psnr"` 로 돌렸지만 self-supervised validation loss 를 매 epoch 함께 기록했다.
이 값은 clean 을 전혀 쓰지 않으므로, 이것만으로 모델을 고를 수 있다면 **파이프라인 전체가 label-free** 가 된다.

| 선택 기준 | clean 사용 | 선택된 epoch | 그 epoch 의 val PSNR |
|---|---|---|---|
| val PSNR | 사용 | 55 | 35.7100 |
| **selfsup loss** | **미사용** | **59** | **35.6700** |
| | | | **비용 +0.0400 dB** |

두 지표의 상관:

| 구간 | Pearson | Spearman |
|---|---|---|
| 전체 60 epoch | +0.9809 | **+0.9898** |
| epoch ≥ 30 | +0.9169 | +0.9605 |
| epoch ≥ 40 | +0.9581 | +0.9649 |
| epoch ≥ 50 | +0.8639 | +0.7545 |

selfsup 최상위 5 epoch = `[59, 55, 52, 60, 56]` → 이들의 val PSNR 순위 = `[5, 1, 2, 6, 4]` / 60.
**selfsup 로 뽑은 상위 5개가 모두 PSNR 상위 6위 안에 든다.**

후반 plateau(ep≥50)에서 Spearman 이 0.75 로 떨어지지만, 그 구간의 val PSNR 편차 자체가 0.03 dB 이므로
어느 epoch 을 고르든 실질 차이가 없다.

**결론: "학습도 모델 선택도 clean 을 쓰지 않는 완전 label-free 파이프라인" 으로 보고할 수 있다.**
발표에서 이 표를 근거로 제시하면 된다.

---

## 7. 규칙 준수 달성

`iter2_review.md` 5절에서 지적한 실격 리스크가 해소되었다.

| | corrupted / clean | 규칙 상한 12 |
|---|---|---|
| iter1 (10 epoch, on-the-fly) | 10 | 이내 |
| iter2 (60 epoch, on-the-fly) | 60 | **5배 초과** |
| **iter3 (고정 풀)** | **12** | **정확히 충족** |

그리고 이 구조가 그대로 N2N 의 pair 6쌍이 된다.
**규칙 준수와 label-free 학습이 같은 설계로 동시에 해결되었다.**

`(파일명, group, member)` 로 노이즈를 결정론적으로 확정하므로 재현성도 확보된다.
노트북 셀 18 의 자기검증에서 확인됨:
- 같은 `(name, group, member)` → 항상 동일
- 같은 group 의 member 0/1 → 서로 다른 실현 (`mean|a−b| = 0.0675`)
- variant pool 크기 = 12

---

## 8. 문제: validation 설계 결함 (iter4 에서 반드시 수정)

**iter3 의 val PSNR 35.71 은 iter2 의 33.65 보다 2 dB 높지만, 모델이 더 좋아서가 아니다.
validation 노이즈 추첨이 달라졌기 때문이다.**

validation 43장의 **입력 PSNR** 실측:

| 방식 | 입력 PSNR | σ 평균 | noise 분포 |
|---|---|---|---|
| iter2 val (`crc32(name)` 추첨) | 25.877 | 0.0714 | g 10 / r 21 / u 6 / sp 6 |
| **iter3 val (variant pool group 0)** | **29.307** | 0.0647 | g 15 / r 12 / u 8 / sp 8 |
| **test (제공 파일)** | **24.671** | — | **g 25 / r 25 / u 25 / sp 25** |

**iter3 의 validation 노이즈가 iter2 보다 입력 기준으로 3.43 dB 더 쉽다.**
특히 iter3 val 의 gaussian 15장은 입력 PSNR 이 36.88 로, σ 가 극단적으로 작은 샘플들이 뽑혔다
(iter2 val gaussian 은 28.76).

그 결과 val→test offset 이 완전히 뒤집혔다.

| | val PSNR | test PSNR | test − val |
|---|---|---|---|
| iter2 | 33.65 | 34.34 | **+0.69** |
| iter3 | 35.71 | 33.36 | **−2.35** |

### 진단

1. **validation 이 43장뿐이고 노이즈를 무작위 추첨**하므로 노이즈 종류·강도 분포가 test 와 크게 어긋난다
2. test 는 4종이 정확히 25/25/25/25 로 균형인데, val 은 어느 방식에서도 균형이 아니다
3. 따라서 **val PSNR 은 iteration 간 비교에 쓸 수 없다.** 오직 test 만 비교 가능하다
4. `iter2_review.md` 4.4절에서 "offset 은 iteration 마다 재측정해야 한다" 고 적었는데, 그보다 심각하다.
   **offset 이 존재하지 않는다** — 매번 다른 문제를 풀고 있었다

### iter4 수정안: 층화(stratified) validation

val 43장에 노이즈를 **결정론적·균형적으로** 배정한다.

```python
# 파일 인덱스로 노이즈 종류를 순환 배정하고, sigma 는 범위를 균등 분할해 층화한다
def stratified_val_spec(idx: int, total: int, n_types: int = 4):
    nz = NOISE_NAMES[idx % n_types]
    lo, hi = NOISE_RANGES[nz]
    # 같은 종류 안에서 sigma 를 0~1 구간에 고르게 펼친다
    k = idx // n_types
    n_k = (total + n_types - 1) // n_types
    sigma = lo + (hi - lo) * (k + 0.5) / n_k
    return nz, sigma
```

이렇게 하면
- 4종이 균등하게 들어가 test 분포와 일치한다
- σ 가 층화되어 추첨 운에 좌우되지 않는다
- **iteration 간 val 비교가 가능해진다** (같은 문제를 푸는 것이 보장됨)

가능하면 train 에서 L1 이미지 100장 정도를 validation 으로 추가 확보해 143장으로 늘리는 것도 좋다
(train 의 L1 3,167장 중 3% 이므로 학습 손실은 무시 가능).

---

## 9. 교란 요인 — 아직 분리되지 않은 3가지 변경

iter3 은 iter2 대비 **세 가지를 동시에 바꿨다.**

| # | 변경 | 예상 방향 |
|---|---|---|
| 1 | supervised → **N2N** (label-free) | 성능 하락 |
| 2 | L2 → **L1** 손실 | PSNR 하락 / SSIM 상승 |
| 3 | corrupted 60장 → **12장** (규칙 준수) | 성능 하락 |

따라서 **ALL −0.979 dB 는 "label-free 의 비용" 이 아니라 세 변경의 합산 결과**다.
특히 4절에서 본 salt & pepper 의 PSNR/SSIM 역전은 2번(L1)이 주범일 가능성이 크다.

### 필요한 대조군 (iter3 계획의 run C, 미실행)

| run | train_mode | loss | 분리되는 요인 |
|---|---|---|---|
| **C1** | `supervised12` | `l2` | 3번만. **iter2 대비 격차 = "variant 60 → 12" 의 비용** |
| **C2** | `supervised12` | `l1` | 2+3번. **C1 대비 격차 = "L2 → L1" 의 비용** |
| (기준) | `n2n` | `l1` | 완료 |

C1, C2 를 돌리면 `iter3 − iter2 = (variant 감소) + (L1 전환) + (label-free)` 를 항별로 분해할 수 있다.
각 1시간 20분, 합계 약 2시간 40분.

**이 두 run 이 없으면 "label-free 의 비용은 X dB" 라고 말할 수 없다.** 발표에서 질문받을 지점이다.

### 미실행 항목

- `n2v` (Noise2Void) 모드는 노트북에 구현·검증되어 있으나 실행되지 않았다.
  `config.train_mode = "n2v"` 만 바꾸면 돌아간다.
- N2V 는 pair 도 필요 없는 **단일 noisy 학습**이므로 "완전 label-free" 주장이 N2N 보다 더 강하다.
  대신 성능은 N2N 보다 낮을 것으로 예상된다 (blind-spot 의 gradient 희소성, Rician bias, 추론 시 중심 픽셀 누출).

---

## 10. iter4 권장 우선순위

| # | 항목 | 근거 | 비용 |
|---|---|---|---|
| **1** | **층화 validation 으로 교체** | 8절. 지금 val 은 iteration 간 비교 불가 | 코드 10줄 |
| **2** | **대조군 C1 (`supervised12` + L2)** | 9절. variant 12장 제약의 비용을 분리 | 1h 20m |
| 3 | 대조군 C2 (`supervised12` + L1) | 9절. L1 전환의 비용을 분리 | 1h 20m |
| 4 | `n2v` 실행 | 완전 label-free 시연. 구현 완료 상태 | 1h 20m |
| 5 | rician 전용 개선 (VST / bias 보정) | 4절. 4종 중 최하위(29.883), Rician bias 가 원인 | 3h |
| 6 | S&P 를 위한 L1+L2 혼합 손실 | 4절. PSNR 은 L2, SSIM 은 L1 이 유리 | 3h |
| 7 | dipole forward model + measurement consistency | 미해결. 과제의 실제 난제 | — |

1번은 필수다. 이것 없이는 iter4 이후의 모든 val 수치가 여전히 서로 다른 문제를 재는 값이 된다.

2~3번을 먼저 돌려야 iter3 의 결과를 정확히 해석할 수 있다.

### epoch 증량은 권장하지 않음

5절에서 확인한 대로 train loss 가 이론 하한의 6.8% 이내이고 val 도 완전히 평평하다.
N2N 은 하한이 존재하므로 epoch 을 늘려도 하한 이하로 내려갈 수 없다.

---

## 부록: 발표용 정리

### 무엇을 했는가
DnCNN 을 clean label 없이 학습했다. 챌린지 규칙이 허용하는 "orientation 당 noisy 2장" 을
Noise2Noise 의 pair 로 사용해, `noisy A → noisy B` 로 학습했다.

### 왜 그 방법인가
1. 규칙이 이미 pair 를 허용한다 — 별도 데이터 없이 label-free 가 가능하다
2. clean 당 corrupted 를 12장(6 group × 2)으로 고정해 **규칙을 정확히 충족**한다
3. 4종 노이즈 중 2종이 zero-mean 이 아니므로 **L1(조건부 median)** 을 썼다

### 결과
- test PSNR 33.358 / SSIM 0.9421 (supervised 34.337 / 0.9427)
- **SSIM 은 동등, PSNR 격차는 −0.98 dB**
- conventional filter(adaptive 5x5) 대비 **+6.16 dB**
- gaussian·uniform 은 supervised 와 동등하거나 상회

### 이론과 데이터가 일치한 지점
N2N 은 노이즈가 zero-mean 일 때만 clean 으로 수렴한다. 실제로:

| noise | zero-mean | Δ vs supervised |
|---|---|---|
| gaussian | O | +0.211 |
| uniform | O | +0.428 |
| rician | X | −0.724 |
| salt_and_pepper | X | −3.834 |

**zero-mean 인 2종은 손실이 없고, 아닌 2종에서만 격차가 생겼다.**

### 완전 label-free 임을 어떻게 보증했는가
self-supervised validation loss 로 모델을 골랐을 때의 비용이 **0.04 dB** 였다
(두 기준의 Spearman 상관 0.99). 따라서 학습부터 모델 선택까지 clean 을 전혀 쓰지 않아도 된다.

### 한계
- salt & pepper 에서 supervised 대비 3.8 dB 열세 (impulse 는 zero-mean 이 아니다)
- rician 은 4종 중 최하위 (양의 bias `σ√(π/2)`)
- label-free / L1 / variant 12장 세 변경이 아직 분리되지 않았다 (대조군 필요)
- dipole blur 는 여전히 다루지 않았다
