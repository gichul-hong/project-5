# iter1 리뷰

- 대상: `claude-final/iter1.ipynb` (Colab 실행 완료, exec 3~32)
- 실행 환경: Colab GPU, `ROOT = /content/drive/MyDrive/DS2026/20260831-pjt5-이종호/실습`
- 평가: `test_deconv_noise` 100장 입력 → `test_label` 기준 PSNR/SSIM
- 총 소요: 캐시 8.7분 + C 41.0분 + A 50.2분 ≈ 1시간 40분

---

## 1. 결론

**C(physics-residual U-Net) + flip TTA 가 제공 baseline 을 PSNR +1.50 dB, SSIM +0.031 앞섰다.**
`iter1_plan.md` 5절 기준으로 **최고 등급(우수)** 이다.

| 등급 | 조건 | 판정 |
|---|---|---|
| 실패 | C < 최선 conventional | — |
| 하한 통과 | C > 최선 conventional | 통과 (+7.23 dB) |
| 성공 | C > 제공 baseline | 통과 (+1.50 dB) |
| **우수** | 위 + noise 4종 **전부** conventional 상회 | **통과** |

### 최종 결과 (test 100장)

| method | PSNR | SSIM |
|---|---|---|
| Measure (input) | 8.161 | −0.0143 |
| Wiener only (K=0.0178) | 15.228 | 0.4146 |
| Mean 3x3 + Wiener | 19.256 | 0.5296 |
| Median 3x3 + Wiener | 19.288 | 0.5453 |
| Adaptive 5x5 + Wiener | 19.088 | 0.5512 |
| 제공 baseline U-Net (100 ep) | 25.016 | 0.8150 |
| A. End2End U-Net (60 ep) | 23.505 | 0.7890 |
| C. Physics-residual U-Net | 26.290 | 0.8419 |
| **C + flip TTA** | **26.514** | **0.8461** |

**B(DnCNN)는 실행되지 않았다** — epoch 1 중 `KeyboardInterrupt` 로 수동 중단되었다 (5절).

---

## 2. B 가 없는데 평가가 나온 이유

의도된 동작이다. `RESULTS` 에 등록된 모델만 평가 대상에 넣는다.

```python
for _k in ("unet_e2e", "dncnn_blur", "phys_unet"):
    if _k in RESULTS:          # 중단된 B 는 여기서 걸러진다
        METHODS.append(_k)
```

셀 19 가 `train_model()` 완료 전에 중단되어 `RESULTS["dncnn_blur"]` 할당문 자체가 실행되지 않았다.
따라서 표·시각화·summary 에서 B 열이 빠진 채 나머지가 정상 산출되었다.

> 셀 이름 혼동 주의: 노트북의 학습 셀 순서는 **C(phys_unet) → A(unet_e2e) → B(dncnn_blur)** 다.
> 세 번째 셀이 B 이며, C 는 첫 번째로 이미 완주했다.

---

## 3. 결과 분석

### 3.1 noise type 별 PSNR

| method | gaussian | rician | uniform | salt_and_pepper | ALL |
|---|---|---|---|---|---|
| Measure (input) | 8.156 | **8.664** | 7.900 | 7.925 | 8.161 |
| Wiener only | 15.778 | 13.676 | 16.181 | 15.276 | 15.228 |
| Mean + Wiener | 20.393 | 16.023 | 20.454 | 20.153 | 19.256 |
| Median + Wiener | 19.485 | 15.535 | 18.880 | **23.252** | 19.288 |
| Adaptive + Wiener | **20.519** | **16.692** | **20.806** | 18.338 | 19.088 |
| 제공 baseline | 26.812 | 21.264 | 25.193 | 26.795 | 25.016 |
| A. End2End | 25.136 | 19.549 | 23.304 | 26.030 | 23.505 |
| C. Phys-residual | 28.434 | 21.241 | 26.955 | 28.530 | 26.290 |
| **C + TTA** | **28.636** | **21.437** | **27.164** | **28.819** | **26.514** |

### 3.2 Rician 이 유일한 실패 지점

입력 PSNR 이 **가장 높은데 출력 PSNR 이 가장 낮다.** 이게 핵심 진단이다.

| noise | 입력 | 출력(C+TTA) | 개선폭 |
|---|---|---|---|
| rician | **8.664 (최고)** | **21.437 (최저)** | **+12.773 (최저)** |
| uniform | 7.900 | 27.164 | +19.264 |
| gaussian | 8.156 | 28.636 | +20.481 |
| salt_and_pepper | 7.925 | 28.819 | +20.894 |

즉 "rician 입력이 더 나빠서" 가 아니라 **방법이 rician 에서 특정하게 실패**한다.

제공 baseline 과의 격차도 rician 에서만 사라진다.

| noise | C+TTA − baseline |
|---|---|
| salt_and_pepper | +2.024 |
| uniform | +1.971 |
| gaussian | +1.824 |
| **rician** | **+0.173** |

TTA 를 빼면 rician 은 **−0.023 dB** 로 사실상 동률이다.
noise type 당 표준오차가 1.041 dB 이므로 이 차이는 통계적으로 무의미하다.

**원인**: Rician 은 `E[|s + n|] > s` 로 **양의 bias** 를 갖는다.
신호가 0인 영역에서 `E = σ√(π/2) ≈ 1.25σ` 이고 σ 범위가 0~0.15 이므로 최대 **0.19** 의 계통 오차가 남는다.
C 의 구조는 Wiener base(선형 필터)와 residual 모두 **zero-mean 가법 노이즈를 전제**하므로 이 가정이 깨진다.

하위 5장 중 **4장이 rician** 이다.

| file | noise | sigma | PSNR |
|---|---|---|---|
| L1_6ce7d8c7... | rician | 0.1157 | 13.045 |
| L1_1bd708f6... | rician | 0.0918 | 14.261 |
| L1_2db776ff... | uniform | 0.1931 | 15.844 |
| L1_2dcc5641... | rician | 0.1213 | 16.622 |
| L1_5bead158... | rician | 0.1203 | 16.938 |

### 3.3 conventional 대비 — "왜 딥러닝인가" 근거 확보

| noise | C+TTA | 최선 conventional | Δ |
|---|---|---|---|
| gaussian | 28.636 | 20.519 (adaptive) | **+8.117** |
| rician | 21.437 | 16.692 (adaptive) | **+4.745** |
| uniform | 27.164 | 20.806 (adaptive) | **+6.358** |
| salt_and_pepper | 28.819 | 23.252 (median) | **+5.567** |
| **ALL** | **26.514** | 19.288 (median) | **+7.226** |

**4종 전부에서 conventional 을 크게 상회한다.** `claude/iter1_review.md` 2.3절에서 지적했던
"S&P 를 빼면 딥러닝 이득이 작다" 는 문제는 여기서 재발하지 않았다 (최소 격차가 rician 의 +4.75 dB).

또한 **단일 conventional 방법이 모든 noise 를 커버하지 못한다.**
adaptive 는 gaussian/rician/uniform 에서 최선이지만 S&P 에서 18.338 로 median(23.252)에 4.9 dB 뒤진다.
noise type 을 모르는 상태에서 하나를 골라야 하므로 conventional 의 실질 상한은 ALL 열의 19.288 이다.

### 3.4 flip TTA

| noise | 이득 |
|---|---|
| gaussian | +0.202 |
| rician | +0.196 |
| uniform | +0.209 |
| salt_and_pepper | +0.289 |
| **ALL** | **+0.224** |

`A(flip x) = flip(A x)` 를 실측 오차 1.2e-07 로 확인한 뒤 적용했고, **4종 전부에서 일관되게 양수**다.
부분군 4개가 모두 같은 방향인 것 자체가 우연이 아님을 뒷받침한다. 추론 비용은 4배다.

---

## 4. 검증된 설계 결정

### 4.1 층화 validation 이 val→test offset 문제를 해결했다

`claude/iter3_review.md` 8절의 핵심 문제는 val→test offset 이 run 마다 **부호까지 뒤집힌 것**이었다
(iter2 +0.69 dB, iter3 −2.35 dB). 원인은 무작위 추첨 validation 이었다.

iter1 에서 층화 validation(43장 × 4 noise × σ 3층 = 516 조합)을 도입한 결과:

| run | val PSNR | test PSNR | offset |
|---|---|---|---|
| C phys_unet | 25.898 | 26.290 | **+0.392** |
| A unet_e2e | 23.210 | 23.505 | **+0.295** |

**두 run 의 offset 이 +0.3~0.4 로 일치한다.** val 수치로 run 간 비교가 가능해졌다.
iter2 이후에도 이 validation 구성을 유지해야 한다.

### 4.2 physics base + residual + zero-init

C 는 zero-init 이라 초기화 시점 출력이 `Wiener(K=0.0178)` 와 정확히 같다 (val 14.798).
epoch 1 만으로 **val 21.703** 이 되었다 — 454 step 에서 +6.9 dB.

같은 시점 A 는 val 10.733 이었고, **A 가 60 epoch 을 다 써서 도달한 23.210 을
C 는 epoch 3(23.069)~4(23.397) 에서 통과했다.**

### 4.3 forward model 자기검증

노트북 안에서 재확인한 값이다. 이게 틀렸으면 학습 전체가 무의미했다.

| 검증 | 결과 |
|---|---|
| `dipole_forward(label)` vs `test_deconv_only` | max err **1.19e-07 ~ 1.79e-07** |
| gaussian 잔차 std vs σ | 0.0597 vs 0.0599 |
| uniform 잔차 std vs σ/√3 | 0.0180 vs 0.0180 / 0.0998 vs 0.1000 |
| rician 잔차 mean | +0.0102 ~ +0.0653 (양의 bias 확인) |
| seed 재현성 | 동일 seed → max diff 0.00e+00 |
| Wiener ≡ Tikhonov | `True` (max diff 1.91e-06) |
| flip 등변성 | 1.19e-07 ~ 1.79e-07 |

### 4.4 수렴 상태

**C 는 완전히 수렴했다.** ep 49~60 의 val PSNR 12개 값:
`25.844 / 25.857 / 25.837 / 25.866 / 25.859 / 25.869 / 25.882 / 25.884 / 25.887 / 25.883 / 25.898 / 25.894`
→ 표준편차 **0.019 dB**. LR 도 설계 하한 4.00e-6 에 도달했다.

`claude/iter2_review.md` 3.4절과 같은 상태다. **이 스케줄에서 epoch 카드는 소진되었다.**
epoch 을 늘려 얻을 이득은 +0.2~0.4 dB 수준으로 예상되므로 비용 대비 효율이 낮다.

**A 는 수렴하지 않았다.** best 가 **마지막 epoch(60)** 이고 ep 55~60 이 계속 상승 중이다
(23.177 → 23.210). `claude/iter1_review.md` 2.1절의 "best 가 마지막 epoch 이면 수렴이 아니다" 에 해당한다.
또 ep 25(21.664) → ep 30(21.073) 에서 −0.59 dB 의 하락이 있었다.

---

## 5. 문제점

### 5.1 B(DnCNN)가 없어 방법론 비교 한 축이 빠졌다

조교 example 이 제시한 두 방법 중 B("measurement domain 에서 noise 만 학습으로 제거하고
deconvolution 은 Wiener 가 담당")의 수치가 없다.

C 가 사실상 B 의 개선판(Wiener 를 뒤에 붙이는 대신 앞에 base 로 주는 구조)이라는 설명은 가능하지만
근거 수치가 없으므로 발표에서 질문받을 지점이다.

`iter1_plan.md` 2절 경로 C 의 축소 설정(12층 64feat, 25 epoch)이면 약 1시간이다.

### 5.2 A vs C 비교에 교란 요인이 3개 있다

`iter3_review.md` 9절과 같은 실수를 반복했다. C 는 A 대비 **네 가지를 동시에** 바꿨다.

| # | 변경 | A | C |
|---|---|---|---|
| 1 | **physics 입력 + residual + zero-init** | 없음 | 적용 |
| 2 | `clean_branch_prob` | 0.5 | 0.0 |
| 3 | params | 13.43 M | 7.56 M |
| 4 | `dc_weight` | 0 | 0.05 |

따라서 **`C − A = +2.785 dB` 는 "physics 구조의 효과" 가 아니다.**
특히 A 는 step 의 절반을 blur 없는 `N(x)→x` 문제에 썼으므로 dipole 학습량이 C 의 절반이다.

다만 C 가 **파라미터 44% 적게** 이겼다는 점에서 방향성은 분명하다.

### 5.3 A 는 제공 baseline 과 공정 비교가 아니다

| | A (내 학습) | 제공 baseline |
|---|---|---|
| epochs | 60 | **100** |
| batch | 16 | **64** |
| loss | L1 + 0.1·SSIM | L2 |
| test PSNR | 23.505 | **25.016** |

A 가 1.51 dB 뒤진 것은 구조 차이가 아니라 **학습량 차이**다 (5.2 의 미수렴과도 일치).
"A vs baseline" 을 아키텍처 비교로 해석하면 안 된다.

### 5.4 통계적 유의성을 paired 로 계산하지 않았다

이미지별 PSNR 표준편차 5.206 dB → noise type 당(n=25) 표준오차 1.041 dB, ALL(n=100) 0.521 dB.

`C+TTA − baseline = +1.498 dB` 는 ALL 기준 2.9 SE 이므로 유의하다.
단 두 방법은 **같은 100장**을 쓰므로 올바른 검정은 차이값의 표준오차를 쓰는 paired 검정이고,
그 값은 0.521 보다 훨씬 작다. 즉 실제 유의성은 이보다 강하다.
`test_metrics.json` 에 장별 값이 있으므로 iter2 에서 paired 로 재계산해야 한다.

### 5.5 `dc_weight = 0.05` 가 검증되지 않았다

data-consistency 항 `‖A x̂ − y‖₁` 의 `y` 는 **노이즈를 포함**하므로, 이 항은 모델을
노이즈에 맞추도록 압력을 준다. 고노이즈 구간에서 해로울 수 있다. ablation 이 필요하다.

### 5.6 캐시 빌드가 8.7분 걸렸다

`train` 7,268장 → 1.77 GB, 총 522.1초. 세션당 1회 비용이다.
Colab 세션이 끊기면 매번 반복된다. `/content/cache` 를 Drive 에 백업해두면 재빌드를 피할 수 있으나
Drive 에서 1.77 GB 를 읽는 시간과 대차대조가 필요하다.

---

## 6. iter2 권장 우선순위

| # | 항목 | 근거 | 예상 효과 | 비용 |
|---|---|---|---|---|
| **1** | **Rician 대응**: VST 또는 bias 보정항 | 3.2. 유일한 실패 지점. gaussian 대비 −7.20 dB, baseline 대비 동률 | rician +1~3 dB | 3h |
| **2** | **σ 조건부 입력** (FFDNet 방식) | 하위 5장이 전부 고σ. σ 를 모델이 모른다 | 전 구간 +0.3~0.8 dB | 3h |
| **3** | **B 실행** (12층 64feat, 25 ep) | 5.1. 방법론 비교 축 복구 | 점수 아닌 발표 근거 | 1h |
| **4** | **교란 요인 분리**: C 에서 `dc_weight=0` ablation | 5.5 | 해석 정확도 | 45분 |
| **5** | **median 채널 추가** | 3.3. median 이 S&P 에서 conventional 최선. impulse 는 detection 문제 | S&P +0.5~1 dB | 1h |
| **6** | Label-free (Noise2Noise) | 가산점. 12개 풀에서 같은 noise type 두 실현을 pair 로 | 점수 아닌 가산점 | 3h |
| **7** | paired 유의성 재계산 | 5.4. `test_metrics.json` 활용 | 보고 정확도 | 15분 |

### 하지 말아야 할 것

- **C 의 epoch 증량**: 4.4 에서 수렴 확인. +0.2~0.4 dB 예상으로 비용 대비 효율이 낮다
- **A 의 개선**: A 는 대조군이다. 제공 baseline 이 이미 같은 구조의 상위 버전이다
- **ALL 평균만 보고 판단**: 3.2 처럼 noise type 별로 분해해야 원인이 보인다

### Rician 대응 구체안 (1번)

세 가지 후보 중 (a) 를 먼저 시도할 것을 권한다.

| 안 | 방법 | 장점 | 단점 |
|---|---|---|---|
| **(a) bias 보정 채널** | `sqrt(max(y² − σ̂², 0))` 을 입력 채널로 추가 | 구현 간단, Rician 의 2차 모멘트 관계를 직접 반영 | σ̂ 추정이 필요 |
| (b) VST | Rician → 근사 Gaussian 변환 후 복원, 역변환 | 이론적으로 깔끔 | 변환/역변환에서 정보 손실 |
| (c) noise type 분류기 | 4종 분류 후 분기 | 각 noise 에 특화 가능 | 학습 파이프라인이 복잡, 분류 오류가 전파 |

`E[y²] = s² + 2σ²` (2D Rician) 이므로 `s ≈ sqrt(y² − 2σ̂²)` 로 bias 를 상당 부분 제거할 수 있다.
σ̂ 는 고주파 대역의 MAD 로 추정 가능하다 (label 불필요).

---

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `logs_final/00000_phys_unet/checkpoint_best.ckpt` | C best (ep 59, val 25.898) |
| `logs_final/00001_unet_e2e/checkpoint_best.ckpt` | A best (ep 60, val 23.210) |
| `logs_final/*/history.json` | epoch 별 loss/lr/val (noise type 별 포함) |
| `test_metrics.json` | test 100장 장별 PSNR/SSIM (전 방법) |
| `summary_iter1.json` | 방법별 집계 + noise type 별 |
| `error_maps/error_maps_phys_unet_tta.png` | noise 4종 상대오차율(%) |
| `best_k.json` | Wiener K (none 0.01778 / mean·median 0.005623 / adaptive 0.01) |

### 발표 요건 충족

| 요건 | 상태 |
|---|---|
| 전체 파이프라인 설명 | 확보 (노트북 2·6절) |
| 복원 전/후 비교 | 확보 (8-1 difference map, 8-2 zoom-in) |
| 방법론 선택 근거 | 확보 — C 가 A 의 60 epoch 결과를 **epoch 3~4 에서 통과**한 곡선이 가장 강한 근거 |
| Error map 2종 이상 | 확보 (noise 4종) |
| Conventional 비교 | 확보 (3.3, 4종 전부 상회) |
| 취약점 분석 | 확보 (3.2 rician) |
| Label-free 가산점 | 미착수 → iter2 6번 |

---

## 8. 부록: 학습 곡선

### C. phys_unet (41.0분, 60 epoch)

| epoch | 1 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| val PSNR | 21.703 | 24.179 | 24.865 | 25.392 | 25.657 | 25.857 | 25.894 |
| train loss | 1.484e-1 | 7.263e-2 | 6.425e-2 | 6.050e-2 | 5.689e-2 | 5.528e-2 | 5.560e-2 |

best: **ep 59, val 25.898 / SSIM 0.8115**

### A. unet_e2e (50.2분, 60 epoch)

| epoch | 1 | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| val PSNR | 10.733 | 19.520 | 21.676 | 21.073 | 22.884 | 23.039 | 23.210 |
| train loss | 2.194e-1 | 9.245e-2 | 7.132e-2 | 6.148e-2 | 5.434e-2 | 5.043e-2 | 5.092e-2 |

best: **ep 60, val 23.210 / SSIM 0.7578** (마지막 epoch = 미수렴)

### 실측 처리 속도

| run | params | 시간/epoch | 60 epoch |
|---|---|---|---|
| C phys_unet | 7.56 M | 41초 | 41.0분 |
| A unet_e2e | 13.43 M | 50초 | 50.2분 |
| B dncnn (20층 96feat) | 1.58 M | 미완 | 추정 2~3시간 |

`iter1_plan.md` 의 T4 추정(C 100~140초/epoch)보다 약 2.5배 빨랐다.
