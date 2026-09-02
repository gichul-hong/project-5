# iter5 리뷰 (iter1 대비)

- 대상: `claude-final/iter5.ipynb` (Colab 실행, exec 3~34)
- 비교 기준: `claude-final/iter1_review.md` (iter1 C = `PhysResUnet`, 60 epoch)
- 평가: `test_deconv_noise` 100장 → `test_label` 기준, 두 iteration 모두 동일

---

## 1. 결론

**PSNR 은 사실상 동률(+0.024 dB), SSIM 은 개선(+0.0074), 학습 시간은 39% 절감.**

| 지표 | iter1 C+TTA | iter5 C+TTA | Δ | 판정 |
|---|---|---|---|---|
| PSNR | 26.514 | **26.538** | **+0.024** | 통계적 무의미 (0.04 SE) |
| SSIM | 0.8461 | **0.8535** | **+0.0074** | 작지만 실재 가능성 높음 (4종 전부 상승) |
| 학습 | 60 ep / 41.0분 | **30 ep / 25.0분** | −39% | 명확한 개선 |
| params | 7,558,321 | 7,955,611 | +397,290 | SE block |

iter5 는 **PSNR 을 올리지 못했지만 SSIM 과 학습 효율을 올렸다.**
PSNR/SSIM 이 함께 채점되므로 SSIM 개선만으로도 채택 가치가 있다.

---

## 2. 변경 내역

iter5 는 iter1 C 대비 **최소 6가지를 동시에** 바꿨다.

| # | 항목 | iter1 C | iter5 |
|---|---|---|---|
| 1 | 입력 채널 | 3 (`measure`, `W(K)`, `W(10K)`) | **5** (+ `W(median(y), K_med)`, `y − A·W(K)`) |
| 2 | 블록 구조 | `ConvBlock` | **`ResConvBlock` + SE attention** |
| 3 | `ssim_weight` | 0.1 | **0.5** |
| 4 | `sobel_weight` | 0 | **0.1** |
| 5 | `dc_weight` | 0.05 | 0.02 |
| 6 | 학습 예산 | 60 ep / batch 16 / lr 2e-4 | **30 ep / batch 32 / lr 3e-4** |

`iter1_review.md` 5.2 에서 지적한 교란 요인 문제가 더 심해졌다.
**개별 기여도는 귀속할 수 없다.** 다만 아래 3절처럼 **변화의 패턴**으로는 추론이 가능하다.

---

## 3. 결과

### 3.1 전체 (test 100장)

| method | PSNR | SSIM |
|---|---|---|
| Measure (input) | 8.161 | −0.0143 |
| Wiener only | 15.228 | 0.4146 |
| Median 3x3 + Wiener | 19.288 | 0.5453 |
| Adaptive 5x5 + Wiener | 19.088 | 0.5512 |
| 제공 baseline U-Net | 25.016 | 0.8150 |
| iter1 C + TTA | 26.514 | 0.8461 |
| **iter5 C + TTA** | **26.538** | **0.8535** |

### 3.2 noise type 별 — 변화의 방향이 갈린다

**PSNR**

| noise | iter1 C+TTA | iter5 C+TTA | Δ |
|---|---|---|---|
| gaussian | 28.636 | 28.497 | **−0.139** |
| rician | 21.437 | 21.329 | **−0.108** |
| uniform | 27.164 | 27.202 | +0.038 |
| **salt_and_pepper** | 28.819 | **29.123** | **+0.304** |
| ALL | 26.514 | 26.538 | +0.024 |

**SSIM**

| noise | iter1 C+TTA | iter5 C+TTA | Δ |
|---|---|---|---|
| gaussian | 0.8712 | 0.8765 | **+0.0053** |
| rician | 0.7191 | 0.7249 | **+0.0058** |
| uniform | 0.8788 | 0.8863 | **+0.0075** |
| salt_and_pepper | 0.9153 | 0.9262 | **+0.0109** |
| ALL | 0.8461 | 0.8535 | **+0.0074** |

### 3.3 패턴에서 읽히는 인과

귀속 실험(ablation)은 없지만, 두 변화 패턴이 서로 다른 축을 따르므로 원인 추정이 가능하다.

**(a) SSIM 이 4종 전부 상승 → `ssim_weight` 0.1→0.5 + Sobel 손실의 효과**

크기가 SSIM 이 낮은 순서와 무관하게 고르게 나타났고(+0.0053 ~ +0.0109),
손실 함수가 SSIM 을 직접 5배 강하게 최적화했으므로 가장 단순한 설명이다.
PSNR 이 정체된 것도 일관된다 — L1 비중이 상대적으로 줄었으므로 MSE 계열 지표는 이득이 없다.

**(b) S&P 만 +0.304, gaussian/rician 은 하락 → `median_wiener` 채널의 효과**

conventional 표를 보면 median+Wiener 는 **S&P 에서만** 압도적이다.

| noise | Median+Wiener | Adaptive+Wiener | 차이 |
|---|---|---|---|
| gaussian | 19.485 | **20.519** | −1.03 |
| rician | 15.535 | **16.692** | −1.16 |
| uniform | 18.880 | **20.806** | −1.93 |
| **salt_and_pepper** | **23.252** | 18.338 | **+4.91** |

즉 iter5 가 추가한 `W(median(y))` 채널은 **S&P 에 특화된 정보**를 주는 것이고,
실제 이득도 S&P 에만 나타났다. 다른 3종에서 −0.1 내외 하락한 것은
채널이 늘어나며 용량이 분산된 대가로 보인다.

**결론**: iter5 의 두 변경은 각각 의도한 곳에서 작동했으나, **서로 다른 지표를 움직였다.**
SSIM 은 손실 변경이, S&P PSNR 은 median 채널이 올렸다.

### 3.4 유의성

이미지별 PSNR 표준편차 5.342 dB → ALL(n=100) 표준오차 **0.534 dB**.

- PSNR `+0.024` = **0.04 SE** → 완전히 무의미
- SSIM `+0.0074` (std 0.1332, SE 0.0133) = **0.56 SE** → unpaired 로는 무의미

단 SSIM 은 **4개 부분군 전부 같은 방향**이고 크기도 일관된다.
부호가 우연히 4/4 일치할 확률은 1/8 이며, 손실 함수 변경이라는 직접적 기전이 있으므로
**작지만 실재하는 개선**으로 판단한다.

**paired 검정은 불가능하다** — 6.3 참조 (iter1 의 장별 데이터가 덮어써졌다).

### 3.5 수렴

iter5 는 **30 epoch 으로 iter1 의 60 epoch 수준에 도달했다.**

| epoch | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|
| val PSNR | 25.920 | 25.946 | **25.956** | 25.925 | 25.953 | 25.955 |

ep 25~30 표준편차 0.014 dB, LR 하한 3.00e-6 도달 → **완전히 수렴**했다.
batch 16→32 와 lr 2e-4→3e-4 가 epoch 당 진도를 높인 결과로 보인다.

`iter1_review.md` 4.4 의 "이 스케줄에서 epoch 카드는 소진되었다" 는 판단이
**batch/lr 를 올리면 절반의 epoch 으로 같은 지점에 도달한다** 는 형태로 보강되었다.

### 3.6 val→test offset 이 세 번째로 일관되게 나왔다

| run | val PSNR | test PSNR | offset |
|---|---|---|---|
| iter1 C | 25.898 | 26.290 | +0.392 |
| iter1 A | 23.210 | 23.505 | +0.295 |
| **iter5 C** | 25.956 | 26.324 | **+0.368** |

`claude/iter3_review.md` 8절의 offset 부호 반전 문제(+0.69 vs −2.35)가
층화 validation 도입 후 **세 run 연속 +0.30~0.39** 로 안정되었다. 설계가 유효하다.

---

## 4. iter1 대비 종합 판정

| 항목 | 판정 |
|---|---|
| PSNR | 개선 없음 (+0.024, 0.04 SE) |
| SSIM | **개선** (+0.0074, 4종 전부) |
| S&P PSNR | **개선** (+0.304) |
| gaussian / rician PSNR | 소폭 악화 (−0.139 / −0.108) |
| 학습 효율 | **개선** (41분 → 25분) |
| **채택 여부** | **조건부 채택** — SSIM 이 채점 대상이므로 |

두 iteration 모두 제공 baseline(25.016 / 0.8150)을 크게 상회한다.
iter5 가 SSIM 에서 +0.0385, iter1 이 +0.0311 이다.

---

## 5. 여전히 해결되지 않은 것

### 5.1 Rician 이 그대로 최하위다

| iter | rician PSNR | gaussian 대비 |
|---|---|---|
| iter1 | 21.437 | −7.20 |
| **iter5** | **21.329** | **−7.17** |

iter5 의 5채널·SE·복합손실은 rician 에 **아무 영향을 주지 못했다** (오히려 −0.108).

`iter1_review.md` 이후 확인한 노이즈 파워 분해에 따르면 이 격차의 구성은 다음과 같다.

| 성분 | 크기 | 성격 |
|---|---|---|
| σ 차이 (rician noise MSE 가 gaussian 의 2.90배) | **4.62 dB** | test 분포의 속성. 알고리즘 문제 아님 |
| Rician 고유 (bias + 부호 접힘) | **2.58 dB** | 개선 여지 |

iter5 의 어떤 채널도 **σ 정보를 주지 않는다.** 이것이 `iter2.ipynb` 의 설계 근거다.

### 5.2 하위 5장이 iter1 과 거의 동일하다

| file | noise | sigma | iter1 | iter5 |
|---|---|---|---|---|
| L1_6ce7d8c7... | rician | 0.1157 | 13.045 | 13.245 |
| L1_1bd708f6... | rician | 0.0918 | 14.261 | 13.987 |
| L1_2db776ff... | uniform | 0.1931 | 15.844 | 16.376 |
| L1_5bead158... | rician | 0.1203 | 16.938 | 16.385 |
| L1_2dcc5641... | rician | 0.1213 | 16.622 | 16.918 |

**하위 5장 중 4장이 rician, 전부 고σ.** 실패 모드가 바뀌지 않았다.

### 5.3 A / B 비교가 없다

셀 18·19 가 `KeyError: 'unet_e2e'`, `KeyError: 'dncnn_blur'` 로 실패했다.
iter5 의 `SPECS` 에는 `phys_unet` 하나만 정의되어 있는데,
iter1 에서 복사한 실행 셀이 남아 존재하지 않는 키를 참조했다.

평가 셀이 `RESULTS` 기반으로 방어되어 있어 결과 산출에는 문제가 없었으나,
**iter1 과 마찬가지로 B(역할 분리 접근)의 수치가 없다.**

### 5.4 Label-free 가산점이 증빙되지 않았다

셀 29 는 `self_supervised_physics_adaptation` 을 **정의만 하고 실행하지 않았다.**
출력은 "모듈 준비 완료" 한 줄뿐이다. 가산점 근거로 제출할 수 있는 수치가 없다.

---

## 6. 프로세스 문제 (다음 iteration 에서 반드시 고칠 것)

### 6.1 iter1 산출물을 덮어썼다

iter5 가 iter1 과 **같은 파일명**으로 저장했다.

| 파일 | iter5 셀 | 결과 |
|---|---|---|
| `summary_iter1.json` | 30 | **iter1 결과 소실** |
| `test_metrics.json` | 23 | **iter1 장별 데이터 소실** |
| `error_maps/error_maps_phys_unet_tta.png` | 26 | **iter1 error map 소실** |

`iter1_review.md` 에 수치를 미리 기록해둔 덕에 비교는 가능했으나,
**장별 데이터가 사라져 paired 유의성 검정을 할 수 없다** (3.4).

→ `iter2.ipynb` 는 `test_metrics_iter2.json`, `summary_iter2.json`,
`error_maps_iter2_*.png` 로 분리해 두었다. iter3 이후도 같은 규칙을 지킬 것.

### 6.2 validation 구성이 iter1 과 미세하게 다르다

conventional 방법은 결정론적이므로 iteration 간 **같은 값이 나와야 한다.**
그런데 val PSNR 이 어긋난다.

| method | iter1 val | iter5 val | 차이 |
|---|---|---|---|
| Wiener only | 14.798 | 14.726 | −0.072 |
| Median + Wiener | 18.927 | 18.906 | −0.021 |
| Adaptive + Wiener | 18.713 | 18.690 | −0.023 |
| Mean + Wiener | 18.775 | 18.779 | +0.004 (K 도 0.005623 → 0.01 로 변경) |

test 값은 완전히 동일하므로(`Wiener only` 15.228, `Median` 19.288 등)
**test 비교는 유효하지만 val 비교는 엄밀하지 않다.**
iter5 의 val 25.956 과 iter1 의 25.898 을 직접 비교하면 안 된다.

원인은 val 구성(`val_sigma_levels` 등)의 차이로 추정된다. iteration 간 val 은 고정해야 한다.

---

## 7. iter2 와의 관계

`claude-final/iter2.ipynb` 는 iter1 C 에서 **변경을 2개로 제한**했다.

| | iter2 | iter5 |
|---|---|---|
| 채널 | 5 (`+ global σ̂`, `+ local σ̂`) | 5 (`+ median_wiener`, `+ resid`) |
| σ 정보 | **있음** | 없음 |
| 블록 | ConvBlock (iter1 동일) | ResConvBlock + SE |
| 손실 | L1 + 0.1·SSIM (iter1 동일) | L1 + 0.5·SSIM + 0.1·Sobel |
| `dc_weight` | 0 | 0.02 |
| 변경 수 | **2** | 6+ |

**두 노트북은 서로 다른 축을 건드리므로 상호 배타적이지 않다.**
iter2 결과가 나오면 다음이 가능하다.

| iter2 결과 | 다음 조치 |
|---|---|
| σ̂ 가 rician/고σ 에서 이득 | **iter5 의 SSIM 손실 + median 채널과 결합** (7채널) |
| σ̂ 이득 없음 | iter5 구성을 채택하고 σ̂ 는 폐기 |

결합 시에도 iter2 와 iter5 의 기여가 각각 측정된 상태이므로 귀속이 가능하다.

---

## 8. 다음 우선순위

| # | 항목 | 근거 | 비용 |
|---|---|---|---|
| **1** | iter2 결과 확인 (진행 중) | σ 가 지배 변수라는 가설 검증 | — |
| **2** | **iter2 + iter5 결합** (σ̂ 2채널 + median 채널 + SSIM 0.5 손실) | 서로 다른 축이고 각각 효과가 측정됨 | 30분 |
| 3 | **B(DnCNN) 완주** | iter1·iter5 모두 없음. 이론적 상한이 낮지 않다 | 1시간 |
| 4 | **Label-free 모듈 실제 실행** | iter5 에서 정의만 됨. 가산점 증빙 필요 | 30분 |
| 5 | 학습 가능한 Wiener `K` (대역별) | 제공 `model_config` 의 `wiener_bands: 4` 힌트. 현재 `K` 는 상수 | 2시간 |

### 하지 말아야 할 것

- **iter5 방식의 다중 동시 변경**: 6개를 함께 바꿔 귀속이 불가능해졌다
- **epoch 증량**: iter5 가 30 epoch 으로 수렴을 확인했다. batch/lr 조정이 더 효율적이다
- **iter1 과 같은 파일명 사용**: 6.1 에서 실제로 데이터를 잃었다
- **Rician 2차 모멘트 debias**: 별도 실측에서 5개 중 4개 악화 확인

---

## 9. 부록: iter5 학습 곡선

| epoch | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| val PSNR | 21.306 | 23.851 | 24.533 | 25.179 | 25.657 | 25.920 | 25.955 |
| val SSIM | 0.6223 | 0.7495 | 0.7743 | 0.8029 | 0.8167 | 0.8208 | 0.8231 |
| train loss | 4.308e-1 | 2.123e-1 | 1.749e-1 | 1.645e-1 | 1.542e-1 | 1.455e-1 | 1.461e-1 |

best: **ep 27, val PSNR 25.956 / SSIM 0.8231**, 25.0분

> train loss 절대값이 iter1(5.5e-2)보다 큰 것은 손실 구성이 다르기 때문이다
> (`ssim_weight` 0.5, `sobel_weight` 0.1 추가). iteration 간 loss 값 직접 비교는 무의미하다.

### epoch 1 비교

| run | ep 1 val PSNR |
|---|---|
| iter1 C | 21.703 |
| iter5 C | 21.306 |
| iter1 A (end-to-end) | 10.733 |

iter5 가 iter1 보다 ep 1 이 낮은 것은 채널·손실이 복잡해진 초기 부담으로 보이나
ep 5 에서 역전한다(23.851 vs 23.571). 두 run 모두 **zero-init 덕분에
Wiener 기준선(val 14.7~14.8)에서 출발**해 ep 1 만으로 +6.5 dB 이상을 얻었다.
