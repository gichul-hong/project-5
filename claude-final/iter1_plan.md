# iter1 실행 계획

> 대상 노트북: `claude-final/iter1.ipynb`
> 작성: 2026-09-02

---

## 0. 먼저 확실히 할 것: 3개는 순차 실행이다

노트북 7-1 절은 셀이 **3개로 분리**되어 있고, 각각 독립적인 `train_model()` 호출이다.
동시에 학습하지 않는다.

```
[셀 29]  RESULTS["phys_unet"]  = train_model(SPECS["phys_unet"])    ← C
[셀 30]  RESULTS["unet_e2e"]   = train_model(SPECS["unet_e2e"])     ← A
[셀 31]  RESULTS["dncnn_blur"] = train_model(SPECS["dncnn_blur"])   ← B + Wiener K 재sweep
```

**셋 다 돌릴 필요는 없다.** 평가 셀은 `RESULTS` 에 있는 것만 표에 넣도록 작성되어 있다.

```python
for _k in ("unet_e2e", "dncnn_blur", "phys_unet"):
    if _k in RESULTS:          # ← 없으면 건너뛴다
        METHODS.append(_k)
```

따라서 **C 하나만 돌려도 전체 평가·시각화·발표 자료가 완성된다.**

---

## 1. 시간 예산 (가장 중요한 제약)

조교가 명시한 실측치는 **L40S 4장, batch 64** 기준이다.

| 모델 | params | 처리량 (L40S×4) | epoch |
|---|---|---|---|
| A. U-Net chans 64 | 13.43 M | 440 img/s | 16.5초 |
| B. DnCNN 20층 96feat | 1.58 M | 140 img/s | 52.1초 |

Colab 단일 GPU 는 이보다 훨씬 느리다. 아래는 **T4 기준 외삽 추정치**이며 반드시 실측으로 교정해야 한다.

| run | params | 추정 epoch 시간 (T4) | 60 epoch | 비고 |
|---|---|---|---|---|
| **C `phys_unet`** (chans 48) | 7.56 M | 약 100~140초 | **약 2.0시간** | 주력 |
| **A `unet_e2e`** (chans 64) | 13.43 M | 약 120~160초 | 약 2.3시간 | 제공 ckpt 와 중복성 있음 |
| **B `dncnn_blur`** (20층 96feat) | 1.58 M | 약 350~450초 | **약 6.5시간** | 파라미터는 작지만 **깊어서 가장 느리다** |

> **주의**: B 는 파라미터가 A 의 1/8 인데 시간은 3배다. DnCNN 은 256×256 해상도를 한 번도 줄이지 않고
> 20층을 통과하므로 연산량이 크다. U-Net 은 pooling 으로 해상도를 줄여 훨씬 효율적이다.

**Colab 제약**: 세션 12시간 한도, 유휴 시 연결 끊김. 셋 다 60 epoch 은 약 11시간이라 **현실적으로 불가능**하다.

---

## 2. 권장 실행 경로

### 경로 A — 최소 (약 2.5시간, 권장)

세션 한 번에 안전하게 끝나고 발표 요건을 모두 충족한다.

| 순서 | 작업 | 시간 |
|---|---|---|
| 1 | 스모크 테스트 (`max_train_images=200`, `epochs=1`) | 5분 |
| 2 | 캐시 빌드 + conventional K sweep | 10분 |
| 3 | **C `phys_unet` 60 epoch** | 2.0시간 |
| 4 | 평가 (conventional + 제공 baseline + C + C·TTA) | 10분 |

비교 대상으로 **제공 baseline ckpt (100 epoch 학습된 end-to-end U-Net)** 가 이미 있으므로
"딥러닝 vs conventional" 과 "physics-informed vs end-to-end" 두 축이 모두 성립한다.

### 경로 B — 표준 (약 5시간, 2세션 권장)

경로 A 에 A 를 추가한다. A 의 목적은 성능이 아니라 **내 학습 파이프라인이 조교 결과를 재현하는지 검증**이다.

| 순서 | 작업 | 시간 |
|---|---|---|
| 1~4 | 경로 A 전부 | 2.5시간 |
| 5 | **A `unet_e2e` 40 epoch** (60 → 40 으로 축소) | 1.5시간 |
| 6 | 재평가 | 10분 |

`epochs=40` 으로 줄여도 cosine schedule 이 그에 맞춰 감쇠하므로 문제없다
(`get_lr` 이 `spec["epochs"]` 를 기준으로 계산한다).

### 경로 C — B 를 포함해야 할 때

B 를 **반드시 축소**한다. 노트북 실행 전에 아래를 넣는다.

```python
SPECS["dncnn_blur"]["epochs"] = 25
SPECS["dncnn_blur"]["model_config"] = {"dncnn_layers": 12, "dncnn_features": 64}  # 1.58M -> 0.41M
SPECS["dncnn_blur"]["batch"] = 32
```

12층 64feat 이면 약 0.41 M params 로 연산량이 1/4 이 되어 25 epoch 약 1시간이다.

B 의 가치는 순위 경쟁이 아니라 **"denoise 와 deconvolve 의 역할을 분리하면 어떻게 되는가"** 라는
방법론 비교 데이터다. 축소해도 그 목적은 달성된다.

---

## 3. 실행 순서를 이렇게 정한 이유

### 왜 C 를 먼저 돌리는가

1. **하한이 보장된다.** `phys_unet` 은 마지막 conv 를 zero-init 하므로 학습 시작 시점의 출력이
   `Wiener(measure, K)` 와 **정확히 같다**. 즉 최악의 경우에도 conventional 성능에서 시작한다.
2. **가장 강한 사전 근거가 있다.** `gemini-deconvolution` 에서 같은 구조 변경으로
   25.59 dB → 78.01 dB (+52 dB) 를 실측했다.
3. **실패해도 진단이 쉽다.** epoch 1 val PSNR 이 conventional 과 일치하는지만 보면
   물리 연산·정규화·clip 배선이 맞는지 바로 알 수 있다.

### 왜 A 의 우선순위가 낮은가

제공 `checkpoint_baseline_best.ckpt` 가 **이미 A 와 동일 구조를 100 epoch 학습한 결과**다.

```
model_config: {in_chans: 1, out_chans: 1, chans: 64, num_pool_layers: 4}
spec: {target: "label", clean_branch_prob: 0.5, epochs: 100, batch: 64, lr: 2e-4, loss_model: "l2"}
```

내가 A 를 60 epoch 돌려도 조교의 100 epoch·batch 64 결과보다 나을 이유가 없다.
A 는 **파이프라인 검증용 대조군**으로만 의미가 있다.

---

## 4. 판단 게이트 (각 시점에서 무엇을 보고 무엇을 할지)

### 게이트 1 — 스모크 테스트 직후

| 확인 | 기대 | 실패 시 |
|---|---|---|
| forward model 자기검증 | `max_err < 1e-6` | **즉시 중단.** kernel 이 배포 데이터와 다르다. 학습해도 무의미 |
| noise 잔차 vs meta | gaussian `std≈σ`, uniform `std≈σ/√3` | noise 함수 확인 |
| `Wiener == Tikhonov` | `True` | — |
| flip 등변성 | `< 1e-6` | TTA 를 끌 것 |

### 게이트 2 — conventional K sweep 직후

`BEST_K` 와 그때의 valid PSNR 을 **반드시 기록**한다. 이 값이 이후 모든 판단의 기준선이다.

- K sweep 곡선이 grid 경계(`1e-4` 또는 `10^0.5`)에서 최대면 grid 를 넓혀야 한다
- 4개 prefilter 중 최선이 무엇인지 기억 (보통 median 또는 adaptive)

### 게이트 3 — C 의 epoch 1 직후 (가장 중요)

zero-init 때문에 **epoch 1 의 val PSNR 은 `Wiener(K=BEST_K["none"])` 의 valid PSNR 과 비슷해야 한다.**

| 관측 | 해석 | 조치 |
|---|---|---|
| epoch 1 ≈ Wiener 기준선 | 정상 | 계속 |
| epoch 1 ≪ Wiener 기준선 | residual 이 base 를 망치고 있다 | `phys_scale` 을 0.1 → 0.02 로 낮춘다 |
| epoch 1 이 NaN | AMP + FFT 문제 | `config.amp = False` |

### 게이트 4 — C 의 epoch 10 전후

| 관측 | 조치 |
|---|---|
| conventional 을 넘어 상승 중 | 그대로 60 epoch 완주 |
| conventional 근처에서 평평 | `phys_scale` 0.1 → 0.3, `dc_weight` 0.05 → 0 으로 재시작 |
| val 이 하락 | `lr` 2e-4 → 1e-4, `grad_clip` 유지 |

`dc_weight` 를 의심하는 이유: data-consistency 항 `‖A x̂ − y‖` 는 `y` 가 **노이즈를 포함**하므로
노이즈에 맞추도록 압력을 준다. 노이즈가 클 때 해로울 수 있다.

### 게이트 5 — 학습 종료 후

noise type 별 표를 보고 다음 iteration 을 정한다. 판단 기준은 표준오차다.

```
이미지별 PSNR 표준편차 / sqrt(25) = noise type 당 표준오차
```

이 값보다 작은 격차는 **통계적 잡음이므로 개선으로 보고하면 안 된다.**
`claude/iter1_review.md` 에서 이 함정에 빠져 "개선 0" 을 개선으로 착각한 사례가 있다.

---

## 5. 성공 기준

절대 수치는 예측하지 않는다. 이 문제는 blur+noise 결합이고 선례가 없어 외삽 근거가 없다.
대신 **상대 기준**으로 판단한다.

| 등급 | 조건 |
|---|---|
| **실패** | C 가 최선 conventional 보다 낮다 |
| **하한 통과** | C > 최선 conventional (physics base 를 활용했으므로 최소 요건) |
| **성공** | C > 제공 baseline U-Net |
| **우수** | C > 제공 baseline 이고, noise 4종 **전부**에서 conventional 을 상회 |

`claude/iter2_review.md` 4절의 교훈대로 **ALL 평균만 보지 않고 noise type 별로 분해**해서 본다.
salt & pepper 는 detection 문제라 상전이형으로 갑자기 좋아지는 항목이므로 평균을 왜곡한다.

---

## 6. 리스크

| # | 리스크 | 완화 |
|---|---|---|
| 1 | **Colab 세션 끊김** | best ckpt 는 매번 Drive 에 저장된다. `history.json` 도 epoch 마다 갱신. 재시작 시 conventional 부분부터 다시 필요 |
| 2 | **캐시 빌드가 Drive 속도에 묶임** | 7,468장 순차 읽기는 최초 1회뿐. 이후 epoch 은 `/content` SSD memmap |
| 3 | `/content` 용량 | train 캐시 약 1.9 GB. Colab 여유 공간으로 충분 |
| 4 | **B 가 예상보다 훨씬 느림** | 2절 표 참조. 반드시 축소하거나 생략 |
| 5 | 채점 noise 가 학습 noise 와 다를 가능성 | `Day 1` 이 "noise characteristic is not given" 이라 명시. 4종 균등 학습 + TTA 로 방어. 근본 대응은 label-free (iter2) |
| 6 | AMP 와 FFT 충돌 | `PhysResUnet.features` 와 dc loss 는 `autocast(enabled=False)` 로 fp32 강제 |

---

## 7. iter1 종료 후 산출물

| 산출물 | 경로 |
|---|---|
| best ckpt · history | `{ROOT}/claude-final/logs_final/{id}_{key}/` |
| 장별 test metric | `{ROOT}/claude-final/test_metrics.json` |
| 종합 요약 | `{ROOT}/claude-final/summary_iter1.json` |
| error map (noise 4종) | `{ROOT}/claude-final/error_maps/` |
| Wiener K | `{ROOT}/claude-final/best_k.json` |

이 파일들을 근거로 `iter1_review.md` 를 작성하고 iter2 항목을 확정한다.

---

## 8. 실행 체크리스트

```
[ ] GPU 런타임 확인 (런타임 > 런타임 유형 변경 > GPU)
[ ] 셀 1~2: Drive mount, ROOT 자동 탐색 결과 확인
[ ] 셀 3: config 출력 확인 (device 가 cuda 인지)
[ ] 셀 4~5: 캐시 빌드 (최초 1회, 수 분)
[ ] 셀 6~8: forward model 자기검증  ← 게이트 1
[ ] 셀 9~12: metric, dataset, 학습 pair 시각화
[ ] 셀 13~15: conventional + K sweep  ← 게이트 2. BEST_K 기록
[ ] 스모크: config.max_train_images=200 / epochs=1 로 셀 28 주석 해제 실행
[ ] 스모크 통과 후 config.max_train_images=None 로 복구, make_loaders() 재실행
[ ] 셀 29: C phys_unet 학습  ← 게이트 3(ep1), 게이트 4(ep10)
[ ] (선택) 셀 30: A unet_e2e 40 epoch
[ ] (선택) 셀 31: B dncnn_blur 25 epoch + 축소 설정
[ ] 셀 32~47: 평가, error map, 취약점 분석, 요약 저장  ← 게이트 5
```

> 스모크 테스트 후 `config.max_train_images` 를 되돌린 다음
> **`train_ds, val_ds, test_ds = make_loaders()` 를 다시 실행**해야 한다.
> 그러지 않으면 200장으로 본 학습이 돌아간다.
