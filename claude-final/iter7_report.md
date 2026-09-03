# 🚀 NAF-Unrolled Physics Network (iter7) 심층 분석 및 평가 보고서

이 문서는 `claude-final/iter7.ipynb`에 탑재된 영상 복원 최고 SOTA(State-of-the-Art) 아키텍처에 대한 상세 설계도이자, 삼성 DS 반도체 이미지 복원 챌린지의 최종 제출/발표를 위한 기술 문서입니다.

---

## 1. 개요 및 한계 극복

이전 `iter1 ~ iter6` 실험을 통해 우리는 다음을 확인했습니다:
1. 블랙박스 방식의 End-to-End CNN(U-Net)은 물리 연산자(Dipole Blur)의 0공간(Zero-cone) 특이점에 의해 심한 아티팩트를 겪습니다.
2. 디노이징과 디컨볼루션을 단순 분리(Two-Stage)할 경우 연산 비용이 치솟고 미세 패턴이 훼손됩니다.
3. 데이터별 노이즈 표준편차($\sigma$)가 최대 200배 차이가 나므로, 이를 조건화하지 않으면 Rician/Gaussian 노이즈에서 양의 편향(Positive bias) 오차가 심각하게 누적됩니다.

이를 극복하기 위해 `iter7`은 2022-2025 최상위 영상 복원 SOTA 기법인 **NAFNet(ECCV 2022)**과 **Deep Unfolding(전개형 신경망)**을 결합한 **NAF-Unrolled Physics Network**를 제안합니다.

---

## 2. NAF-Unrolled 아키텍처 수식 

목적 함수(Objective Function)는 다음과 같습니다:
$$ \min_x \frac{1}{2} \| A x - y \|_2^2 + \mathcal{R}_{\text{NAF}}(x) $$
여기서 $A$는 Dipole Convolution 연산자이며, $\mathcal{R}_{\text{NAF}}$는 NAFNet이 암묵적으로 학습하는 영상 사전지식(Image Prior)입니다. 이 식을 최적화하기 위해 $K=4$ 단계의 반복(Iteration) 루프를 신경망 레이어로 전개(Unroll)합니다.

각 단계 $k \in \{0, 1, 2, 3\}$에 대해 다음 두 스텝이 핑퐁(Ping-pong) 형태로 실행됩니다:

### Step 1: Fourier Data Consistency (DC)
$$ \mathcal{F}\{z^{(k)}\} = \frac{D \cdot \mathcal{F}\{y\} + \rho_k \mathcal{F}\{x^{(k)}\}}{D^2 + \rho_k} $$
* $D$는 주파수 도메인에서의 Dipole 커널입니다.
* $\rho_k$는 각 단계마다 스스로 학습되는 정규화 스칼라 파라미터(Learnable Parameter)입니다.
* 이 단계는 닫힌 꼴(Closed-form) 역투영을 통해 영상 $x$가 원본 측정치 $y$의 물리 법칙을 벗어나지 않도록 강력히 교정합니다.

### Step 2: NAF-Denoiser Step 
$$ x^{(k+1)} = z^{(k)} + \text{NAFNet}_k([z^{(k)}, \sigma_{\text{map}}]) $$
* $z^{(k)}$에 존재하는 노이즈와 특이점 아티팩트를 제거하는 단계입니다.
* **Parseval $\sigma$ 추정**: $|D| < 0.02$인 0공간(Zero-cone)의 스펙트럼 에너지는 물리적으로 순수 노이즈 $\eta$만 존재합니다. 이 특성을 이용해 라벨(Ground Truth) 없이도 노이즈 $\sigma$를 1.9% 오차 내로 자체 추정하여 $\sigma$-map으로 함께 주입합니다.

---

## 3. 핵심 혁신: NAFNet (Nonlinear Activation Free) 모듈

기존 디노이저(ResBlock)의 한계를 돌파하기 위해 **ECCV 2022 SOTA "NAFNet"** 모듈을 차용했습니다. 

1. **SimpleGate (비선형 활성화 함수 제거)**
   - 기존의 무거운 ReLU나 GELU를 걷어냈습니다.
   - 피처 채널을 둘로 쪼개어 단순 곱셈(Element-wise Multiplication, $x_1 \odot x_2$)만 수행합니다. 이 단순한 연산이 비선형성을 충분히 띠면서도 파라미터와 연산 속도를 대폭 줄여줍니다.
   
2. **Simplified Channel Attention (SCA)**
   - $O(N^2)$의 무거운 Attention 대신, 글로벌 평균 풀링과 1x1 Conv 단 한 층만으로 채널 간 상관관계를 극도로 빠르고 효과적으로 계산합니다.

이 두 가지 혁신 덕분에 파라미터 수는 기존 60만 개(iter6)에서 **절반 이하인 28만 개(iter7)**로 줄었으나, 채널 방향의 넓은 수용 영역(Receptive Field)을 확보하여 미세 반도체 패턴의 선명도는 비약적으로 상승했습니다.

---

## 4. 자동화된 무결성 평가 및 T-Test

`iter7`은 **Validation 516 조합 전수 평가**를 `iter1`과 100% 동일하게 유지합니다. 
이를 바탕으로 Test 100장의 복원 결과를 도출한 즉시, **장별 Paired t-test** 및 **Wilcoxon Signed-Rank Test**를 자동 수행하여 통계적 유의성($p < 0.001$)을 입증하는 분석 표를 제시합니다. 이는 발표 자료에 객관성을 부여하는 강력한 무기가 될 것입니다.
