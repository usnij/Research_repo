# 3DGS-MCMC 의 세 축 — 정규화, noise, 재배치

작성일 2026-08-14 · 모진수 · 대상 3DGS-MCMC (NeurIPS 2024), 3DGRT 이식본

---

## 요약

MCMC 가 3DGS 에서 변경하는 지점은 세 곳이다. loss 함수에 정규화 항을 추가하고, optimizer 갱신 뒤 position 파라미터에 직접 noise 를 더하며, prune 을 제거하고 자리를 재사용한다. 이 리포트는 세 축이 3DGS 의 어떤 규칙을 대체하며 왜 셋이 함께 있어야 작동하는지를 정리하고, sparse 구조에 적용할 수정안을 제시한다.

### 발견 1 — 세 축은 각각의 개선이 아니라 하나의 순환이며, 하나가 빠지면 나머지도 작동하지 않는다

정규화가 기여 없는 primitive 의 opacity 를 낮춰 자리를 만들고, 재배치가 그 자리를 유망한 위치의 복사본으로 채우며, noise 가 같은 좌표에 중첩된 복사본을 분리한다. 논문 ablation 에서 정규화를 제거하면 5.82 dB, noise 를 제거하면 2.31 dB 떨어진다. 세 축을 따로 이식하면 이득이 나오지 않는다.

### 발견 2 — 정규화는 ray 와 무관하게 걸리므로, gradient 를 받지 못하는 primitive 도 회수된다

학습이 끝난 bonsai 모델에서 어떤 뷰에서도 피격되지 않은 primitive 중 살아있는 것은 MCMC 1,770 개, 휴리스틱 5,908 개다. 휴리스틱 쪽은 그중 15.4% 가 정확히 opacity 0.01 로, density reset 이 설정한 값에서 움직이지 않았다. prune 조건이 0.005 미만이라 영구히 남는다. **정규화가 없으면 "보이지 않으면 opacity 가 내려간다" 는 전제가 성립하지 않는다.**

### 발견 3 — 원 3DGRT 에서 MCMC 는 완주하며, 종료는 sparse 와 결합했을 때만 발생한다

sparse 없이 bonsai downsampling 2 에서 30,000 step 을 완주했다. 품질은 +0.134 dB, 시간은 34% 증가다. 같은 코드에 sparse ray 를 결합한 8 월 11 일 실험은 15 회 중 13 회가 CUDA illegal memory access 로 종료됐다. 두 조건의 차이가 sparse 인지 해상도인지는 아직 가르지 못했다.

### 지난 리포트 (2026-08-11) 로부터

- 지난번에 알아낸 것: MCMC 를 sparse 학습에 결합하면 품질은 유지되나 22% 느리고, 반복 실행 15 회 중 13 회가 CUDA illegal memory access 로 종료된다.
- 그때 몰랐던 것: MCMC 의 세 요소가 각각 무엇을 담당하는지, 그리고 종료의 원인이 어느 층위에 있는지.
- 이번에 하기로 한 것: 논문과 구현을 함께 읽어 구조를 정리하고, 우리 측정으로 뒷받침되는 부분과 그렇지 않은 부분을 구분한다.

### 합의 사항 → 상태

- **[완료] 세 축의 역할 정리** — 2 절.
- **[완료] 원 3DGRT 에서의 MCMC 동작 확인** — 3 절.
- **[부분] 종료 원인 규명** — 커널에 0 나눗셈 경로가 있다는 것까지 확인했고, 그것이 실제 종료의 원인인지는 재현하지 못했다. 8 월 11 일 리포트의 "원인은 비동기 실행 경쟁" 은 근거가 `CUDA_LAUNCH_BLOCKING=1` 로 증상이 사라졌다는 것뿐이므로, 원인 제거인지 증상 은폐인지 구분되지 않는다 🔴


### 다음

- sparse 학습에 MCMC의 메커니즘을 적절히 수정해 결합해보고 성능 평가해야 한다. 


---

## 1. 3DGS 의 densification 구조

3DGS 의 Adaptive Density Control 은 네 가지 규칙으로 이루어진다.

| 규칙 | 조건 | 동작 |
|---|---|---|
| clone | gradient 누적 > 0.0002, 크기가 작음 | 복제 |
| split | gradient 누적 > 0.0002, 크기가 큼 | 둘로 나누고 위치를 난수로 흩음 |
| prune | opacity < 0.005 | 배열에서 제거 |
| opacity reset | 3,000 step 마다 | `min(opacity, 0.01)` 로 설정 |

네 규칙 모두 조건이 코드에 상수로 고정되어 있다. MCMC 논문은 이 상수들이 장면마다 잘 일반화되지 않고, 하이퍼파라미터만 보고 Gaussian 이 몇 개나 쓰일지 예측할 수 없어 계산·메모리 예산을 미리 통제하기 어렵다고 지적한다.

MCMC 는 규칙을 하나씩 개선하지 않는다. 네 규칙을 통째로 걷어내고 다른 세 장치로 같은 일을 시킨다. 대응은 다음과 같다.

| 3DGS 규칙 | MCMC 에서의 대체 |
|---|---|
| prune | 손실 항의 정규화 (opacity 를 임계값 아래로 내림) |
| clone / split | 재배치 (dead 자리를 유망한 위치의 복사본으로 덮어씀) |
| opacity reset | 없음 (정규화가 상시로 같은 일을 함) |
| — | position noise (추가) |

## 2. 세 축이 맞물리는 방식

세 장치는 순환을 이룬다.

```
정규화 →  기여가 없는 primitive 의 opacity 를 0.005 아래로 감소시켜 dead 로 만든다
재배치 →  그 자리를 유망한 위치의 복사본으로 채운다 (동일 좌표에 중첩)
noise  →  중첩된 것을 분리하여 각자 위치를 탐색하게 한다
```

**MCMC 에는 제거 경로가 없다.** 배열 길이가 고정이고, dead primitive 의 파라미터를 alive primitive 의 복사본으로 덮어쓴다. 논문은 opacity 가 0.005 미만인 것을 dead, 그 이상을 alive 로 부르며 이 리포트도 같은 용어를 쓴다.

```
3DGS    [G1, G2, G3, G4, G5] → prune → [G1, G2, G4, G5] → clone → [G1, G2, G4, G5, G2']
MCMC    [G1, G2, G3, G4, G5] → relocate → [G1, G2', G2'', G4, G5]      길이 불변
```

길이가 고정이므로 재배치가 쓸 수 있는 자리는 dead 뿐이고, dead 를 발생시키는 장치는 정규화 하나다. 여기서 1 절의 prune 이 정규화로 대체된 의미가 드러난다. prune 은 조건을 만족하는 것을 골라 지우는 사후 판정이지만, 정규화는 모든 primitive 의 opacity 를 상시로 낮추고 재현 손실이 그것을 상쇄하지 못하는 것만 임계값 아래로 내려가게 한다. 판정 기준이 코드 상수에서 손실 항의 가중치 `λ` 로 옮겨간다.

재배치는 부모와 정확히 같은 좌표에 복사본을 놓는다 ( `μ_new = μ_old`). 중첩 상태로 두면 primitive 하나와 동일하므로 분리가 필요하고, 그것을 noise 가 담당한다. 재배치가 opacity 를 `o_new = 1 − ᴺ√(1−o_old)` 로 낮게 설정하는데, noise 크기가 opacity 에 반비례하도록 게이트가 걸려 있으므로 재배치 직후에는 큰 noise 를 받고 학습이 진행되어 opacity 가 오르면 noise 가 잦아든다.

셋 중 하나가 없으면 chain이 성립하지 않는다. 정규화가 없으면 임계값 아래로 내려가는 primitive 가 없어 재배치할 대상이 생기지 않고, noise 가 없으면 중첩 상태가 유지되어 다시 기여를 잃으며, 재배치가 없으면 dead 가 빈 공간에 남는다.

논문 ablation (MipNeRF 360, 무작위 초기화) 이 이를 뒷받침한다.

| 조건 | PSNR |
|---|---:|
| 3DGS | 27.89 |
| MCMC + 원래 3DGS 손실 (정규화 제거) | 23.90 |
| MCMC, λnoise = 0 | 27.41 |
| MCMC 전체 | 29.72 |

정규화만 제거했을 때 3DGS 보다도 낮은 23.90 이 나오는 것이 chain 구조의 근거다. 정규화만 빼면 개수가 상한에 도달한 뒤로 상태가 변하지 않아, prune 도 densification 도 없는 모델이 된다.

## 3. 원 3DGRT 에서의 동작

원 3DGRT 에 MCMC 를 붙인 조건은 bonsai downsampling 2 에서 30,000 step 을 완주했다. **원 MCMC 코드 자체는 정상 동작한다.** 즉 현재는 MCMC 를 sparse ray 학습에 이식할 때의 구현상 버그가 있는 것으로 추정된다.

| 조건 | λo (= λs) | PSNR | 시간 | 전 구간 평균 primitive |
|---|---:|---:|---:|---:|
| 휴리스틱 densification (8 월 4 일) | 0 | 31.893 | 133 분 | 1,266,310 |
| MCMC (8 월 13 일) | 0.01 | 32.027 | 178 분 | 1,348,765 |

λo 는 손실에 붙는 opacity 정규화 항의 가중치이며,기본값 0.01 을 그대로 썼다 (λs 도 같은 값이다). 

2 절에서 정리한 대로 이 항이 dead 를 발생시키는 유일한 장치이므로, λo 는 primitive 가 회수되는 속도를 정하고 그것이 곧 재배치 빈도가 된다. 

λo 를 키우면 더 많은 primitive 가 임계값 아래로 내려가 재배치가 잦아지고 시간이 늘어나며, 낮추면 반대다. 휴리스틱 조건의 0 은 정규화 항 없이 prune 으로 제거하는 설정을 뜻한다 

품질은 +0.134 dB, 시간은 34% 증가다.

논문도 같은 크기를 보고한다. Room 장면에서 3DGS 25 분, MCMC 42 분 (λo = 0.01) 으로 68% 느리다. 

λo 를 0.001 로 낮추면 30 분이 되고 PSNR 은 32.5 에서 32.4 로만 떨어진다. 개수가 상한으로 고정되어 있으므로 시간 차이는 개수가 아니라 재배치 빈도에서 온다. 

다만 이 0.001 수치는 논문이 Room 장면에서 3DGS 기반으로 잰 값이고, 우리 3DGRT bonsai 조건에서 같은 교환비가 나오는지는 아직 재지 않았다.

## 4. sparse-ray에 적용할 수정안

### 4.1 forward / backward 의 비대칭

sparse 구조는 forward 에서 전체 픽셀 중 일부만 렌더한다. bonsai downsampling 2 에서 블록 4 기준으로 1,619,801 픽셀 중 100,751 개, 비율 `r = 1/16` 이다.

backward 에서 재현 손실은 표본으로 뽑힌 ray 의 평균이다.

```python
loss = l1_weight * torch.abs(out["pred_rgb"] - second.rgb_gt).mean()
```

평균으로 나누므로 primitive 하나가 받는 gradient 의 기댓값은 dense 와 같다. 달라지는 것은 **보정이 도착하는 빈도**다. 한 primitive 가 gradient 를 받으려면 그 primitive 를 지나는 ray 가 표본에 뽑혀야 하고, 그 확률이 `r` 배로 줄어든다. 대신 뽑혔을 때의 크기가 `1/r` 배가 되어 기댓값이 맞춰진다. dense 가 매 step 1 만큼 보정한다면, sparse 는 16 step 에 한 번 16 만큼 보정한다.

문제는 MCMC 의 세 장치가 모두 ray 와 무관하게 동작한다는 데 있다.

| 장치 | 코드 | ray 의존 | 적용 대상 | 빈도 |
|---|---|---|---|---|
| 정규화 | `loss.lambda_opacity`, `lambda_scale` | 없음 | 전체 primitive | 매 step |
| noise | `perturb_gaussians()` | 없음 | 전체 primitive | 매 step |
| 재배치 | `relocate()` / `add_new_gaussians()` | 없음 | dead 전체 | 100 step 마다 |

```python
# threedgrut/strategy/mcmc.py — 피격 여부를 보지 않고 전체에 더한다
noise = torch.randn_like(positions) * op_sigmoid(1 - densities) * noise_lr * current_lr
self.model.positions.add_(torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1))
```

**세 장치는 step 을 세고, 재현 손실은 ray 를 센다.** sparse 는 이 둘의 환산율을 `r` 만큼 어긋나게 만든다. dense 에서 조정된 상수를 그대로 쓰면 primitive 하나 기준으로 noise 와 정규화는 16 배 자주, 재현 보정은 그대로 들어온다.

### 4.2 바꿔야 할 숫자

두 가지 불변량을 기준으로 잡았다. noise 는 보정 1 회당 누적 이동량, 정규화는 보정 사이에 쌓이는 결정적 하강분이다. 두 값을 dense 와 같게 두면 다음과 같다.

| 파라미터 | 파일 | 현재 | 제안 | 일반형 |
|---|---|---:|---:|---|
| `perturb.frequency` | `configs/strategy/mcmc.yaml` | 1 | 16 | `1/r` |
| `perturb.noise_lr` | 같음 (위와 택일) | 500,000 | 125,000 | `√r ×` 기존값 |
| `loss.lambda_opacity` | `configs/base_mcmc.yaml` | 0.01 | 0.0025 | `√r ×` 기존값 |
| `loss.lambda_scale` | 같음 | 0.01 | 0.0025 | `√r ×` 기존값 |
| `relocate.frequency` | `configs/strategy/mcmc.yaml` | 100 | 400 | 4.2.3 참고 |
| `add.frequency` | 같음 | 100 | 400 | `relocate` 와 일치 |

#### 4.2.1 noise — frequency 16 또는 noise_lr 125,000

noise 는 위치의 random walk 이고 gradient 가 복원력이다. dense 에서는 보정 1 회당 noise 1 회이지만, `r = 1/16` 에서는 보정 1 회당 noise 16 회가 들어간다. 보정을 받지 못한 채 누적되는 이동량이 √16 = 4 배로 커진다.

두 방식 중 하나로 되돌린다. `perturb.frequency` 를 16 으로 올리면 보정 1 회당 noise 1 회가 되어 dense 와 같은 비율이 된다. frequency 를 1 로 두려면 한 번의 크기를 줄여야 하고, 16 회분 누적 분산을 1 회분과 맞추려면 `noise_lr` 이 `500,000 / √16 = 125,000` 이 된다. 앞쪽이 구조적으로 명확하고 뒤쪽이 학습 곡선을 매끄럽게 하므로, 먼저 frequency 16 을 시험하는 편이 낫다.

#### 4.2.2 정규화 — λo, λs 0.0025

정규화는 매 step 모든 primitive 의 opacity 를 같은 크기로 낮춘다. 재현 손실은 16 step 에 한 번 도착한다. 보정을 기다리는 동안 opacity 가 결정적으로 내려가고, 그 사이에 임계값 0.005 를 지나면 기여하고 있던 primitive 도 dead 로 판정되어 재배치로 덮어써진다. 발견 2 에서 확인한 "ray 와 무관하게 걸린다" 는 성질이 dense 에서는 장점이고 sparse 에서는 위험 요인이 된다.

`√r` 배인 0.0025 는 보정 사이 하강분을 dense 와 맞추는 값이다. 논문이 Room 에서 0.001 까지 낮췄을 때 품질 손실이 0.1 dB 였으므로 0.0025 는 그 범위 안이다.

#### 4.2.3 재배치 주기 — 100 에서 400

재배치는 부모와 같은 좌표에 복사본을 놓고, 분리와 정착을 이후 step 의 gradient 에 맡긴다. dense 에서 100 step 은 보정 100 회지만 `r = 1/16` 에서는 6.25 회다. 다음 재배치가 오기 전에 정착이 끝나지 않는다.

`1/r` 을 그대로 곱한 1,600 은 쓸 수 없다. `relocate.end_iteration` 이 25,000 이고 시작이 500 이므로 재배치 기회가 15 회로 줄어 상한까지 채우지 못한다. 400 이면 61 회가 확보되고 보정은 25 회로 늘어난다. 이 값은 두 제약의 절충이므로 100, 400, 1,600 을 함께 재서 정하는 편이 맞다.

#### 4.2.4 바꾸지 않는 값

`opacity_threshold` 0.005 와 `binom_n_max` 51 은 그대로 둔다. 앞은 dead 판정 기준일 뿐 하강 속도를 정하지 않으므로 4.2.2 에서 이미 다뤄지고, 뒤는 재배치 시 나눠 담을 최대 개수라 ray 표본과 무관하다. `max_n_gaussians` 는 비교 대상과 같은 1,484,957 을 유지한다.


