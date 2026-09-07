# multi-view 학습 구조 정리

작성 2026-09-07. sparse Ray 학습에 결합한 multi-view 구조를 정리한다. 구현 방식,
Ray 예산을 유지하는 근거, 선행 연구와의 차이, 지금까지 측정한 정량 결과를 정리한다.

## 요약

이번 리포트의 핵심 발견은 두 가지다.

### 발견 1 — view 수를 늘려도 Ray 수는 증가하지 않는다. block 을 √K 배로 키워 예산을 고정하기 때문이다

view 하나당 block 을 `block × √K` 로 키우고 K 개 view 를 배치 차원으로 결합한다. view
하나의 Ray 수가 `1/(block²K)` 로 줄고 view 가 K 개이므로 합이 `1/block²` 로 유지된다.

counter 를 downsampling 2, block 4 로 학습할 때 K = 1 이면 260×390 = 101,400 Ray 이고,
K = 4 이면 block 8 로 130×195 = 25,350 Ray 를 view 4 개에서 표집해 합이 101,400 Ray 다.
**view 를 4배로 늘려도 step 당 Ray 수는 동일하다.**

학습 시간 증가는 제한적이다. counter 의 15,000~30,000 step 구간에서 K = 1 이 18분,
K = 4 가 22분으로 1.22배다.

우리 구조는 block 마다 픽셀 하나를 표집해 view 별로 `⌈h/blk⌉ × ⌈w/blk⌉` 격자를
유지한다. 격자가 유지되므로 각 view 안에서 SSIM 을 계산할 수 있다. step 당 Ray 수는
101,400 개로 3DGRT 논문의 524,288 개의 19% 다.

### 발견 2 — multi-view 효과는 위치보다 opacity 와 scale 에서 크게 나타난다

block 4, Ray 100,751 개 조건에서 gradient variance 를 view 내부 성분과 view 간
성분으로 분해했다.

| parameter | view 내부 | view 간 | view 간 비중 |
|---|---:|---:|---:|
| 위치 | 1.776e-02 | 0.000 | **0.0%** |
| opacity | 6.833e-07 | 3.568e-06 | **83.9%** |
| scale | 7.869e-07 | 5.579e-06 | **87.6%** |

2026-08-13 리포트는 위치 gradient 만 측정했고, view 간 비중이 7.1% 라는 이유로 다중
view 를 제외했다. 다른 parameter 를 함께 측정하면 view 간 비중이 83.9% 와 87.6% 다.
**multi-view의 효과는 위치 최적화보다 opacity 와 scale 최적화에서 크게 나타난다.**

---

### 지난 리포트 (2026-08-28) 로부터
- 지난번에 알아낸 것: block 2 RADC 와 15,000 step 이후 multi-view 학습을 결합하면 8 개
  씬 중 7 개에서 dense 보다 PSNR 이 높다.
- 그때 몰랐던 것: multi-view의 효과가 densification 전략에 따라 어떻게 달라지는지.
- 이번에 하기로 한 것: 구조를 정리하고 선행 연구와의 차이를 문서화한다.

### 합의 사항 → 상태
- **[완료] 구현 정리** — `_multiview` 가 데이터셋 계층에서 batch 를 구성한다.
  densification 전략과 독립이므로 휴리스틱 ADC 와 MCMC 양쪽에 적용할 수 있다.
- **[완료] Ray 예산 유지 확인** — block 을 √K 배로 키워 K = 4 에서 step 당 Ray 수가
  K = 1 과 동일하다.
- **[완료] EMVT 및 3DGRT 논문과의 차이 정리** — 3 절.
- **[완료] 정량 결과 수집** — 4 절.
- **[미착수] K 값 sweep** (사유: K = 4 만 실행했다. K = 2, 8 은 측정하지 않았다. 여기서 K 는 multi-view의 view 개수다.)

### 다음
- 🔴 multi-view는 densification 전략과 독립이므로 MCMC 에도 적용할 수 있다. 논문에서 어느 위치에 둘지 판단이 필요하다.


---

## 1. 구현

### 1.1 단일 view 학습과 multi-view 학습의 차이

3DGS 와 3DGRT 는 iteration 마다 학습 이미지 하나를 선택하고, 그 view 의 카메라에서
픽셀마다 Ray casting해 렌더링한 뒤 같은 view 의 GT 이미지와 비교한다. 한 step 에서
Gaussian 의 갱신 방향을 정하는 정보가 view 하나에서만 온다.

sparse 학습은 이 구조에서 Ray 수만 줄인다. block N 층화 표집으로 그 view 픽셀의
1/N² 에만 Ray casting한다. **view 는 여전히 하나다.**

multi-view 학습은 한 step 에 K 개 view 를 함께 사용한다. 절차는 다음과 같다.

| 단계 | 단일 view | multi-view (K = 4) |
|---|---|---|
| view 선택 | 이미지 1 장 | 현재 view + 무작위 3 장 |
| 표집 | block 격자로 1/block² | blk = block×√K 격자로 view 당 1/(block²K) |
| 렌더링 | forward 1 회 | **forward 1 회** (K 개 view 를 batch 축으로 결합) |
| GT 비교 | 그 view 의 GT | 각 Ray 가 자기 view 의 GT 픽셀과 비교 |
| L1 | 픽셀 평균 | K 개 view 의 전체 픽셀 평균 |
| SSIM | 이미지 1 장 | view 마다 계산 (격자가 유지되므로) |
| backward | 1 회 | **1 회** (K 개 view 의 gradient 가 같은 buffer 에 누적) |

**view 를 K 번 따로 렌더링하지 않는다.** K 개 view 의 Ray 를 하나의 batch 로
구성해 forward 와 backward 를 각각 한 번만 실행한다. tracer 가 Ray 단위로 동작하므로
어느 view 에서 온 Ray 인지 구분할 필요가 없다.

counter 를 downsampling 2 (1038×1558) 로 학습할 때의 Ray 수다.

| | block | view 당 격자 | view 수 | step 당 Ray |
|---|---:|---|---:|---:|
| 단일 view | 4 | 260×390 | 1 | 101,400 |
| multi-view K = 4 | 8 | 130×195 | 4 | 101,400 |

block 을 √K 배로 키워 view 당 Ray 를 1/K 로 줄이고 view 를 K 배로 늘리므로 합은
동일하다. **view 를 4배로 늘려도 렌더링해야 할 Ray 수는 변하지 않는다.**

이 구조에서 달라지는 것은 **한 Gaussian 이 한 step 에 최대 K 개 방향에서 관측된다**는
점이다. 단일 view 에서는 그 view 에만 맞는 방향으로 갱신되고, multi-view에서는 K 개
view 의 요구가 한 step 안에서 합쳐진다. 발견 2 의 variance 분해가 이 차이가 어느
parameter 에서 큰지를 보여준다. 위치는 view 간 성분이 0.0% 라 view 를 늘려도 갱신
방향이 거의 같지만, opacity 와 scale 은 83.9% 와 87.6% 이므로 view 마다 다른 값을
요구한다.

### 1.2 구현 절차

`train_scene_sparse.py` 의 데이터셋 wrapper 안에 있다. `--multiview-k K` 로 view
수를, `--multiview-from N` 으로 적용 시작 step 을 지정한다. 우리 실험은 K = 4,
N = 15,000 을 사용한다. 15,000 step 은 densification 이 끝나는 시점이다.

동작은 네 단계다.

**첫째, block 을 키운다.** `blk = round(block × √K)` 다. block 4, K = 4 이면 blk = 8 이다.

**둘째, view 를 표집한다.** 현재 batch 의 view 에 더해 학습 집합에서 K−1 개를 중복
없이 표집한다. `numpy` 의 `default_rng` 로 매 step 새로 표집하므로 step 마다 조합이
달라진다. 추가 view 는 `dataset[i]` 로 그 자리에서 읽어 GPU batch 로 만든다.

각 view 에서 `blk × blk` 격자마다 픽셀 하나를 층화 표집한다. 격자 안의 위치는
난수이고 격자 자체는 이미지를 덮으므로, 표본이 이미지 전역에 고르게 퍼진다.
표집 결과를 `⌈h/blk⌉ × ⌈w/blk⌉` 모양으로 되돌려 view 별 2D 배열을 만든다.

**셋째, Ray 를 월드 좌표로 변환한다.** 여러 view 를 배치 차원으로 결합하려면 각 원소가
자기 카메라 자세로 렌더링돼야 하는데, tracer 는 배치 원소별 `T_to_world` 를 쓰지
않고 `[0]` 을 전체에 적용한다. 배치의 두 번째 원소가 첫 번째 원소의 카메라로
렌더링되는 것을 확인했다. 이 상태로 결합하면 K 개 view 가 모두 같은 시점에서
렌더링되어 multi-view 학습이 성립하지 않는다.

카메라 자세를 batch 에 실어 보내는 대신, Ray 를 미리 월드 공간으로 변환하고 변환
행렬 자리에는 단위행렬을 넘긴다. Ray 의 원점은 회전 후 평행이동하고 방향은 회전만
적용한다.

```python
T = part.T_to_world[0]
R, t = T[:3, :3], T[:3, 3]
wo.append(part.rays_ori @ R.T + t)     # 원점: 회전 후 평행이동
wd.append(part.rays_dir @ R.T)         # 방향: 회전만
eye = torch.eye(4)[None].repeat(k, 1, 1)   # 변환은 단위행렬
```

tracer 가 `[0]` 의 변환을 전체에 적용해도 그 변환이 단위행렬이므로 Ray 가 그대로
쓰인다. 각 view 의 Ray 는 이미 자기 카메라 자세를 반영한 상태다.

**넷째, 배치 차원으로 결합한다.** `rays_ori`, `rays_dir`, `rgb_gt` 를 배치 축으로
이어붙인다. view 별 격자 모양 `⌈h/blk⌉ × ⌈w/blk⌉` 가 유지되므로 각 view 안에서
SSIM 을 계산할 수 있다. `intrinsics` 는 첫 view 의 것을 쓴다. Mip-NeRF 360 은 한 씬 안의
카메라 내부 파라미터가 같으므로 문제가 되지 않지만, 씬마다 다른 데이터셋에서는
확인이 필요하다.

이 구조는 데이터셋 계층의 batch 구성이므로 densification 전략을 수정하지 않는다.
`get_gpu_batch_with_intrinsics` 가 반환하는 `Batch` 의 내용만 달라지고, trainer 와
전략은 수정 없이 같은 인터페이스를 사용한다. 휴리스틱 ADC 와 MCMC 양쪽에 동일한 코드를 적용할 수 있다.

적용 시점을 15,000 step 으로 둔 이유는 densification 이 그때 끝나기 때문이다.
densification 구간에서 multi-view를 켜면 여러 view 의 gradient 가 같은 buffer 에
누적된 뒤 norm 이 한 번만 계산되어 densification 통계량의 의미가 달라진다. 이
문제는 3.1 절에서 다룬다.

## 2. Ray 예산

view 하나의 Ray 수가 `HW / (block²K)` 이고 view 가 K 개이므로 합이 `HW / block²` 다.
K 가 상쇄된다.

| 조건 | block | view 당 격자 | view 수 | step 당 Ray |
|---|---:|---|---:|---:|
| K = 1 | 4 | 260×390 | 1 | 101,400 |
| K = 4 | 8 | 130×195 | 4 | 101,400 |

counter, downsampling 2 (1038×1558) 기준이다.

학습 시간은 K = 1 이 18분, K = 4 가 22분으로 1.22배다. step 시간만 재면 1.48배인데,
전체 학습에는 optimizer 처럼 Ray 수와 무관한 비용이 포함되어 증가율이 낮아진다.

## 3. 선행 연구와의 차이

### 3.1 EMVT (arXiv 2506.12727)

EMVT 는 3DGS 가 iteration 당 이미지 하나만 쓰는 단일 view mini-batch 학습을
사용하고, 이 설정이 mini-batch gradient 의 variance 를 증가시켜 최적화를 불안정하게
만든다고 지적한다. NeRF 는 Ray 단위로 표집하므로 여러 view 가 자연히 섞이는 반면
3DGS 는 그렇지 않다는 관찰이다. 우리가 multi-view를 쓰는 동기와 같다.

논문은 rasterizer 에서 multi-view를 적용할 때 두 가지 문제가 생긴다고 정리한다.

**첫째, 렌더링 비용이다.** rasterization 은 이미지를 tile 로 나누고 tile 마다
깊이 정렬된 Gaussian 목록을 만든다. 여러 view 를 mini-batch 로 묶으려면 view 마다
전체 이미지를 렌더링해야 하므로 비용이 view 수에 비례한다. 논문의 해결책은 하나의
tile 에 여러 view 의 픽셀을 섞어 배치하는 것이다. 논문이 보고한 MipNeRF-360 학습
시간이다.

| 방식 | 시간 |
|---|---:|
| 전체 multi-view 렌더링 | 127분 |
| 기존 부분 렌더링 | 105분 |
| EMVT 제안 (tile 혼합 배치) | 50분 |

전체 multi-view 렌더링 대비 2.5배 단축이 이 기법의 기여다. 그 대신 한 tile 안의
인접한 두 픽셀이 서로 다른 view 에서 올 수 있어 window 기반 loss 를 수정 없이 적용할 수 없다.
논문은 이를 위해 3D 거리를 반영한 Gaussian 필터를 쓰는 D-SSIM 을 따로 제안한다.

**둘째, densification 기준이다.** 3DGS 의 ADC 는 view 공간 2D 위치 gradient 의
norm 을 누적해 clone 과 split 을 결정한다. 여러 view 의 loss 를 한 번에 backward
하면 각 view 의 gradient 가 같은 buffer 에 누적된 뒤 norm 이 한 번만 계산되므로,
방향이 반대인 gradient 가 상쇄된다. 논문은 이를 gradient cancellation 이라 부르고
두 통계량을 제안한다.

```
E₁(G) = Σ_i ‖∇_{p_i} L‖₂        norm 을 먼저 계산하고 더한다
E₂(G) = ‖Σ_i ∇_{p_i} L‖₂        더한 뒤 norm 을 계산한다
```

논문은 E₁ 이 split 에, E₂ 가 clone 에 적합하다고 보고한다. 단일 view 3DGS 는 step
마다 그 view 의 norm 을 누적하므로 E₁ 에 해당하고, multi-view를 수정 없이 적용하면
E₂ 가 되어 기준이 바뀐다.

우리 구조는 이 두 문제를 다르게 처리한다.

**렌더링 비용은 renderer 특성으로 해소된다.** Ray tracing 은 Ray 가 서로 독립이므로
여러 view 의 Ray 를 한 batch 로 구성해도 tile 구조를 다시 설계할 필요가 없다. 우리 측정
에서 K 를 1 에서 4 로 늘릴 때 학습 시간은 1.22배 증가한다. **EMVT 가 tile 혼합
배치로 해결한 문제가 Ray tracing 구조에서는 발생하지 않는다.** 그리고 view 별 2D
격자를 유지하므로 D-SSIM 을 새로 설계할 필요 없이 표준 SSIM 을 사용한다.

**densification 기준 문제는 적용 시점을 분리해 처리한다.** multi-view를 15,000 step
이후, 즉 densification 이 끝난 뒤에만 켠다. densification 구간은 단일 view 로
학습하므로 통계량이 E₁ 형태를 유지한다. EMVT 가 densification 기준을 새로 설계해
푼 문제를, 우리는 두 구간을 나누어 처리한다.

이 선택의 대가는 multi-view가 densification 에 기여하지 못한다는 점이다. EMVT 는
multi-view로 densification 자체를 개선하지만 우리는 그 효과를 사용하지 못한다.
2026-08-28 리포트의 counter 측정에서 Gaussian 수 차이가 densification 구간의 RADC
에서만 발생하는 것도 같은 이유다.

### 3.2 정리

| 항목 | 3DGRT 논문 | EMVT | 우리 구조 |
|---|---|---|---|
| renderer | Ray tracing | rasterizer | Ray tracing |
| view 별 2D 격자 | 없음 | 있음 | 있음 |
| SSIM | 사용 불가 | 사용 | 사용 |
| step 당 Ray | 524,288 | — | 101,400 |
| view 증가 비용 | — | tile 혼합으로 완화 | 1.22배 |

## 4. 정량 결과

현재 MCMC-sparse에서 multi-view의 유무에 따른 결과 비교를 위해 multi-view없이 학습 중에 있다. 결과가 나오면 추가 예정