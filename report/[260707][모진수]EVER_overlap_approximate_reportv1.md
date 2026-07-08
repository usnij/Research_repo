# EVER Overlap Approximation 구현 구조 정리

## 1. 연구 목표

EVER의 렌더링 과정에서 ray 위에 여러 Gaussian이 겹칠 경우, 기존 방식은 각 Gaussian의 `entry`, `exit` event를 모두 정렬하고 모든 event마다 `SplineState.update()`를 수행한다. 이 방식은 ray 위의 density/color 변화를 정확하게 적분할 수 있지만, overlap이 많은 ray에서는 update 수가 증가한다.

본 실험의 목표는 이 overlap 처리 과정 중 일부를 근사하여 ray당 event/update 수를 줄이고, 이를 통해 렌더링 속도 향상 가능성을 확인하는 것이다.

## 2. 기존 EVER의 Overlap 처리 구조

EVER의 fast renderer는 OptiX/Slang shader stage 기준으로 다음 흐름을 가진다.

```text
intersection → anyhit → raygeneration(raygen)
```

보고서에서는 overlap 처리와 직접 관련된 위 세 단계를 중심으로 설명한다. 분석한 핵심은 `anyhit`의 payload 구성과 `raygeneration`의 compositing 처리이다.

### 2.1 intersection 단계: Gaussian과 ray의 교차 구간 계산

intersection 단계에서는 ray가 각 Gaussian ellipsoid와 만나는지를 계산한다. 하나의 Gaussian이 ray와 교차하면 ray 위에서 하나의 구간(interval)이 생긴다.

```text
Gaussian A: [entry_A, exit_A]
Gaussian B: [entry_B, exit_B]
Gaussian C: [entry_C, exit_C]
```

여기서 `entry`와 `exit`은 실제 3D 좌표가 아니라 ray parameter `t` 값이다.

```text
position(t) = ray_origin + t * ray_direction
```

즉 `entry_A`는 ray가 A Gaussian 내부로 들어가는 t 값이고, `exit_A`는 A Gaussian 밖으로 나가는 t 값이다.

### 2.2 anyhit 단계: entry/exit를 payload event로 저장

기존 EVER는 Gaussian interval을 그대로 처리하지 않고, interval의 시작과 끝을 두 개의 event로 변환한다.

```text
entry event: 해당 Gaussian의 density/color가 active set에 추가됨
exit event : 해당 Gaussian의 density/color가 active set에서 제거됨
```

anyhit 단계에서는 intersection에서 계산된 교차 정보를 payload에 저장한다. 기존 EVER의 payload item은 개념적으로 다음과 같다.

```text
(event_t, tri_id)
```

여기서 `event_t`는 해당 entry 또는 exit의 ray parameter 값이고, `tri_id`는 어떤 Gaussian의 entry/exit인지를 나타낸다.

```text
tri_id = 2 * prim_id + 1  → entry event
tri_id = 2 * prim_id + 0  → exit event
```

예를 들어 ray 위에서 세 Gaussian이 다음과 같이 배치되어 있다고 하자.

```text
A_entry ---- B_entry ---- A_exit --- C_entry -- B_exit ---- C_exit
```

이를 간단히 표시하면 다음과 같다.

```text
A------B------A---C--B----C
```

기존 EVER의 payload event stream은 다음과 같이 구성된다.

```text
EVER exact event stream:
[+A, +B, -A, +C, -B, -C]
```

여기서 `+A`는 A Gaussian의 entry event, `-A`는 A Gaussian의 exit event를 의미한다.

### 2.3 raygeneration 단계: event stream을 순서대로 적분

raygeneration(raygen) 단계에서는 anyhit이 payload에 저장한 event stream을 읽고 실제 color/transmittance를 계산한다.

기존 EVER의 raygen은 payload에 저장된 event를 t 순서대로 처리한다.

```python
for event_t, tri_id in payload:
    ctrl_pt = get_ctrl_pt(tri_id, event_t)
    state = SplineState.update(state, ctrl_pt)
```

`SplineState.update()`는 현재 event 하나만 처리하는 함수처럼 보이지만, 실제로는 다음 두 작업을 같이 수행한다.

```text
1. 이전 event 위치부터 현재 event_t까지의 구간을 현재 active density/color로 적분
2. 현재 event의 dirac을 active density/color에 반영
```

따라서 EVER는 event 사이 구간마다 현재 active Gaussian들의 density/color를 이용해 volume rendering을 수행한다.


위 예시의 event stream을 raygen에서 처리하면 다음 active 구간들이 생긴다.

```text
[+A, +B, -A, +C, -B, -C]

구간 1: A only
구간 2: A + B
구간 3: B only
구간 4: B + C
구간 5: C only
```

즉 EVER의 정확한 overlap 처리는 아래와 같이 요약할 수 있다.

```text
intersection:
  Gaussian별 [entry, exit] 계산

anyhit:
  [entry, exit]를 +event / -event로 payload에 저장

raygen:
  event stream을 순서대로 읽으면서
  active density/color를 정확히 갱신하고 구간별 적분 수행
```

이 방식의 장점은 Gaussian들이 복잡하게 겹치더라도 ray 위의 density 변화가 정확히 반영된다는 점이다. 반면 단점은 overlap이 많을수록 event 수가 많아지고, 그만큼 `SplineState.update()` 호출 수가 증가한다는 점이다.

## 3. z-thickness 논문 아이디어와 EVER로의 매핑

참고한 z-thickness 계열 접근에서는 fragment를 단순한 depth point가 아니라 두께를 가진 구간으로 본다.

```text
fragment = [front, back]
sort key = back depth
```

이를 EVER의 Gaussian-ray intersection에 대응시키면 다음과 같다.

```text
front = entry
back  = exit
Gaussian interval = [entry, exit]
```


기존 EVER가 다음처럼 `entry`, `exit` event를 모두 정렬한다면,

```text
[+A, +B, -A, +C, -B, -C]
```

근사 접근에서는 각 Gaussian을 interval 단위로 보고, `exit`을 기준으로 정렬한다.

```text
A interval: [A_entry, A_exit]
B interval: [B_entry, B_exit]
C interval: [C_entry, C_exit]

exit-sorted interval stream:
[A, B, C] 또는 exit 순서에 따른 interval list
```

초기 아이디어는 이 exit-sorted interval stream에서 서로 겹치는 interval들을 하나의 cluster로 합치는 것이었다.


## 4. 예시 상황 설정

아래와 같은 ray 위 Gaussian 배치를 예시로 둔다. A, B, C는 서로 연결된 overlap을 만들고, D와 E는 앞 cluster와도 서로 간에도 겹치지 않는 상황으로 설정한다.

```text
A------B------A---C--B----C        D---D       E---E
```

각 Gaussian의 `entry`, `exit`, density, color를 임의로 다음과 같이 설정한다.

| Gaussian | entry t | exit t | density | color |
|---|---:|---:|---:|---|
| A | 1.0 | 4.0 | 0.8 | Red |
| B | 2.0 | 6.0 | 0.5 | Green |
| C | 5.0 | 7.0 | 1.0 | Blue |
| D | 8.5 | 9.5 | 0.7 | Yellow |
| E | 11.0 | 12.0 | 0.6 | Cyan |

따라서 ray 위 event 순서는 다음과 같다.

```text
t=1.0 : +A
t=2.0 : +B
t=4.0 : -A
t=5.0 : +C
t=6.0 : -B
t=7.0 : -C
t=8.5 : +D
t=9.5 : -D
t=11.0: +E
t=12.0: -E
```

기존 EVER의 payload event stream은 다음과 같다.

```text
[+A, +B, -A, +C, -B, -C, +D, -D, +E, -E]
```

## 5. 기존 EVER의 compositing 진행

기존 EVER는 각 event마다 `SplineState.update()`를 호출한다. `update(event)`는 이전 event 위치부터 현재 event 위치까지의 구간을 현재 active density/color로 적분한 뒤, 현재 event를 active set에 반영한다.

위 예시에서 update 호출은 다음 순서로 진행된다.

| update 호출 | 적분 구간 | active Gaussian | 구간 density | 구간 optical thickness |
|---|---|---|---:|---:|
| `update(+A)` | ray start → 1.0 | 없음 | 0.0 | 0.0 |
| `update(+B)` | 1.0 → 2.0 | A | 0.8 | 0.8 × 1 = 0.8 |
| `update(-A)` | 2.0 → 4.0 | A + B | 1.3 | 1.3 × 2 = 2.6 |
| `update(+C)` | 4.0 → 5.0 | B | 0.5 | 0.5 × 1 = 0.5 |
| `update(-B)` | 5.0 → 6.0 | B + C | 1.5 | 1.5 × 1 = 1.5 |
| `update(-C)` | 6.0 → 7.0 | C | 1.0 | 1.0 × 1 = 1.0 |
| `update(+D)` | 7.0 → 8.5 | 없음 | 0.0 | 0.0 |
| `update(-D)` | 8.5 → 9.5 | D | 0.7 | 0.7 × 1 = 0.7 |
| `update(+E)` | 9.5 → 11.0 | 없음 | 0.0 | 0.0 |
| `update(-E)` | 11.0 → 12.0 | E | 0.6 | 0.6 × 1 = 0.6 |

기존 EVER가 실제로 compositing하는 density 구간은 다음과 같다.

```text
[1.0, 2.0]   : A only
[2.0, 4.0]   : A + B
[4.0, 5.0]   : B only
[5.0, 6.0]   : B + C
[6.0, 7.0]   : C only
[8.5, 9.5]   : D only
[11.0, 12.0] : E only
```

따라서 기존 EVER는 active set이 바뀌는 모든 구간을 따로 보존한다. 이 예시에서는 총 10번의 update 호출이 발생한다.

## 6. Exit-Sorted 근사 구조의 compositing 진행

근사 구조에서는 payload에 `entry/exit event`를 각각 저장하지 않고, Gaussian interval 자체를 저장한다.

```text
A interval = [1.0, 4.0]
B interval = [2.0, 6.0]
C interval = [5.0, 7.0]
D interval = [8.5, 9.5]
E interval = [11.0, 12.0]
```

payload는 `exit` 기준으로 정렬되므로 다음 순서가 된다.

```text
[(A_exit=4.0,  A, A_entry=1.0),
 (B_exit=6.0,  B, B_entry=2.0),
 (C_exit=7.0,  C, C_entry=5.0),
 (D_exit=9.5,  D, D_entry=8.5),
 (E_exit=12.0, E, E_entry=11.0)]
```

### 6.1 cluster 생성 과정

첫 번째 interval A를 읽으면 cluster가 새로 시작된다.

```text
cluster = {A}
cluster_front = 1.0
cluster_exit  = 4.0
```

두 번째 interval B를 읽는다.

```text
B_entry = 2.0 <= cluster_exit = 4.0
```

따라서 B는 현재 cluster와 overlap한다고 판단되어 같은 cluster에 들어간다.

```text
cluster = {A, B}
cluster_front = 1.0
cluster_exit  = max(4.0, 6.0) = 6.0
```

세 번째 interval C를 읽는다.

```text
C_entry = 5.0 <= cluster_exit = 6.0
```

따라서 C도 같은 cluster에 들어간다.

```text
cluster = {A, B, C}
cluster_front = 1.0
cluster_exit  = max(6.0, 7.0) = 7.0
```

네 번째 interval D를 읽는다.

```text
D_entry = 8.5 > cluster_exit = 7.0
```

이 시점에서 D는 현재 cluster와 overlap하지 않는다고 판단된다. 따라서 `{A, B, C}` cluster는 여기서 완성되고 flush된다.

```text
cluster_ABC = [1.0, 7.0]
```

이후 D는 새 cluster를 시작한다.

```text
cluster = {D}
cluster_front = 8.5
cluster_exit  = 9.5
```

다섯 번째 interval E를 읽는다.

```text
E_entry = 11.0 > cluster_exit = 9.5
```

따라서 D cluster도 여기서 완성되고 flush된다.

```text
cluster_D = [8.5, 9.5]
```

이후 E는 새 cluster를 시작한다.

```text
cluster = {E}
cluster_front = 11.0
cluster_exit  = 12.0
```

payload가 끝났으므로 마지막 E cluster도 flush된다.

```text
cluster_E = [11.0, 12.0]
```

결과적으로 근사 구조에서는 다음 세 개의 cluster가 만들어진다.

```text
cluster_ABC = {A, B, C}
cluster_D   = {D}
cluster_E   = {E}
```

### 6.2 cluster를 uniform medium으로 변환

A, B, C cluster의 optical thickness는 다음과 같다.

| Gaussian | length | density | tau |
|---|---:|---:|---:|
| A | 4.0 - 1.0 = 3.0 | 0.8 | 2.4 |
| B | 6.0 - 2.0 = 4.0 | 0.5 | 2.0 |
| C | 7.0 - 5.0 = 2.0 | 1.0 | 2.0 |

```text
cluster_ABC_tau = 2.4 + 2.0 + 2.0 = 6.4
cluster_ABC_len = 7.0 - 1.0 = 6.0
cluster_ABC_density = 6.4 / 6.0 = 1.0667
cluster_ABC_color = (2.4 Red + 2.0 Green + 2.0 Blue) / 6.4
```

D와 E는 각각 혼자 cluster가 되므로 원래 interval과 동일한 uniform medium으로 유지된다.

```text
cluster_D_density = 0.7
cluster_D_color = Yellow

cluster_E_density = 0.6
cluster_E_color = Cyan
```

### 6.3 근사 구조에서의 update 호출

각 cluster는 pseudo entry/exit event로 변환된다.

```text
cluster_ABC: +cluster_ABC at 1.0,  -cluster_ABC at 7.0
cluster_D  : +cluster_D   at 8.5,  -cluster_D   at 9.5
cluster_E  : +cluster_E   at 11.0, -cluster_E   at 12.0
```

따라서 근사 구조의 update 호출은 다음과 같다.

| update 호출 | 적분 구간 | active medium | density | optical thickness |
|---|---|---|---:|---:|
| `update(+cluster_ABC)` | ray start → 1.0 | 없음 | 0.0 | 0.0 |
| `update(-cluster_ABC)` | 1.0 → 7.0 | cluster_ABC | 1.0667 | 1.0667 × 6 = 6.4 |
| `update(+cluster_D)` | 7.0 → 8.5 | 없음 | 0.0 | 0.0 |
| `update(-cluster_D)` | 8.5 → 9.5 | cluster_D | 0.7 | 0.7 × 1 = 0.7 |
| `update(+cluster_E)` | 9.5 → 11.0 | 없음 | 0.0 | 0.0 |
| `update(-cluster_E)` | 11.0 → 12.0 | cluster_E | 0.6 | 0.6 × 1 = 0.6 |

기존 EVER와 비교하면 다음과 같다.

```text
기존 EVER:
[+A, +B, -A, +C, -B, -C, +D, -D, +E, -E]
→ 10번 update
→ A only, A+B, B only, B+C, C only, D only, E only 구간을 각각 compositing

근사 구조:
[+cluster_ABC, -cluster_ABC, +cluster_D, -cluster_D, +cluster_E, -cluster_E]
→ 6번 update
→ ABC는 하나의 uniform medium으로 압축하고, D/E는 각각 단일 cluster로 compositing
```

따라서 이 예시에서 근사 구조는 A/B/C의 overlap 구간을 하나의 cluster로 합쳐 update 수를 줄인다. 반면 D와 E처럼 overlap되는 interval이 없는 경우에는 각각 독립 cluster로 처리되어 기존 단일 Gaussian interval과 거의 같은 방식으로 compositing된다.

## 7. 이후 개선한 근사 구조들

초기 single-cluster 방식은 cluster 내부 순서 정보를 크게 잃기 때문에, 여러 개선을 시도했다.

### 7.1 Tau-weighted color

가장 단순한 방식은 optical thickness 기준 평균이다.

```text
cluster_color = sum(tau_i * color_i) / sum(tau_i)
```

이는 전체 optical thickness는 보존하지만, front-to-back compositing 순서는 보존하지 못한다.

### 7.2 Front-biased color

entry 순서에 따라 앞쪽 Gaussian의 영향이 더 크게 반영되도록 했다.

```text
w_i = T_i * (1 - exp(-tau_i))
cluster_color = sum(w_i * color_i) / sum(w_i)
```

여기서 `T_i`는 i번째 Gaussian 앞까지의 transmittance 근사값이다.

### 7.3 Small-cluster exact

cluster 크기가 작을 때는 근사하지 않고 cluster 내부 entry/exit를 다시 정렬해서 exact하게 처리했다.

```text
if cluster_size <= 3:
    local exact event processing
else:
    approximate cluster processing
```

목적은 작은 overlap에서는 정확도를 유지하고, 큰 overlap에서만 속도를 얻는 것이었다.

### 7.4 Lobe compression

하나의 cluster를 하나의 uniform medium으로 만드는 것이 너무 거칠다고 판단하여, cluster를 2개 또는 3개의 lobe로 나누는 방식도 시도했다.

```text
cluster
→ lobe_1, lobe_2, lobe_3
```

각 lobe는 해당 구간의 optical thickness와 color mass를 보존하도록 만들었다.

```text
lobe_tau = sum(density_i * overlap_len_i)
lobe_color = sum(color_i * density_i * overlap_len_i) / lobe_tau
```

이 방식은 cluster 내부의 depth 분포를 일부 보존하려는 시도였다.

### 7.5 Fixed-k adaptive layer

OIT의 adaptive transparency 계열을 참고하여, ray 위 interval들을 최대 K개의 layer로만 유지하는 방식도 구현했다.

```text
새 interval 삽입
layer 수가 K 초과
→ 인접 layer 중 combined span이 가장 작은 두 layer merge
```

merge된 layer는 두 layer의 내부 compositing 결과와 optical thickness를 보존하는 equivalent uniform layer로 만들었다.

CPU probe에서는 가능성이 있어 보였지만, shader에서 exit-sorted payload 위에 구현했을 때는 품질이 회복되지 않았다.

## 8. 구조적 한계가 드러난 부분

중요한 진단은 `local_exact` 실험이었다.

이 모드는 exit-sorted cluster 자체는 유지하되, cluster 내부에서는 entry/exit event를 다시 정렬해서 EVER와 같은 방식으로 exact하게 처리한다.

즉 cluster color 수식 문제를 제거하고, cluster 내부는 최대한 정확하게 만든 것이다.

그런데도 품질이 회복되지 않았다.

이를 통해 문제의 핵심이 단순히 cluster color 수식에 있는 것이 아니라, exit-sorted payload/window 구조 자체에 있음을 확인했다.

구체적으로는 다음 문제가 있다.

```text
어떤 Gaussian의 entry는 이미 현재 ray 위치보다 앞에 있음
하지만 exit는 payload window 밖의 먼 위치에 있음
```

이 Gaussian은 현재 구간의 density에 기여해야 하지만, exit 기준 payload에서는 아직 선택되지 않을 수 있다. 따라서 raygen에서 보는 active set이 불완전해진다.

즉 exit-sorted interval payload는 다음 정보를 놓칠 수 있다.

```text
현재 구간에 이미 영향을 주고 있지만
exit가 아직 멀어서 payload에 들어오지 않은 long interval
```

이 경우 이후 cluster 수식을 아무리 개선해도, 애초에 필요한 visibility 정보가 누락되어 품질을 회복하기 어렵다.

## 9. 정리

본 구현은 다음 흐름으로 진행되었다.

```text
1. EVER exact event stream 분석
2. Gaussian을 [entry, exit] interval로 해석
3. z-thickness식 exit-sorted interval payload 구현
4. overlap interval cluster 생성
5. cluster를 pseudo uniform medium으로 변환
6. 기존 SplineState.update()에 pseudo entry/exit로 연결
7. tau 평균, front-biased color, lobe compression, small exact, fixed-k layer 등 개선 시도
8. local_exact 진단을 통해 cluster 수식보다 payload/window 구조 문제가 큼을 확인
```

현재까지의 결론은 다음과 같다.

```text
단순 exit-sorted interval merge는 event 수를 줄이는 데는 효과적이지만,
EVER의 정확한 compositing이 의존하는 entry/exit event stream을 충분히 보존하지 못한다.
따라서 품질을 유지하면서 속도를 얻기 위해서는
payload interval 위에서 cluster를 만드는 방식보다,
entry/exit event stream 위에서 bounded visibility state를 유지하는 방식이 더 적절해 보인다.
```
