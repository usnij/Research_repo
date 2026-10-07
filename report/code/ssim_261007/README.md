# SSIM loss 계산 코드 — dense 경로와 sparse 경로

SSIM loss 계산 코드 정리 



## 1. 파일별 설명

| 순서 | 파일 | 행 | 내용 |
|---:|---|---:|---|
| 1 | [losses.py](losses.py#L31-L33) | 31–33 | `ssim()` 래퍼. dense·sparse 공통 |
| 2 | [trainer.py](trainer.py#L436-L443) | 436–443 | dense DSSIM loss |
| 3 | [train_scene_sparse.py](train_scene_sparse.py#L169-L213) | **169–213** | `draw()`. sparse 격자 생성 |
| 4 | [ssim.cu](ssim.cu#L9-L19) | 9–19 | 11×11 window 상수 |

3 번이 dense 와 sparse 를 갈라놓는 지점이다.

---

## 2. 우리 코드

원본 위치는 `3dgrut/threedgrut/` 와 `3dgrut/experiments/mc_sparse_training/` 이다.

| 파일 | 전체 행 | 볼 곳 | 내용 |
|---|---:|---:|---|
| [losses.py](losses.py) | 33 | [31–33](losses.py#L31-L33) | `ssim()` 래퍼 |
| [trainer.py](trainer.py) | 1,051 | [38](trainer.py#L38) | import |
| | | [436–443](trainer.py#L436-L443) | dense DSSIM loss |
| | | [509–520](trainer.py#L509-L520) | 최종 loss 합산 |
| [train_scene_sparse.py](train_scene_sparse.py) | 2,753 | [169–213](train_scene_sparse.py#L169-L213) | `draw()`. sparse 격자 생성 |
| | | [259–287](train_scene_sparse.py#L259-L287) | 학습 루프의 `draw()` 호출 |
| | | [328–334](train_scene_sparse.py#L328-L334) | sparse L1 과 SSIM |
| | | [902–918](train_scene_sparse.py#L902-L918) | calibration 경로의 같은 패턴 |
| [stratified_mc.py](stratified_mc.py) | 307 | [31–74](stratified_mc.py#L31-L74) | `sample_stratified_pixels()` |

### 2.1 공통 래퍼

[losses.py:31–33](losses.py#L31-L33) 에서 dense 와 sparse 가 같은 함수를 호출한다. `window_size` 와 `size_average` 인자는 받기만 하고 `fused_ssim` 으로 전달되지 않는다. 실제 window 크기는 CUDA 커널에 고정되어 있다. 6 절 참조.

### 2.2 dense 경로

[trainer.py:436–443](trainer.py#L436-L443) 에서 `rgb_pred` 를 `[BS, CH, H, W]` 로 permute 하여 그대로 넣는다. counter downsample 2 기준으로 `1038×1558` 전체다. 합산은 [509–520](trainer.py#L509-L520) 이다.

### 2.3 sparse 경로

[train_scene_sparse.py:169–213](train_scene_sparse.py#L169-L213) 의 `draw()` 가 핵심이다. 


sampling된 ray로 L1 과 SSIM 을 계산하는 지점은 [328–334](train_scene_sparse.py#L328-L334) 이고, densification calibration 쪽의 같은 패턴은 [902–918](train_scene_sparse.py#L902-L918) 이다.

---

## 3. 외부 패키지 `fused_ssim`

| 항목 | 값 |
|---|---|
| 저장소 | https://github.com/rahul-goel/fused-ssim |
| 커밋 | `1272e21a282342e89537159e4bad508b19b34157` (2024-09-16) |
| 라이선스 | [LICENSE](LICENSE) |
| 설치 경로 | `miniconda3/envs/3dgrut/lib/python3.11/site-packages/fused_ssim/` |
| 소스 경로 | `gaussian-splatting/submodules/fused-ssim/` |

아래 다섯 파일이 이 패키지에서 온 것이다. `__init__.py` 와 `setup.py` 는 `fused_ssim` 패키지의 것이며 우리 코드가 아니다.

| 파일 | 전체 행 | 볼 곳 | 내용 |
|---|---:|---:|---|
| [\_\_init\_\_.py](__init__.py) | 41 | [33–41](__init__.py#L33-L41) | `C1`, `C2` 와 호출 |
| | | [9–20](__init__.py#L9-L20) | forward 의 `padding="valid"` 처리 |
| | | [22–32](__init__.py#L22-L32) | backward 의 같은 처리 |
| [ssim.cu](ssim.cu) | 444 | [9–19](ssim.cu#L9-L19) | 11×11 Gaussian 계수 상수 |
| | | [100–180](ssim.cu#L100-L180) | separable convolution |
| | | [255–275](ssim.cu#L255-L275) | SSIM 본식 |
| [ssim.h](ssim.h) | 26 | [1–26](ssim.h#L1-L26) | 함수 선언 |
| [ext.cpp](ext.cpp) | 7 | [1–7](ext.cpp#L1-L7) | pybind 등록 |
| [setup.py](setup.py) | 13 | [1–13](setup.py#L1-L13) | 빌드 설정 |

`padding="valid"` 이면 forward 에서 `ssim_map[:, :, 5:-5, 5:-5]` 로 경계 5 pixel 을 잘라내고, backward 에서도 같은 영역에만 gradient 를 되돌린다. 11×11 window 의 반지름이 5 이기 때문이다.

---

## 4. 파일 목록

| 파일 | 행 | 출처 |
|---|---:|---|
| `losses.py` | 33 | 우리 코드 |
| `trainer.py` | 1,051 | 우리 코드 |
| `train_scene_sparse.py` | 2,753 | 우리 코드 |
| `stratified_mc.py` | 307 | 우리 코드 |
| `__init__.py` | 41 | `fused_ssim` 패키지 |
| `ssim.cu` | 444 | `fused_ssim` 패키지 |
| `ssim.h` | 26 | `fused_ssim` 패키지 |
| `ext.cpp` | 7 | `fused_ssim` 패키지 |
| `setup.py` | 13 | `fused_ssim` 패키지 |
| `LICENSE` | — | `fused_ssim` 패키지 |

모든 파일은 원본 전문이다. 우리 코드는 `3dgrut` 작업 복사본에서, `fused_ssim` 파일은 위 커밋에서 가져왔다.
