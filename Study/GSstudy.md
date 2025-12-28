# GS(Gaussian Splatting) 구동해보기 

## 데이터셋 기본 디렉토리 구조
파이프라인의 최적화 도구는 기본적으로 COLMAP의 데이터 형식을 기대한다. 
최종적으로 학습에 사용되는 데이터셋은 다음과 같은 구조를 갖추어야 한다.

```bash
|---images          # 왜곡이 제거된(Undistorted) 이미지 파일들
|---sparse          # SfM(Structure-from-Motion) 결과 데이터
    |---0
        |---cameras.bin    # 카메라 파라미터 정보
        |---images.bin     # 이미지 포즈 및 관련 데이터
        |---points3D.bin   # 초기 3D 포인트 클라우드 데이터

```


# `convert.py` (Colmap) 과정 정리

이 문서는 제공된 `convert.py` 코드 기준으로, **이미지 폴더를 COLMAP 결과(카메라/포인트클라우드) + undistort 이미지**로 변환하는 전체 과정을 md에 옮기기 쉽게 정리한 것이다.



## 1) 입력/출력 디렉토리 구조

### 입력(필수)

`convert.py`는 **입력 이미지 폴더명을 `input/`으로 가정**한다.

```text
<source_path>/
 └── input/
     ├── frame_0001.jpg
     ├── frame_0002.jpg
     └── ...
```


### 출력(주요 결과)

COLMAP undistort 후 `source_path`에 다음이 생성된다.

```text
<source_path>/
 ├── distorted/
 │   ├── database.db
 │   └── sparse/0/              # (mapping 결과)
 ├── images/                    # (undistort된 이미지)
 ├── sparse/0/                  # (최종 정리된 sparse 결과)
 └── (옵션) images_2, images_4, images_8
```

---

## 2) 실행 인자(Arguments)

```text
--source_path, -s   (필수) 데이터셋 루트 경로
--no_gpu            COLMAP에서 GPU 사용 안 함
--skip_matching     feature/matching/mapping 단계 스킵 (이미 결과가 있을 때)
--camera            카메라 모델 (기본: OPENCV)
--colmap_executable colmap 바이너리 경로 지정(없으면 'colmap')
--resize            undistort 결과 이미지를 1/2,1/4,1/8로 추가 생성
--magick_executable ImageMagick(magick) 경로 지정(없으면 'magick')
```

---

## 3) 전체 파이프라인 개요

`convert.py`의 큰 흐름은 아래와 같다.

```text
(선택) Feature extraction
(선택) Feature matching
(선택) Mapper(SfM / BA)
Image undistortion
sparse 결과 정리(0 폴더로 이동)
(선택) 이미지 리사이즈(images_2/images_4/images_8)
```

---

## 4) 단계별 상세 과정

### Step A. (skip_matching이 아닐 때만) distorted 작업 폴더 생성

```python
os.makedirs(source_path + "/distorted/sparse", exist_ok=True)
```

* COLMAP 중간 산출물은 `distorted/` 아래에 쌓는다.

---

### Step B. Feature Extraction (SIFT 추출)

실행 명령(개념):

```bash
colmap feature_extractor \
  --database_path <source_path>/distorted/database.db \
  --image_path    <source_path>/input \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model  OPENCV \
  --SiftExtraction.use_gpu    {0 or 1}
```

산출물:

* `<source_path>/distorted/database.db`

  * 각 이미지의 keypoint/descriptor가 저장되는 SQLite DB

실패 시:

* exit code 확인 후 즉시 종료

---

### Step C. Feature Matching (exhaustive_matcher)

실행 명령(개념):

```bash
colmap exhaustive_matcher \
  --database_path <source_path>/distorted/database.db \
  --SiftMatching.use_gpu {0 or 1}
```

산출물:

* `database.db` 내부에 이미지 쌍 매칭 결과 저장

실패 시 즉시 종료

---

### Step D. Mapping (SfM + Bundle Adjustment)

실행 명령(개념):

```bash
colmap mapper \
  --database_path <source_path>/distorted/database.db \
  --image_path    <source_path>/input \
  --output_path   <source_path>/distorted/sparse \
  --Mapper.ba_global_function_tolerance=0.000001
```

* SfM으로 카메라 포즈 + sparse point cloud 생성
* BA tolerance를 줄여 속도/수렴 개선

산출물(대표):

```text
<source_path>/distorted/sparse/0/
 ├── cameras.bin
 ├── images.bin
 └── points3D.bin
```

---

### Step E. Image Undistortion (왜곡 보정)

실행 명령(개념):

```bash
colmap image_undistorter \
  --image_path  <source_path>/input \
  --input_path  <source_path>/distorted/sparse/0 \
  --output_path <source_path>/ \
  --output_type COLMAP
```

역할:

* distorted 카메라 모델/이미지를 **pinhole 기준으로 보정**
* Gaussian Splatting 학습이 안정적으로 되도록 데이터 정규화

산출물(대표):

```text
<source_path>/
 ├── images/      # undistort된 이미지
 └── sparse/      # undistort에 대응되는 sparse 폴더
```

---

### Step F. sparse 결과를 `sparse/0/`로 정리

코드 동작:

* `source_path/sparse` 안의 파일들을 확인
* `source_path/sparse/0` 생성
* `0` 폴더가 아닌 파일들은 `sparse/0`로 이동

목적:

* downstream 코드가 기대하는 표준 구조(`sparse/0`)로 맞추기 위함

---

### Step G. (옵션) 이미지 리사이즈 생성 (`--resize`)

생성 폴더:

```text
images_2/  # 50%
images_4/  # 25%
images_8/  # 12.5%
```

동작:

* `images/`에서 각 파일을 복사한 뒤
* `magick mogrify -resize ...`로 크기 변경

---







# `train.py` 구조 및 학습 과정 정리 (3D Gaussian Splatting)

이 문서는 `train.py`가 **어떤 순서로**, **무엇을**, **왜 수행하는지**를 코드 흐름에 맞춰 설명한다.

---

## 1. 파일 역할 요약

`train.py`는 **3D Gaussian Splatting의 핵심 학습 루프**를 담당한다.

* 입력:

  * COLMAP 또는 Synthetic Scene 데이터
  * 카메라 포즈, 이미지, (옵션) depth 정보
* 출력:

  * 최적화된 3D Gaussian 파라미터 (`.ply`, checkpoint)
* 핵심 기능:

  * Differentiable Gaussian Rendering
  * L1 + SSIM 기반 이미지 재구성 학습
  * Gaussian densification & pruning
  * 실시간 Viewer(GUI) 연동

---

## 2. 주요 의존 모듈 개요

```python
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim
```

| 모듈                | 역할                                            |
| ----------------- | --------------------------------------------- |
| `Scene`           | 데이터셋 + 카메라 관리                                 |
| `GaussianModel`   | 모든 3D Gaussian 파라미터 관리                        |
| `render()`        | Gaussian Splatting 기반 differentiable renderer |
| `network_gui`     | 실시간 뷰어와 통신                                    |
| `l1_loss`, `ssim` | 학습 손실 함수                                      |

---

## 3. `training()` 함수 전체 개요

```python
def training(dataset, opt, pipe, ...)
```

이 함수는 **전체 학습 파이프라인의 중심**이다.

### 전체 흐름 요약

```text
초기화
 └─ Scene & GaussianModel 생성
 └─ Optimizer / LR / Viewer 설정

for iteration:
 ├─ 카메라 선택
 ├─ 렌더링
 ├─ Loss 계산 (L1 + SSIM + Depth)
 ├─ Backpropagation
 ├─ Densification & Pruning
 ├─ Optimizer Step
 ├─ Logging / Save / Viewer
```

---

## 4. 초기 설정 단계

### 4.1 Optimizer 확인

```python
if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
    sys.exit(...)
```

* `SparseGaussianAdam`은 **CUDA rasterizer 확장**이 필요
* 설치되지 않은 경우 즉시 종료

---

### 4.2 출력 폴더 준비

```python
tb_writer = prepare_output_and_logger(dataset)
```

* `output/<uuid>/` 자동 생성
* `cfg_args` 파일에 실행 인자 기록


---

### 4.3 Gaussian / Scene 초기화

```python
gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
scene = Scene(dataset, gaussians)
gaussians.training_setup(opt)
```

* 초기 Gaussian:

  * 위치: SfM sparse points
  * 색상: SH(Spherical Harmonics)
  * opacity, scale 포함
* optimizer, learning rate 설정

---

### 4.4 Checkpoint 로드 (옵션)

```python
if checkpoint:
    gaussians.restore(model_params, opt)
```

* 중간 학습 재개 가능

---

## 5. 학습 루프 (Main Training Loop)

```python
for iteration in range(first_iter, opt.iterations + 1):
```

---

### 5.1 Viewer(GUI) 연동

```python
network_gui.try_connect()
render(custom_cam, ...)
```

* 외부 Viewer에서:

  * 임의 카메라로 렌더 요청
  * 학습 on/off 제어 가능
* **학습 중 실시간 시각화** 가능

---

### 5.2 Learning Rate & SH Degree 조절

```python
gaussians.update_learning_rate(iteration)

if iteration % 1000 == 0:
    gaussians.oneupSHdegree()
```

* SH degree를 점진적으로 증가
* 초기에는 저차 SH → 안정적 수렴
* 후반부에 고차 SH → 디테일 표현

---

### 5.3 랜덤 카메라 샘플링

```python
viewpoint_cam = viewpoint_stack.pop(rand_idx)
```

* 모든 train camera를 **shuffle 없이 랜덤 소모**
* 한 epoch ≈ 모든 카메라 1회 사용

---

## 6. 렌더링 단계

```python
render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
```

### 반환값

```python
image                  # 렌더된 RGB 이미지
viewspace_point_tensor # projected gaussian points
visibility_filter      # 화면에 보이는 Gaussian mask
radii                  # image-space Gaussian 크기
```

---

## 7. Loss 계산

### 7.1 RGB Reconstruction Loss

```python
Ll1 = l1_loss(image, gt_image)
ssim_value = ssim(image, gt_image)
loss = (1-λ)*Ll1 + λ*(1-SSIM)
```

* 기본 손실:

  * **L1 Loss**
  * **SSIM (구조 유사도)**
* `lambda_dssim`으로 비율 조절

---

### 7.2 Depth Regularization (옵션)

```python
Ll1depth = |invDepth - mono_invdepth|
loss += depth_weight * Ll1depth
```

* monocular depth prior 활용
* depth 신뢰도 있는 카메라만 사용
* 학습 후반으로 갈수록 weight 감소 (exponential)

---

## 8. Backpropagation

```python
loss.backward()
```

* Gaussian 위치, 크기, opacity, 색상 모두 미분 가능
* **NeRF와 달리 네트워크 없음**
* 모든 파라미터는 직접 최적화 대상

---

## 9. Densification & Pruning (핵심 로직)

```python
gaussians.densify_and_prune(...)
```

### 수행 조건

* `iteration < densify_until_iter`
* 일정 iteration 간격마다 수행

### 동작 개념

| 동작            | 설명                          |
| ------------- | --------------------------- |
| Densify       | gradient 큰 Gaussian → split |
| Prune         | 기여도 낮은 Gaussian 제거          |
| Reset Opacity | 투명도 재조정으로 안정화               |

➡ **Sparse → Dense Gaussian Field로 진화**

---

## 10. Optimizer Step

### Exposure Optimizer

```python
gaussians.exposure_optimizer.step()
```

* 이미지 노출 차이 보정

### Gaussian Optimizer

```python
gaussians.optimizer.step()
```

* sparse_adam 사용 시:

  * 보이는 Gaussian만 업데이트
  * 성능 대폭 향상

---

# GS_test

## 0. 환경
- OS : Ubuntu 22.04
- GPU : RTX 3070Ti
- CUDA version : 11.8
- Python version : 3.9.*


## 1. Data
- 샘플데이터로 있는 train, truck, playroom등에 대해 먼저 학습을 진행했고 적절하게 결과가 나와서 직접 얻은 데이터(사진, 동영상)에 대해 돌려보고자 했다. 
-  <video controls src="./image/IMG_2293.mov" title="Title"></video>
- 해당영상은 iphone 12의 후방카메라로 찍은 영상이다(480 × 854).
- 이 영상을 ffmpeg application을 이용해 아래와 같은 명령어를 통해 5fps로 설정해 이미지로 변환해주었다. 
```
$ffmpeg -i IMG_2293.mov -vf fps=5 -q:v 1  images/frame_%04d.jpg
```
- 135장의 이미지가 생성되었다.

## 2. Convert.py

- 1의 과정에서 얻은 이미지를 GS를 돌리기 위해서는 적절한 형식에 맞춰줘야 한다. 이는 잘 구현된 ```convert.py```을 이용하면 된다. 
- 구동 결과 이미지의 갯수에 따라 다르지만 위 동영상 기준으로 2분정도의 시간이 소요됐고 결과는 아래와 같이 나오게 된다. 
```bash
|---images          # 왜곡이 제거된(Undistorted) 이미지 파일들
|---sparse          # SfM(Structure-from-Motion) 결과 데이터
|---input           # 원본 이미지 파일
    |---0
        |---cameras.bin    # 카메라 파라미터 정보
        |---images.bin     # 이미지 포즈 및 관련 데이터
        |---points3D.bin   # 초기 3D 포인트 클라우드 데이터

```
## 3. train.py
- ```python train.py -s data/test_01 -m output/test_01``` 과 같은 명령어를 통해 학습을 진행하게 된다. 

```
python train.py -s data/test_01 -m output/test_01
Optimizing output/test_01
Output folder: output/test_01 [29/12 06:28:10]
Tensorboard not available: not logging progress [29/12 06:28:10]
Reading camera 264/264 [29/12 06:28:10]
Converting point3d.bin to .ply, will happen only the first time you open the scene. [29/12 06:28:10]
Loading Training Cameras [29/12 06:28:10]
Loading Test Cameras [29/12 06:28:17]
Number of points at initialisation :  6480 [29/12 06:28:17]
Training progress:  23%|▏| 7000/30000 [02:07<08:39, 44.26it/s, Loss=0.0853250, D
[ITER 7000] Evaluating train: L1 0.04595177620649338 PSNR 21.01699447631836 [29/12 06:30:25]

[ITER 7000] Saving Gaussians [29/12 06:30:25]
Training progress: 100%|█| 30000/30000 [10:27<00:00, 47.83it/s, Loss=0.0475140, 

[ITER 30000] Evaluating train: L1 0.027780645340681077 PSNR 25.32618293762207 [29/12 06:38:45]

[ITER 30000] Saving Gaussians [29/12 06:38:45]

Training complete. [29/12 06:38:45]

```
- 학습 로그는 위와 같다. 

## 4. Rendering 결과

- SIBR_viewers를 통해 학습된 결과물에 대해 볼 수 있다. 
<video controls src="./image/Screencast.mp4" title="Title"></video>
- 휴대폰카메라로 찍어 렌더링한 결과 또한 같은 비율임을 알 수 있다.


## 5. 고찰

- 동영상을 통해 이미지를 얻어 GS를 진행했는데 이미지를 직접 찍어 진행할 때 성능에 어떤 차이가 있는지 확인 해 볼 필요가 있다고 생각한다.
- 렌더링한 영상에 대해 eval를 진행하지 않았는데 이 부분에 대해 진행 할 필요가 있다고 생각한다.
- 똑같은 동영상에 대해 이미지로 변환할 때 fps에 따라 성능 변화 또한 확인할 필요성이 있다고 생각한다. 
- 결과에 대해 SIBR_viewers를 통해 확인 할 수 있었는데 해당 GUI에 어떤 기능이 있는지 제대로 숙지해야 한다.
