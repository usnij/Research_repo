# 3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting

## 1. Introduction


### 3DGS의 근본적 한계

- 3DGS의 빠른 속도는 rasterization 덕분이지만, 그로인해 심각한 한계가 있다. 
- 왜곡 카메라와 rolling shutter에 취약함.
  - 3DGS는 3D Gaussian을 2D로 투영할 때 비선형 카메라 투영 함수의 Jacobian(미분)을 이용해 근사한다. 
  - 이는 완벽한 pinhole 카메라에서도 근사 오차가 존재하며, 렌즈 왜곡이 강할수록 오차가 폭증한다. 
  - rolling shutter처럼 시간에 따라 카메라가 변하면 Jacobian 기반 모델이 아예 정의 불가능하다. 
  - 즉 EWA splatting은 비선형·시간의존 투영을 처리할 수 없다.
- Rasterization은 반사(reflection), 굴절(refraction), 그림자(shadow)와 같은 secondary ray 효과를 표현할 수 없다.


### 3DGUT의 핵심 아이디어

- Jacobian 대신 Unscented Transform 사용
  - Gaussian → 몇 개의 sigma points로 샘플링 → 각 점을 정확한 투영함수로 변환 → 다시 2D Gaussian으로 재조합
  - 즉 비선형 투영을 근사하는게 아니라 Gaussian 자체를 근사해서 정확하게 투영
- 3DGUT는 Gaussian의 반응을 3D에서 평가하고 ray-tracing과 같은 순서로 입자를 정렬한다. 이를 통해 빠른 렌더링과 물리효과 둘 다 가능해진다. 
- ![alt text](/Study/image/3DGUT_image1.png)

## 2. method 


### Unscented Transform

- 3D Gaussian 하나를 아래의 식처럼 7개의 점으로 근사한다.
```math
x_i =
\begin{cases}
\mu, & i = 0 \\
\mu + \sqrt{(3 + \lambda)\,\Sigma_{[i]}}, & i = 1, 2, 3 \\
\mu - \sqrt{(3 + \lambda)\,\Sigma_{[i-3]}}, & i = 4, 5, 6
\end{cases}
```
- μ : 입자의 위치, Σ : 공분산 행렬 


- 각 시그마 포인트에 대한 **가중치 W**는 다음과 같다. 
```math
w_i^{\mu} =
\begin{cases}
\dfrac{\lambda}{3 + \lambda}, & i = 0 \\
\dfrac{1}{2(3 + \lambda)}, & i = 1, \ldots, 6
\end{cases} \\

w_i^{\Sigma} =
\begin{cases}
\dfrac{\lambda}{3 + \lambda} + (1 - \alpha^2 + \beta), & i = 0 \\
\dfrac{1}{2(3 + \lambda)}, & i = 1, \ldots, 6
\end{cases}
```
  
  - 여기서 $\ λ = a^2(3 + k) - 3$이며 α
 는 평균 주위로 포인트들의 분포를 제어하는 **하이퍼파라미터**이고, k는 스케일링 파라미터이고, β는분포에 대한 prior를 통합하는 데 사용된다.


- 아래 식은 projection된 sigmapoints를 가중 평균내어 projection된 2D Gaussian의 중심(평균)을 계산하는 식이다. 
```math

\nu_\mu = \sum_{i=0}^{6} w_i^{\mu} \, v_{x_i} \\

```
- 의미 :
  - $\ x_i$ : 3D Gaussian에서 뽑은 sigma point
  - $\ v_{x_i}$ : 이 점을 실제 카메라 모델로 투영한 2D 위치
  - $\ w^μ_i$ : 각 sigma point의 평균 가중치

- 아래식은 projection된 Gaussian의 공분산을 복원하는 식이다.
```math
\nu_\Sigma = \sum_{i=0}^{6} w_i^{\Sigma} \, (v_{x_i} - \nu_\mu)(v_{x_i} - \nu_\mu)^{\top}

```
- 의미
  - $\ v_{x_i}-v_\mu$ : 각 sigma point가 평균에서 얼마나 벗어났는지
  - $\ w^Σ_i$ : 공분산 가중치 
- 위 과정들을 요약하면 3D Gaussian을 7개의 sigmapoint로 분해 -> 이 점들이 Gaussian을 제대로 대표할 수 있게 가중치 부여 -> 투영된 점들로 정확한 2D Gaussian 재구성이다

- 이는 3D_GUT의 핵심적인 부분이다.


## 3. Evaluating Particle Response
![alt text](/Study/image/3DGUT_image2.png)

- 기존 3DGS는 Gaussian의 response에 대한 평가를 2D 이미지 평면에서 평가한다. 이는 왜곡이나, rolling shutter 같은 환경에서 불안정하게 된다. 

- 3DGUT은 Gaussian에 대한 response평가를 3D 공간에서 ray와의 관계로 평가한다. 
- 수식적으로 보면 아래와 같다. 
  - Gaussian의 response가 ray위에서 최대가 되는 지점 $\ r_{max}$는 아래와 간다.
  
    $\ r_{\max} = \arg\max_r \, G(o + r d)$
  - 이를 미분하면 

    $\ r_{\max}
    = \frac{(\mu - o)^{\top} S^{-1} d}{d^{\top} S^{-1} d}$

  - 여기서 G는 Gaussian을 의미하고, camera ray $\ x(r)= o+r d$다
    - o : 카메라 중심, d : ray 방향, r : ray파라미터(깊이) 
  - 정리하면 Gaussian 중심 μ에서 ray 방향으로 “수직으로 가장 가까운 지점”을 구하게 되는 것이다. 