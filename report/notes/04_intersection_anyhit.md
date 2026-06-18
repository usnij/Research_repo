# __intersection__ellipsoid / __anyhit__ah — shaders.slang

---

## 핵심 설명

### 연결된 파일

| 방향 | 파일 | 설명 |
|------|------|------|
| 호출됨 (optixTrace 발생 시) | [03_rg_float.md](03_rg_float.md) | rg_float()의 optixTrace가 이 두 셰이더를 트리거 |
| hit 결과 소비 | [03_rg_float.md](03_rg_float.md) | payload에서 ctrl_pt 읽어 update() 호출 |
| 전역 배열 출처 | [Forward_cpp.md](Forward_cpp.md) | Params struct에서 포인터 연결 |
| ControlPoint 자료구조 | [01_SplineState.md](01_SplineState.md) | get_ctrl_pt()가 반환하는 타입 (spline-machine.slang) |

### 전역 배열 출처 (Params struct → Forward.cpp)

셰이더 안의 `means[]`, `scales[]` 등은 모두 `SLANG_globalParams`를 통해 전달된다. → [Forward_cpp.md](Forward_cpp.md)

| 셰이더 전역 | Params 필드 | 설정 위치 |
|------------|------------|---------|
| `means[]` | `params.means.data` | `Forward::Forward()` 생성자 (ctx.prims에서 연결) |
| `scales[]` | `params.scales.data` | 동일 |
| `quats[]` | `params.quats.data` | 동일 |
| `heights[]` (= densities) | `params.densities.data` | 동일 |
| `features[]` | `params.features.data` | 동일 |

### __intersection__ellipsoid() — 정밀 교차 계산

BVH가 AABB 후보를 찾을 때마다 자동으로 호출된다. AABB는 근사 바운딩이므로, 실제 ellipsoid 수식으로 정밀 교차를 계산한다.

**tri 인코딩**:
```
tri = 2 × prim_id + hitkind
hitkind = 1 → entry (t_enter)
hitkind = 0 → exit  (t_exit)
```

- `optixReportHit(t_enter, 1u, ...)` → entry 이벤트 보고
- `optixReportHit(t_exit, 0u, ...)`  → exit 이벤트 보고

ray origin이 ellipsoid 내부에 있는 경우 (`cur_t >= t_enter`):
- entry는 이미 지나쳤으므로 보고하지 않음
- exit만 보고 (`optixReportHit(t_exit, 0u, ...)`)

### __anyhit__ah() — payload 삽입 정렬

intersection이 hit를 보고할 때마다 호출. payload 16슬롯을 t 기준 오름차순으로 유지한다.

```
[entry_A] [exit_C] [entry_B] [exit_A] ... [빈슬롯 1e10]
   t=0.3    t=0.5    t=0.7    t=1.2   ...
```

한 번의 intersection 호출에서 entry와 exit **두 이벤트 모두** 삽입 정렬한다 (`for n in [0,1]`).

슬롯이 꽉 찼을 때: 새 hit의 t가 가장 큰 슬롯보다 크면 `IgnoreHit()`으로 버림. 작으면 삽입하고 가장 큰 슬롯을 밀어냄.

**`IgnoreHit()`의 의미**: 이 hit는 payload에 들어가지 않지만, BVH 탐색은 계속됨. 16개보다 많은 hit가 있으면 t가 작은 16개만 수집한다.

---

## 전체 코드

```slang
ControlPoint get_ctrl_pt(uint tri, float t) {
    ControlPoint ctrl_pt;
    let prim_ind = tri / 2;
    let hitkind  = tri % 2;
    let height   = heights[prim_ind];
    let dirac_height = height * ((hitkind == 1) ? 1 : -1);

    Features feat;
    SHFeatures sh_feats = {prim_ind, sh_degree, features};
    feat.f0 = get_sh(sh_feats, 0);
    float3 rayd = {0, 0, 1};
    let color = eval_sh_col0(rayd, feat);

    ctrl_pt.t        = t;
    ctrl_pt.dirac.x  = dirac_height;
    ctrl_pt.dirac.y  = dirac_height * color.x;
    ctrl_pt.dirac.z  = dirac_height * color.y;
    ctrl_pt.dirac.w  = dirac_height * color.z;
    return ctrl_pt;
}

[shader("intersection")]
void ellipsoid() {
    uint prim_ind = PrimitiveIndex();

    float3 rayd = WorldRayDirection();
    float3 rayo = WorldRayOrigin();

    let mean  = means[prim_ind];
    let scale = scales[prim_ind];
    let quat  = quats[prim_ind];
    float2 minmaxt = ray_intersect_ellipsoid(rayo - mean, rayd, scale, quat);

    float cur_t = RayTMin();
    if ((minmaxt.y < cur_t)) {
        return;
    } else {
        bool use_min = cur_t < minmaxt.x;
        if (use_min) {
            optixReportHit(minmaxt.x, 1u, asuint(minmaxt.y));  // entry: hitkind=1
        } else if (cur_t < minmaxt.y) {
            optixReportHit(minmaxt.y, 0u, asuint(minmaxt.x));  // exit:  hitkind=0
        }
    }
}

[shader("anyhit")]
void ah()
{
    float t        = RayTCurrent();
    float other_t  = asfloat(optixGetAttribute_0());
    uint  ind      = PrimitiveIndex();
    uint  hitkind  = optixGetHitKind();

    if (hitkind == 0) {
        float temp_t = other_t;
        other_t = t;
        t = temp_t;
    }

    float cur_t = RayTMin();

    float h_t;
    uint  h_i;
    float test_t;
    uint  test_i;

    // entry(h_i=2*ind+1)와 exit(h_i=2*ind) 각각 삽입 정렬
    for (int n=0; n<2; n++) {
        if (n == 0) {
            h_t = t;
            h_i = 2 * ind + 1;   // entry tri
        } else {
            h_t = other_t;
            h_i = 2 * ind;       // exit tri
        }
        if (h_t > cur_t) {
            [ForceUnroll]
            for (int i=0; i<BUFFER_SIZE; i++) {
                test_t = asfloat(get_payload(i*2));
                if (h_t < test_t) {
                    set_payload(i*2,   asuint(h_t));
                    test_i = get_payload(i*2+1);
                    set_payload(i*2+1, h_i);
                    h_i = test_i;
                    h_t = test_t;
                }
            }
        }
    }

    if (t < asfloat(get_payload(2*(BUFFER_SIZE-1)))) {
        IgnoreHit();
    }
}

[shader("miss")]
void ms()
{
}
```
