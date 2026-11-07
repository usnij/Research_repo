# 실사 13씬 벤치마크 실행

첫 차수는 최종 방법 13씬, seed 777이다. 기존 논문 체크리스트의 실사 범위를 따른다.
학습 코어는 독립 설치를 검증한 `b42b8a3`과 같다. 아래 도구는 입력 검사, 배포,
작업 순서, 백업, 결과 취합을 담당하며 학습 수식이나 CUDA 커널을 바꾸지 않는다.

## 고정 조건

- Sparse GRT + 입력 기반 개수 추정 + MCMC + 처음부터 MV4 + 좌표 구조손실.
- `0.8 L1 + 0.2 L_coord`, opacity·scale 정규화 각각 0.01, sigma 6, `boundary=all`.
- `fine_tracks_v1`: 전체 train RGB와 train SfM incidence를 사용한다. test RGB는 제외한다.
- SfM 초기 개수에서 입력 목표까지 index 600~5000의 45회 로그 성장. 이후 개수 고정.
  index 5000은 5001번째 완료 update다. 재배치율 기반 추가·삭제 보정은 이 설정에 없다.
- 처음부터 30,000 update. 7k·15k·30k에서 모든 test 뷰 평가. 최종 표는 30k 값이다.
- 체크포인트는 1k 간격 최신본, 7k·30k 보존본. 15k는 평가 기록을 남긴다.
- 씬별 기존 DS·Ray 예산을 유지한다. MV4의 총예산이며, 뷰마다 그 예산을 쓰지 않는다.

설정 원본은 [`real13_v1.json`](../benchmarks/real13_v1.json)이다. 기존 결과에 맞춰
계수나 예산을 다시 고르지 않는다. 개수 식의 9개 개발 씬과 나머지 데이터셋의 역할은
논문에서 구분한다.

## 인스턴스 수와 배정

**RTX 4090 24GB 총 8대**를 권한다. 현재 검증 인스턴스를 포함하면 추가 7대다.
한 GPU에는 학습 하나만 실행한다. 13대는 한 씬씩 동시에 처리할 수 있지만,
현재 과제에는 8대로 1~2개 씬씩 맡기는 구성이 적절하다.

기존 최종 방법의 30k 기록은 Counter 약 27.6분, Bonsai 약 24.0분, Bicycle 약 35.3분이었다.
새 서버 전체의 완료 시간은 아직 측정하지 않았다. 이 기록은 대수 산정의 참고이며,
이번 610스텝 검사 시간을 풀학습으로 외삽한 값이 아니다. 새 씬은 cap×Ray 수를 작업 배분의
대략적인 비용 지표로만 사용한다. 학습 종료 예상시간이나 실제 렌더 비용으로 해석하지 않는다.

| 인스턴스 | 순서 | DS | MV4 총 명목 Ray 예산 |
|---|---|---|---|
| 1, 현재 인스턴스 | Counter → Bonsai | 2 / 2 | 1/16 / 1/16 |
| 2 | Bicycle | 4 | 1/9 |
| 3 | Kitchen | 2 | 1/9 |
| 4 | Treehill | 4 | 1/16 |
| 5 | Stump → Truck | 4 / 1 | 1/16 / 1/16 |
| 6 | Flowers → Playroom | 4 / 1 | 1/16 / 1/16 |
| 7 | DrJohnson → Train | 1 / 1 | 1/16 / 1/9 |
| 8 | Garden → Room | 4 / 2 | 1/16 / 1/16 |

임대 조건은 1×RTX 4090 24GB, RAM 최소 32GB(64GB 권장), 디스크 최소 100GB
(150GB 권장), SSH 접속이 가능한 PyTorch 템플릿이다. `uv`·`mamba`가 제공되는 이미지를
사용한다. 드라이버는 바꾸지 않고 검증된 Python 3.11/CUDA 11.8 환경을 별도로 만든다.
기존 인스턴스의 검증 환경은 재사용할 수 있다.

## 입력 준비 결과

2026-11-02에 13씬의 카메라·이미지 크기·RGB 형식·mask 유무·동일 Ray 수·SfM·split·cap을
CPU에서 점검했다. 입력 파일은 모두 합쳐 약 3.02GiB이며, 맡은 씬만 전송한다.
아래 개수는 학습 결과가 아니라 고정된 입력 식의 출력이다.

| 씬 | Train / test | DS | Rays/update | 입력 목표 N |
|---|---:|---:|---:|---:|
| Counter | 210 / 30 | 2 | 101,400 | 1,229,851 |
| Room | 272 / 39 | 2 | 101,400 | 1,013,619 |
| Bonsai | 255 / 37 | 2 | 101,400 | 1,209,886 |
| Kitchen | 244 / 35 | 2 | 180,960 | 1,590,685 |
| Bicycle | 169 / 25 | 4 | 113,436 | 3,021,939 |
| Flowers | 151 / 22 | 4 | 65,312 | 2,577,733 |
| Garden | 161 / 24 | 4 | 68,460 | 2,044,672 |
| Stump | 109 / 16 | 4 | 64,896 | 2,824,473 |
| Treehill | 123 / 18 | 4 | 66,144 | 3,061,821 |
| Train | 263 / 38 | 1 | 59,696 | 1,538,857 |
| Truck | 219 / 32 | 1 | 33,948 | 1,475,247 |
| DrJohnson | 230 / 33 | 1 | 73,480 | 2,282,236 |
| Playroom | 196 / 29 | 1 | 65,728 | 1,378,867 |

원본 이미지의 경계 때문에 실제 Ray 비율은 명목값과 조금 다르며 실행 기록에 정확히 남긴다.
Synthetic·London/Berlin의 별도 입력 어댑터는 이 13씬 실행 범위에 포함하지 않는다.

## 자동 실행 순서

1. 로컬에서 씬별 파일 목록·SHA256·설정·cap·train/test 목록을 고정하고 데이터 묶음을 만든다.
2. 주소가 들어온 인스턴스에 코드부터 보내 환경 설치를 시작한다. 설치 중 해당 씬 데이터를 전송한다.
   검증 환경의 의존성 버전을 constraints로 전달해 새 서버마다 패키지 버전이 달라지지 않게 한다.
3. 설치된 패키지와 소스 해시, 의존성 버전, JPEG 디코더, GPU 유휴 상태를 검사한다.
4. CPU 검사와 구조손실 CUDA 검사를 실행하고, 평가용 공개 VGG 가중치를 미리 받는다.
5. 각 씬에서 610스텝을 처음부터 실행해 cap·Ray 수·초기 개수·설정이 고정 명세와 같은지 확인한다.
6. 새 출력 폴더에서 30k 학습과 세 차례 평가를 실행한다. NaN, 개수 오류, 디스크 부족이면 중단한다.
7. 결과 파일과 7k·30k 체크포인트의 SHA256 목록을 만든다. GPU 작업은 이때 백업 확인을 기다린다.
8. 로컬 수집기가 파일을 이어받아 SHA256을 검사하고, 지정한 Google Drive로 복사해 해시를 대조한다.
   확인 응답을 받은 인스턴스만 다음 씬을 시작한다. 백업 지연을 학습시간에 넣지 않는다.
9. 완료·평가·보관 확인이 끝난 실행만 JSON/Markdown 결과표에 들어간다. 일부 씬만 끝났을 때
   데이터셋 전체 평균으로 표시하지 않는다.

SSH 개인키와 Drive 인증정보는 로컬에만 둔다. 인스턴스로 복사하지 않는다.
인스턴스 파기는 자동화하지 않는다. 보관을 확인한 뒤 별도로 처리한다.

## 실행 명령

데이터 위치와 SSH 주소는 저장소 밖의 로컬 JSON으로 관리한다. 저장소 루트 기준이다.

```bash
# 1. GPU를 사용하지 않는 사전 준비. data_roots.json: {"counter":"/path/to/counter", ...}
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=src python tools/benchmark_prepare.py \
  --spec benchmarks/real13_v1.json --data-roots /path/to/data_roots.json \
  --output /path/to/prepared --nodes 8

# 2. 현재 소스와 준비된 명세를 별도 묶음으로 고정
python tools/benchmark_fleet.py freeze --code . \
  --prepared /path/to/prepared --output /path/to/frozen_release

# 3. 새 주소를 fleet.json에 추가한 뒤 해당 노드만 배포
python tools/benchmark_fleet.py deploy --fleet /path/to/fleet.json --nodes gpu02

# 4. 로컬에서 수집기를 계속 실행. 이 프로세스가 멈추면 원격 큐는 백업 대기 상태에서 멈춘다.
python tools/benchmark_fleet.py watch --fleet /path/to/fleet.json

# 상태 조회와 한 번만 결과 회수
python tools/benchmark_fleet.py status --fleet /path/to/fleet.json
python tools/benchmark_fleet.py collect --fleet /path/to/fleet.json
```

`fleet.json` 예시는 다음과 같다. 이미 배포한 노드의 씬 배정이나 코드는 바꾸지 않는다.
새 주소는 노드 목록에 추가할 수 있고 수집기가 다음 확인 때 읽는다.

```json
{
  "release": "/path/to/frozen_release",
  "collected": "/path/to/collected",
  "expected_jobs": [["counter", "structure"], ["bonsai", "structure"]],
  "drive": {"destination": "gdrive:benchmark/campaign-id", "root_id": "YOUR_FOLDER_ID"},
  "nodes": [{
    "name": "gpu01", "host": "HOST", "port": 22,
    "root": "/workspace/sparse-grt-real13",
    "scenes": ["counter", "bonsai"], "arms": ["structure"]
  }]
}
```

실제 13씬 파일은 `expected_jobs`에 13개를 모두 넣는다. `existing_activation`을 지정하면
해당 활성화 스크립트를 쓰되 버전·설치 소스 검사는 동일하게 적용한다.
`smoke_only: true`는 배포 검사에만 사용하며 30k 결과표에 넣지 않는다.

## 비교군과 재실행

첫 큐는 최종 방법만 실행한다. 같은 설정의 L1 대조군은 노드의 `arms`를 배포 전에
`["structure", "l1"]`로 지정해 같은 GPU에서 이어 실행할 수 있다. 이 경우 두 실행의
초기 상태·뷰/Ray manifest·성장 일정·첫 입력을 추가 대조한다. 본실험을 시작한 후에는
별도 배포 폴더를 만들어 추가한다.

원 GRT·3DGS·GUT는 각 공식 실행기를 별도로 고정해야 한다. 이 도구가 그 비교군을
제안 방법의 설정으로 대체하지 않는다. GPU 모델이 같아도 호스트별 속도가 다를 수 있으므로
속도비를 낼 대조군은 같은 인스턴스에 배정하고, 하드웨어·power limit 기록을 함께 남긴다.

중단된 실행을 새 실행처럼 덮어쓰지 않는다. 현재 일반 학습 CLI에는 임의 스텝 `--resume`이
없다. 저장 상태의 복원 검증은 완료했지만 자동 재개 기능과 구분한다. 실패한 실행은 보존하고
원인을 해결한 뒤 새 실행 폴더로 시작한다. 완료된 씬과 보관 확인을 받은 결과는 재실행하지 않는다.

CUDA OOM이나 성능 차이가 나도 Ray 예산·DS·cap·손실을 자동으로 낮추지 않는다.
해당 조건의 실패를 기록하고 점검한다. 로컬 수집기는 네트워크·Drive 오류를 기록한 뒤
재시도하며, 검증되지 않은 체크포인트에는 완료 응답을 보내지 않는다.
