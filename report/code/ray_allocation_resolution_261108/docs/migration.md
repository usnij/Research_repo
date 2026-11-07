# 공개용 코드 이식

## 별도 저장소가 필요한 이유

현재 연구 폴더는 실험 기록 보존에 적합하지만 실행 코드의 기준점으로 삼기는 어렵다.
실행기에서 과거 실험 helper를 거쳐 다른 runtime을 가져오며, import가 cwd와 sys.path도 바꾼다.
`frozen`이라는 폴더 이름만 보고 실제로 그 코드가 쓰인다고 판단할 수 없다.

로컬 CPU import 해석에서 확인한 흐름은 다음과 같다. CUDA는 초기화하지 않았다.

```text
coordinate_ssim_refine/train_refined
  → area_full/train
  → sv_full_forward_sparse_backward/cached_v2/full_training
  → experiment → mv_mcmc_decision/support → handoff_factorial/common
  → hybrid_AB_compare/common/threedgrut
  → single_reset_omit/runtime/threedgrt_tracer
  → setup에서 area_tracer512 native extension으로 교체
```

로컬 파일 해석과 원격 인스턴스에 배포된 파일 내용의 동일성은 별도 해시 확인 사항이다.
현재 대조군은 같은 실행 체인을 공유하므로, 이 발견 자체가 기존 결과가 잘못됐다는 의미는 아니다.

## 이번에 분리한 것

- 기본 좌표 SSIM Python/CUDA는 그대로 복사하고 SHA-256으로 고정했다.
- 표집과 view 선택, loss 조합, 학습 업데이트를 별도 모듈로 분리했다.
- L1 계수별로 trainer를 복제하지 않고 설정 파일로 지정하도록 했다.
- 입력 최적화 후보와 손실 최적화 후보를 기본 동작에서 분리했다.
- 호환 실행기는 기존 renderer setup·평가·checkpoint writer를 명시적으로 주입받는다.
- 실제로 로드된 Python·native extension의 경로와 해시, precision flag를 남기도록 했다.
- 데이터·checkpoint·SSH 키·인스턴스 주소·바이너리는 저장소에 넣지 않았다.

## 남은 공개 준비

1. 새 모듈과 기존 실행의 GPU parity를 통과시킨다. 현재 CPU 검사만 완료했다.
2. 실제 학습에 쓰인 threedgrut 버전과 upstream 수정 목록을 하나의 backend로 고정한다.
3. 그 backend에 renderer/MCMC/CUDA build entry를 제공하고 호환 bootstrap 의존성을 제거한다.
4. 데이터 경로·도구체인 경로를 설정으로 받으며 import의 chdir/sys.path 부작용을 없앤다.
5. 개수 추정식의 최종 버전을 선택하고 독립 모듈로 옮긴다. 과거 F2/add/delete 실험을 섞지 않는다.
6. Counter 고정 cap 대조 실험과 자동 cap 방식의 결과를 분리해 명시한다.
7. dataset별 adapter와 evaluation protocol, 저장/재개 시 Adam·RNG·스케줄 상태를 검증한다.
8. upstream notice·라이선스, 재현 명령, 필요한 평가 가중치 취득 방법을 정리한 뒤 공개한다.

## 앞으로의 원칙

- 기본 trainer는 하나다. 씬·가중치·seed 변경은 config로 표현한다.
- 변경마다 기존 실험의 코드 해시와 결과 경로를 연결한다.
- 실행 폴더는 새로 만들고, 이전 실행을 덮어쓰지 않는다.
- ‘수식 변경’, ‘backend 변경’, ‘같은 계산의 최적화’를 한 비교 실험에 섞지 않는다.
- 논문 결과는 이식 전 코드의 것인지, 이식 검증을 통과한 코드의 것인지 구분한다.

로컬 Git 저장소를 만들었으며 remote는 설정하지 않았다. 기존 연구 저장소/파일의 이동·삭제,
GitHub 업로드, 공개 라이선스 결정은 수행하지 않았다.
