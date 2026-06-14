# Live Preview Home 프레임워크

`demo_tasks`의 live preview 태스크(MovePan 등)가 공통으로 사용하는 Home pose·복귀 타임라인 모듈입니다.

## 구조

```
robocasa/demos/live_preview/
  home_pose.py      — RobotHomePose, preset load/save/resolve
  robot_session.py  — base/EEF pose, IK, apply/capture home
  return_home.py    — RETURN_BASE / RETURN_ARM / HOLD_END 타임라인
  timeline_utils.py — smoothstep, lerp
  fixture_transition.py — fixture joint transitions (fridge door close)
  pre_return/         — task-specific hooks before return-home
  registry.py         — LIVE_DEMO_REGISTRY / HOME_DEMO_REGISTRY

robocasa/demos/home_presets/<TaskName>/
  default_layout15_seed0.json
```

레거시 preset 경로(`move_pan_home_presets/`)도 `resolve_home_preset_path`가 자동 fallback 합니다.

## Home preset JSON

| 필드 | 설명 |
|------|------|
| `task_name` | 태스크 이름 (예: `MovePan`) |
| `layout`, `style`, `seed` | kitchen 메타데이터 (불일치 시 경고) |
| `source_fixture`, `target_fixture` | MovePan 전용 메타 |
| `base_pos`, `base_yaw` | 모바일 베이스 |
| `eef_pos`, `eef_quat_wxyz` | EEF 목표 |
| `gripper` | 그리퍼 (0=open) |

## 타임라인 (복귀 구간)

공통 상수 (`return_home.py`):

- `PHASE_RETURN_BASE = 30` — place 위치에서 home base로 이동
- `PHASE_RETURN_ARM = 20` — retreat EEF → home EEF
- `PHASE_HOLD_END = 30` — home에서 정지

MovePan 전체 preview는 ~250 frames (task 동작 + 위 80 frames).

## 장애물 회피 (base 경로)

Home 이동 구간(출발·복귀)은 직선 보간 대신 `base_path_planning.py`로 base XY 경로를 계획합니다.

- **충돌 검사**: `detect_robot_collision` + 앵커 state 고정 후 base pose만 변경
- **경로 탐색 순서**: 직선 샘플 → detour(수직 우회) → A* 그리드
- **검증**: 계획된 waypoint를 `PHASE_RETURN_BASE` 프레임 수로 샘플링해 충돌 재확인
- **MovePan**: 작업 타임라인 재생 후 `end_anchor`에서 home까지 `plan_base_waypoints` → `append_return_home_frames(..., base_waypoints=...)`
- **Human demo** (DeliverStraw 등): `robot_bridge.build_robot_transition_states(..., avoid_obstacles=True)`

콘솔에 `Planned detour/A* base path for Home transition`이 출력되면 회피 경로가 적용된 것입니다.  
`falling back to straight line` 경고는 회피 경로를 찾지 못한 경우입니다.

## HotDogSetup: Home 복귀 전 냉장고 문 닫기

Human demo 종료 후 Home 복귀 전 `pre_return/hot_dog_setup.py` hook:

1. 냉장고 문이 열려 있으면 **Goto fridge** (`PHASE_GOTO_FRIDGE_BASE=30`, 장애물 회피)
2. **Close door** (`PHASE_CLOSE_FRIDGE=20`, door joint 보간)
3. **Hold** (`PHASE_HOLD_AFTER_CLOSE=10`)
4. **Return home** — `return_anchor` = 문 닫힌 scene 기준

Registry: `HotDogSetup.pre_return_fn` → `append_hot_dog_pre_return_states`

## CLI

```powershell
# demo_tasks (registry가 preset 자동 해석)
python -m robocasa.demos.demo_tasks --task MovePan

# preset 캡처 (일반화)
.\.venv\Scripts\python robocasa/scripts/asset_scripts/capture_robot_home.py --task MovePan

# MovePan 래퍼 (동일)
.\.venv\Scripts\python robocasa/scripts/asset_scripts/capture_move_pan_home.py

# 검증
.\.venv\Scripts\python robocasa/scripts/asset_scripts/verify_move_pan.py
.\.venv\Scripts\python robocasa/scripts/asset_scripts/verify_hot_dog_setup_home.py
```

## 새 live preview 태스크 추가

1. `LIVE_DEMO_REGISTRY`에 `LiveDemoSpec` 등록 (`module`, `play_fn_name`, `default_home_preset`)
2. `DEFAULT_HOME_PRESETS`에 preset 상대 경로 추가
3. `home_presets/<TaskName>/`에 JSON 배치 (또는 `capture_robot_home.py --task ...`)
4. 태스크 타임라인 끝에 `append_return_home_frames(...)` 호출
5. `play_*_live`에서 `resolve_home_pose(env, task_name, ...)` + `apply_home_pose` 사용

Human demo 재생 태스크(DeliverStraw, HotDogSetup)는 `HOME_DEMO_REGISTRY` + `play_human_demo_with_home`를 사용합니다.

## 재생 성능 및 렌더러

Home-wrapped human demo(HotDogSetup 등)는 프레임 수·state 복원·렌더 경로가 겹치면 끊김(stutter)이 발생합니다. 아래 모듈로 완화합니다.

### 통합 재생 루프 (`playback_loop.py`)

- `fast_reset_state`: `set_state_from_flattened` + `forward`만 수행 (model reload 없음)
- `play_state_sequence`: **단일 pacing** (`max_fps=60`, adaptive skip: 처리 시간이 frame budget 초과 시 sleep 생략)
- Home dwell: `--no-dwell` / `--dwell-sec` (기본 1.5s)

### 렌더러 선택

| `--renderer` | 특징 |
|--------------|------|
| **mjviewer** (기본) | MuJoCo native GLFW viewer, GPU 직접 출력, 가장 부드러움 |
| mujoco | OpenCV `imshow` (OpenCV GUI 필요) |
| pygame | 호환성 최고, offscreen GPU render + CPU readback + blit (느림) |

Windows HOME demo 기본값은 **mjviewer**. pygame 사용 시 기본 해상도 **1280×720** (`--viewer-width/height`).  
pygame `update(..., pace=False)` — outer loop만 FPS 제한 (이중 `clock.tick` 제거).

### Preview 프레임 수 (HotDogSetup)

Registry (`HomeDemoSpec`):

- `demo_stride=2` — human demo 프레임 2배 subsample (preview 전용)
- `demo_tail_extend=20` — demo 종료 후 hold 프레임

검증 스크립트(`verify_hot_dog_setup_home.py`)는 `demo_stride=1`, registry tail로 **full fidelity** 유지.

### Wrap 빌드 캐시

경로: `robocasa/demos/.cache/home_wrap/{task}_ep{idx}_{preset_hash}.npz`

- 첫 실행: A* 경로 계획 등으로 30~60s+ 소요 가능
- 캐시 hit: 수 초 내 재생 시작
- `--rebuild-wrap` 으로 무효화 (preset/pre_return 코드 변경 시)

### CLI 예시

```powershell
python -m robocasa.demos.demo_tasks --task HotDogSetup --renderer mjviewer --playback-fps 60
python -m robocasa.demos.demo_tasks --task HotDogSetup --demo-stride 1 --rebuild-wrap
python -m robocasa.demos.demo_tasks --task HotDogSetup --renderer pygame --viewer-width 1280 --viewer-height 720
python -m robocasa.scripts.asset_scripts.benchmark_home_playback --no-render
python -m robocasa.scripts.asset_scripts.benchmark_home_playback --demo-stride 2 --rebuild-wrap
```

### 벤치마크

`benchmark_home_playback.py`: wrap 빌드 시간, `fast_reset` 평균 ms, 이론적 재생 시간, (선택) mjviewer 30프레임 샘플.
