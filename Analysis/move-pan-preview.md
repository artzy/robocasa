# MovePan Preview — Home Pose

**날짜:** 2026-06-14

MovePan live preview (`demo_tasks` → MovePan)는 JSON 프리셋으로 정의된 **Home pose**에서 시작하고, pick-place 작업 후 **Home으로 복귀**해 종료합니다.

## Home 프리셋

| 파일 | 설명 |
|------|------|
| [`robocasa/demos/move_pan_home_presets/default_layout15_seed0.json`](../robocasa/demos/move_pan_home_presets/default_layout15_seed0.json) | layout=15, style=34, seed=0, counter→stove 기본 Home |

스키마:

- `base_pos`, `base_yaw` — mobile base world pose
- `eef_pos`, `eef_quat_wxyz` — gripper site pose
- `gripper` — 0=open

## 캘리브레이션 (layout/seed 변경 시)

```powershell
cd d:\Github\artzy_github\Robot_Simulation\robocasa
.\.venv\Scripts\Activate.ps1
python robocasa/scripts/asset_scripts/capture_move_pan_home.py `
  --layout 15 --style 34 --seed 0 `
  --source-fixture counter --target-fixture stove `
  --output robocasa/demos/move_pan_home_presets/default_layout15_seed0.json
```

## 실행

```powershell
python -m robocasa.demos.demo_tasks --task MovePan
python -m robocasa.demos.demo_tasks --task MovePan --home-preset path/to/home.json
python robocasa/scripts/asset_scripts/verify_move_pan.py
```

## Timeline (Home 복귀)

```
Hold@Home → MoveToPick → Grasp → Lift → MoveToPlace → Place → Retreat
  → ReturnBaseToHome → ReturnArmToHome → Hold@Home
```

- **ReturnBaseToHome** (30 frames): base만 home으로, arm은 retreat 높이 유지
- **ReturnArmToHome** (20 frames): base=home 고정, EEF home으로
- **Hold@Home** (30 frames): 종료 대기

총 프레임: ~230
