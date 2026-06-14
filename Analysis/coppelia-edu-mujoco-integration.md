# CoppeliaSim Edu → MuJoCo/RoboCasa 통합

**날짜:** 2026-06-13  
**소스:** `C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\models` (223개 `.ttm`)

## 판단 요약

| 항목 | 결론 |
|------|------|
| MuJoCo 직접 로드 | **불가** (`.ttm` = VREP/CoppeliaSim 바이너리) |
| 간접 사용 | **가능** — CoppeliaSim → OBJ/URDF export → RoboCasa MJCF |
| RoboCasa Kitchen | **파일럿 완료** (4 objects + 1 fixture) |
| RoboCasa 로봇 | **MuJoCo URDF compile OK** (Dobot Magician, DAE→OBJ) |

## 구현 산출물

| 파일 | 역할 |
|------|------|
| [`coppelia_export_models.lua`](../robocasa/scripts/asset_scripts/coppelia_export_models.lua) | mesh + `urdf:` batch export |
| [`run_coppelia_export.ps1`](../robocasa/scripts/asset_scripts/run_coppelia_export.ps1) | headless export 실행 |
| [`import_coppelia_mesh.py`](../robocasa/scripts/asset_scripts/import_coppelia_mesh.py) | OBJ → MJCF (coacd) |
| [`import_coppelia_batch.py`](../robocasa/scripts/asset_scripts/import_coppelia_batch.py) | 객체 + fixture 일괄 import |
| [`import_coppelia_robot_urdf.py`](../robocasa/scripts/asset_scripts/import_coppelia_robot_urdf.py) | DAE→OBJ + URDF MuJoCo compile |
| [`convert_coppelia_urdf_meshes.py`](../robocasa/scripts/asset_scripts/convert_coppelia_urdf_meshes.py) | URDF mesh DAE→OBJ 변환 유틸 |
| [`verify_coppelia_edu.py`](../robocasa/scripts/asset_scripts/verify_coppelia_edu.py) | compile / sampling / env 검증 |

## 사용법

```powershell
# 1) Export (CoppeliaSim Edu)
robocasa\scripts\asset_scripts\run_coppelia_export.ps1

# 2) Import
.\.venv\Scripts\Activate.ps1
pip install coacd trimesh pycollada
python robocasa/scripts/asset_scripts/import_coppelia_batch.py

# 로봇만 (DAE→OBJ + MuJoCo compile)
python robocasa/scripts/asset_scripts/import_coppelia_robot_urdf.py

# MuJoCo viewer로 Dobot 확인
python -m robocasa.demos.demo_mujoco_physics --model dobot_magician

# 3) Verify
python robocasa/scripts/asset_scripts/verify_coppelia_edu.py
```

설정: [`exports/coppelia_edu/export_config.example.txt`](../exports/coppelia_edu/export_config.example.txt)  
`urdf:` 접두사로 로봇 URDF export (예: `urdf:robots/non-mobile/Dobot Magician.ttm`).

## 레지스트리 (`coppelia_edu`)

| CoppeliaSim | RoboCasa category | asset name |
|-------------|-------------------|------------|
| household/cup.ttm | cup | coppelia_cup |
| household/bowl.ttm | bowl | coppelia_bowl |
| household/largeBasket.ttm | basket | coppelia_basket |
| kitchenware/frying_pan_01.ttm | pan | coppelia_frying_pan |
| furniture/tables/diningTable.ttm | fixture | coppelia_dining_table |

```python
from robocasa.utils.env_utils import create_env

env = create_env(
    "PickPlaceCounterToSink",
    split="pretrain",
    obj_registries=("coppelia_edu",),
)
```

## 파일럿 결과

| 모델 | Export | MJCF | 비고 |
|------|--------|------|------|
| cup | OK | OK | ~9×9×12 cm |
| bowl | OK | OK | ~26×26×12 cm |
| largeBasket | OK | OK | |
| frying_pan_01 | OK | OK | ~49 cm 지름 |
| diningTable | OK | OK | fixture (`fixtures/coppelia_edu/`) |
| Dobot Magician | URDF OK | **MuJoCo OK** | 56× DAE→OBJ, `nbody=12`, `njnt=11` |

## CoppeliaSim import 주의

1. **단위:** Edu 모델은 **미터** — `prescale=False`, `rot=none`
2. **Windows collision:** `TestVHACD` 대신 **coacd**
3. **출력 경로:** `furniture/tables/diningTable.ttm` → `exports/.../furniture/tables/diningTable/`
4. **URDF:** MuJoCo는 `.dae` 미지원 → `convert_coppelia_urdf_meshes.py`로 **DAE→OBJ** 변환 (`pycollada` 필요). OBJ는 URDF와 **같은 폴더**에 두어야 함 (MuJoCo가 mesh filename basename만 사용)
5. **Edu 라이선스:** `exports/` 로컬 only (gitignore)

## 검증 체크리스트

- [x] CoppeliaSim OBJ export (6/6 mesh + URDF)
- [x] `mujoco.MjModel.from_xml_path` (Kitchen 객체 + fixture)
- [x] `MJCFObject` 로드
- [x] `sample_kitchen_object(..., obj_registries=("coppelia_edu",))`
- [x] Kitchen env reset (layout seed에 따라 실패 가능 — verify가 seed 0–4 시도)
- [x] URDF export (Dobot Magician)
- [x] URDF MuJoCo compile (DAE→OBJ, Dobot Magician)

## graspPoint → MJCF grasp_site (MovePan)

CoppeliaSim `frying_pan_01.ttm`의 `graspPoint` dummy를 export/import 파이프라인에서 MJCF `<site name="grasp_site">`로 보존합니다.

```
frying_pan_01.ttm (graspPoint)
  → coppelia_export_models.lua → grasp_point.json (world pos)
  → import_coppelia_mesh.py → model.xml grasp_site (bbox center offset)
  → MovePan preview: pan_grasp_site 기준 IK/attach
```

| 단계 | 파일 | 비고 |
|------|------|------|
| Export | `coppelia_export_models.lua` | alias index `0`, world-frame `pos` |
| Sidecar | `exports/.../frying_pan_01/grasp_point.json` | mesh center와 동일 `-center` 적용 |
| MJCF | `coppelia_frying_pan/model.xml` | sim에서 `pan_grasp_site` |
| Preview | `move_pan_live.py` | approach/lift는 grasp site, place는 body center |

**주의:** sidecar `pos`는 export OBJ와 동일한 **world frame**이어야 합니다 (model-local이 아님).

## 다음 단계 (선택)

- robosuite 로봇 등록 (Dobot Magician URDF → Kitchen env 연동)
- furniture Kitchen layout 연동
- robosuite 미지원 로봇 (Niryo One 등) — CoppeliaSim GUI에서 base shape 추가 후 URDF export
