# MuJoCo 공식 MJCF → RoboCasa 통합

**날짜:** 2026-06-13  
**출처:** [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) `model/` 디렉터리 (Apache 2.0)

## 요약

| 항목 | 상태 |
|------|------|
| upstream clone | `D:\Github\artzy_github\Robot_Simulation\mujoco-models` |
| 변환 스크립트 | [`import_mujoco_mjcf.py`](../robocasa/scripts/asset_scripts/import_mujoco_mjcf.py) |
| 일괄 import | [`import_mujoco_batch.py`](../robocasa/scripts/asset_scripts/import_mujoco_batch.py) |
| 물리 sandbox | [`demo_mujoco_physics.py`](../robocasa/demos/demo_mujoco_physics.py) |
| Kitchen 객체 | `mug`, `mujoco_playing_card` (`mujoco_official` registry) |

## upstream `model/` 인벤토리

| 모델 | nbody | njnt | ngeom | plugin | RoboCasa |
|------|------:|-----:|------:|--------|----------|
| **mug** | 3 | 1 | 36 | none | **Kitchen 객체** (`mujoco_mug`) |
| **cube** | 28 | 26 | 28 | none | **sandbox only** (`demo_mujoco_physics`) |
| **cards** (full deck) | 53 | 52 | 105 | none | sandbox only (XML `cameraid` 이슈 가능) |
| **cards** (single) | 1 | 0 | 3 | none | **Kitchen** (`mujoco_card_2_clubs`) |
| humanoid | — | — | — | — | 보류 (robosuite GR1 별도) |
| flex / adhesion / balloons / plugin | — | — | — | **required** | Phase 3 보류 |

## upstream 확보

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco.git D:\Github\artzy_github\Robot_Simulation\mujoco-models
cd D:\Github\artzy_github\Robot_Simulation\mujoco-models
git sparse-checkout set model/mug model/cube model/cards model/cube/assets model/cards/assets
```

## 변환 / import

```powershell
cd D:\Github\artzy_github\Robot_Simulation\robocasa
.\.venv\Scripts\Activate.ps1
pip install trimesh

# 일괄 (mug + card)
python robocasa/scripts/asset_scripts/import_mujoco_batch.py

# 개별
python robocasa/scripts/asset_scripts/import_mujoco_mjcf.py `
  --src ..\mujoco-models\model\mug\mug.xml --name mujoco_mug
python robocasa/scripts/asset_scripts/import_mujoco_mjcf.py `
  --src ..\mujoco-models\model\cards\cards.xml --name mujoco_card_2_clubs --body-index 0
```

출력 경로: [`robocasa/models/assets/objects/mujoco_official/`](../robocasa/models/assets/objects/mujoco_official/)

## RoboCasa에서 사용

```python
from robocasa.utils.env_utils import create_env

env = create_env(
    "PickPlaceCounterToCabinet",
    split="pretrain",
    obj_registries=("mujoco_official",),
)
env.reset()
```

레지스트리: `kitchen_objects.py` → `mug.mujoco_official`, `mujoco_playing_card.mujoco_official`

## 물리 sandbox (Kitchen 외)

Rubik's cube 등 articulation 전체 씬:

```powershell
python -m robocasa.demos.demo_mujoco_physics --model cube
python -m robocasa.demos.demo_mujoco_physics --model mug
```

환경 변수 `MUJOCO_MODELS_ROOT`로 upstream 경로 지정 가능.

## 변환 파이프라인

1. `freejoint` body 추출 (`--body-index`로 다중 body 선택)
2. asset `assetdir` 지원, texture/mesh → `visual/`
3. nested `<default>` → inline flatten (robosuite 호환)
4. `reg_bbox` (mesh OBJ 또는 collision geom AABB)
5. geom `group` 정규화, tiny/zero `mass` 제거

## 검증 완료

- [x] `mujoco.MjModel.from_xml_path` (변환본)
- [x] `MJCFObject` 로드 (mug, card)
- [x] Kitchen env `obj_registries=("mujoco_official",)` reset
- [x] upstream cube compile (`njnt=26`)
- [ ] `demo_teleop --task PrepareCoffee` (수동 GUI 테스트)
- [ ] `browse_mjcf_model` (OpenCV headless 시 pygame/`mujoco.viewer` 대안)

## 제약

- human demo는 objaverse 기준 — MuJoCo 객체 교체 시 새 데모 필요
- cards 전체 덱 XML은 `visual/global@cameraid` 등으로 일부 MuJoCo 버전에서 compile 실패 가능 → 단일 카드 import 사용
- flex/adhesion 등 plugin 모델은 RoboCasa Kitchen merge **비권장**
