<h1 align="center">RoboCasa</h1>
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/readme.webp" width="100%" />

**RoboCasa**는 일상적인 작업을 수행하는 범용 로봇을 학습하기 위한 대규모 시뮬레이션 프레임워크입니다. UT Austin 연구진이 2024년에 [최초 공개](https://robocasa.ai/assets/robocasa_rss24.pdf)했습니다. 최신 버전인 **RoboCasa365**는 기존 릴리스를 기반으로 시뮬레이션에서의 대규모 학습 및 벤치마킹을 지원하는 새로운 기능을 대폭 추가했습니다. RoboCasa365의 네 가지 핵심 요소는 다음과 같습니다.
- **다양한 태스크**: 대규모 언어 모델(LLM)의 가이드를 받아 설계된 365개 태스크
- **다양한 에셋**: 2,500개 이상의 주방 씬과 3,200개 이상의 3D 오브젝트 포함
- **고품질 시연 데이터**: 인간 시연 600시간 이상, 자동 궤적 생성 도구로 만든 로봇 데이터셋 1,600시간 이상
- **벤치마킹 지원**: Diffusion Policy, π, GR00T 등 인기 정책 학습 방법 및 [리더보드](https://robocasa.ai/leaderboard.html)에 등록된 사용자 제출 모델


이 가이드에는 설치 및 설정 방법이 담겨 있습니다. 추가 정보는 아래 자료를 참고하세요.

[**[홈페이지]**](https://robocasa.ai) &ensp; [**[문서]**](https://robocasa.ai/docs/introduction/overview.html) &ensp; [**[RoboCasa365 논문]**](https://robocasa.ai/assets/robocasa365_iclr26.pdf) &ensp; [**[RoboCasa 원 논문]**](https://robocasa.ai/assets/robocasa_rss24.pdf) &ensp; [**[리더보드]**](https://robocasa.ai/leaderboard.html)

-------
## 설치
RoboCasa는 주요 컴퓨팅 플랫폼에서 모두 동작합니다. 가장 쉬운 설치 방법은 [Anaconda](https://www.anaconda.com/) 패키지 관리 시스템을 사용하는 것입니다. 아래 순서대로 설치하세요.
1. conda 환경 생성:

   ```sh
   conda create -c conda-forge -n robocasa python=3.11
   ```
2. conda 환경 활성화:
   ```sh
   conda activate robocasa
   ```
3. robosuite 의존성 클론 및 설치 (**중요: master 브랜치를 사용하세요!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. 이 저장소 클론 및 설치:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # 선택 사항: 코드 포매터 설정.

   (선택 사항: numba/numpy 관련 문제가 발생하면 다음을 실행: conda install -c numba numba=0.56.4 -y)
   ```
5. 패키지 설치 및 에셋 다운로드:
   ```sh
   python -m robocasa.scripts.setup_macros              # 시스템 변수 설정.
   python -m robocasa.scripts.download_kitchen_assets   # 주의: 다운로드할 에셋은 약 10GB입니다.
   ```

-------
## 기본 사용법

### Gym wrapper
Gym wrapper를 사용해 환경을 생성하고 롤아웃을 실행할 수 있습니다:
```py
import gymnasium as gym
import robocasa
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain", # 'pretrain' 또는 'target' 주방 씬 및 오브젝트 사용
    seed=0 # 필요에 따라 환경 시드 설정. seed=None이면 시드 없이 실행
)

# 랜덤 행동으로 롤아웃 실행 후 영상 저장
run_random_rollouts(
    env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
)
```

### 태스크 샘플 시연 재생
**(Mac 사용자: 아래 스크립트 실행 시 "python" 앞에 "mj"를 붙이세요: `mjpython ...`)**

태스크를 선택하고 해당 태스크의 샘플 시연을 재생합니다:
```
python -m robocasa.demos.demo_tasks
```

### 주방 씬 탐색
2,500개 이상의 주방 씬을 탐색합니다:
```
python -m robocasa.demos.demo_kitchen_scenes
```

### 2,500개 이상 오브젝트 라이브러리 탐색
사람이 설계한 오브젝트와 AI 생성 오브젝트를 보고 상호작용합니다:
```
python -m robocasa.demos.demo_objects
```
참고: 기본적으로 이 데모는 objaverse 오브젝트를 표시합니다. AI 생성 오브젝트를 보려면 `--obj_types aigen` 플래그를 추가하세요.

### 로봇 텔레오퍼레이션
키보드 컨트롤러 또는 SpaceMouse로 로봇을 직접 조작합니다. 이 스크립트는 가림을 줄이고 시야를 확보하기 위해 로봇을 반투명하게 렌더링합니다.
```
python -m robocasa.demos.demo_teleop
```
참고: SpaceMouse를 사용하는 경우, `robocasa/macros_private.py`에서 `SPACEMOUSE_PRODUCT_ID`를 사용 중인 모델에 맞게 수정해야 할 수 있습니다.

-------
## Windows 및 로컬 추가 기능

이 저장소(로컬 개발 브랜치)에는 Windows 환경과 MuJoCo 공식 에셋 연동을 위해 아래 기능이 추가되어 있습니다. upstream [robocasa/robocasa](https://github.com/robocasa/robocasa) 공식 README에는 아직 없을 수 있습니다.

### Windows에서 venv로 설치 (conda 대안)

Anaconda가 없을 때 PowerShell 예시:

```powershell
cd D:\Github\artzy_github\Robot_Simulation\robocasa
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ..\robosuite
pip install -e .
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

실행할 때마다 `.\.venv\Scripts\Activate.ps1`로 가상 환경을 활성화한 뒤 `python`을 사용하세요.

### Windows 화면 뷰어 (pygame)

`opencv-python-headless`(lerobot 등) 때문에 OpenCV 창(`cv2.imshow`)이 동작하지 않는 Windows 환경에서는 **pygame 뷰어**로 자동 전환됩니다.

| 스크립트 | 동작 |
|----------|------|
| `python -m robocasa.demos.demo_tasks` | human demo 재생, `Opening viewer (pygame window)...` |
| `python -m robocasa.demos.demo_teleop` | 키보드/SpaceMouse 텔레오퍼레이션 |

재생·텔레op 종료: 뷰어 창 닫기, 뷰어에서 **Esc/Enter**, 또는 터미널에서 **Enter**.

GUI 없이 영상만 저장:

```powershell
python -m robocasa.demos.demo_tasks --task TurnOnSinkFaucet --render_offscreen --video_path D:\tmp\robocasa_videos
```

### demo_tasks 태스크 목록 확장

`demo_tasks`는 atomic 태스크 외에 **Navigation** 태스크(예: `HotDogSetup`, `DeliverStraw`, `GatherTableware`)도 선택할 수 있습니다. 태스크 번호를 입력하면 해당 human demo가 없을 때 자동으로 다운로드를 제안합니다.

### MuJoCo 공식 MJCF 객체 (`mujoco_official`)

[google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) 저장소 `model/`의 MJCF를 RoboCasa 조작 객체로 변환·등록하는 기능입니다.

**upstream 모델 clone (별도 폴더, pip `mujoco` 패키지와 무관):**

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco.git ..\mujoco-models
cd ..\mujoco-models
git sparse-checkout set model/mug model/cube model/cards model/cube/assets model/cards/assets
```

**RoboCasa로 import (mug + playing card 일괄):**

```powershell
pip install trimesh
python robocasa/scripts/asset_scripts/import_mujoco_batch.py
```

변환 결과: `robocasa/models/assets/objects/mujoco_official/`  
레지스트리: `kitchen_objects.py`의 `mug`, `mujoco_playing_card` → `mujoco_official`

Kitchen 환경에서 MuJoCo 공식 객체만 쓰기:

```python
from robocasa.utils.env_utils import create_env

env = create_env(
    "PickPlaceCounterToCabinet",
    split="pretrain",
    obj_registries=("mujoco_official",),
)
env.reset()
```

**Rubik's cube 등 articulation 전체 씬**은 Kitchen에 merge하지 않고 별도 sandbox:

```powershell
python -m robocasa.demos.demo_mujoco_physics --model cube
python -m robocasa.demos.demo_mujoco_physics --model mug
```

자세한 인벤토리·제약: [`Analysis/mujoco-mjcf-integration.md`](Analysis/mujoco-mjcf-integration.md)

### 로봇 변경

기본 로봇은 **PandaOmron**입니다. 주방 씬 탐색 시 로봇 지정:

```powershell
python -m robocasa.demos.demo_kitchen_scenes --robot GR1FixedLowerBody
```

데모 수집:

```powershell
python -m robocasa.scripts.collect_demos --environment PickPlaceCounterToCabinet --robots GR1FixedLowerBody
```

`demo_teleop`은 현재 PandaOmron 고정입니다. GR1 사용 시 `mink` 설치가 필요할 수 있습니다. Gym/정책 학습 파이프라인은 PandaOmron 관측·행동 공간에 맞춰져 있습니다.

### 시작 시 경고 메시지 (무시 가능)

| 메시지 | 의미 |
|--------|------|
| `mimicgen environments not imported` | MimicGen 미설치. human demo 재생·텔레op에는 **영향 없음**. 합성 데이터 생성 시에만 필요. |
| `Could not import robosuite_models` | 추가 로봇 모델 패키지 없음. PandaOmron만 쓰면 **무관**. |
| `Could not load the mink-based whole-body IK` | GR1 whole-body IK용. PandaOmron만 쓰면 **무관**. |

-------
## 태스크, 데이터셋, 정책 학습 및 추가 사용 사례
태스크, 데이터셋, 벤치마킹 등에 대한 자세한 내용은 [문서 페이지](https://robocasa.ai/docs/introduction/overview.html)를 참고하세요.

-------
## 릴리스
* [2026/5/12] **v1.0.1**: 일관성을 위해 모든 태스크의 horizon 길이를 1.5배로 업데이트했습니다. 평가 실행 시 최신 버전으로 업데이트하세요.
* [2026/2/18] **v1.0**: RoboCasa365 릴리스. 365개 태스크, 2,500개 이상 주방 씬, 2,200시간 이상 로봇 시연 데이터, 벤치마킹 지원 포함.
* [2024/10/31] **v0.2**: RoboSuite `v1.5`를 백엔드로 사용. 커스텀 로봇 구성, 복합 컨트롤러, 추가 텔레오퍼레이션 장치, 사실적인 렌더링 지원 개선.

-------
## 라이선스
코드: [MIT License](https://opensource.org/license/mit)

에셋 및 데이터셋: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)

-------
## 인용

**RoboCasa365:**

```bibtex
@inproceedings{robocasa365,
  title={RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author={Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

**RoboCasa (Original Release):**

```bibtex
@inproceedings{robocasa2024,
  title={RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author={Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2024}
}
```
