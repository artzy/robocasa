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
