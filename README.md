# AI & Machine Learning 과제 저장소

3D 관절 좌표의 보정 전후를 비교한 **과제 3**의 코드와 결과를 담았습니다. 이전 과제인 **과제 2 · MNIST 손글씨 숫자 분류**는 문서 하단에 별도로 정리했습니다.

---

## 과제 3 · 3D 모션 데이터 후처리

오버워치 댄스 영상에서 MediaPipe BlazePose로 3D 관절 좌표를 추출했습니다. 분석에 사용한 `1118` 클립에는 507프레임, 프레임당 33개의 관절점이 있습니다.

- **기본 보정** — 매 프레임 낮은 쪽 발끝을 바닥 높이에 맞췄습니다.
- **추가 보정** — 원본 영상을 보면서 뒤바뀐 다리 관절과 한 프레임에서 튄 좌표를 수정하고, 3프레임 스무딩을 적용했습니다.
- **전후 비교** — Viser에서 기본 보정과 최종 보정을 같은 프레임으로 재생해 차이를 확인할 수 있습니다.

### 폴더 구조

```text
.
├── scripts/        관절 좌표 추출, 기본 보정, Viser 비교 코드
├── raw_videos/     추출에 사용한 원본 영상
├── raw_keypoints/  추출한 보정 전 관절 좌표
├── processed/      뷰어에서 읽는 원본·기본·최종 좌표
└── motion_1118/    1118 클립의 기본·최종 보정 결과 CSV
```

### 전후 비교 뷰어 실행

저장소의 최상위 폴더에서 Windows PowerShell을 열고 아래 명령을 실행합니다.

```powershell
python -m pip install numpy pandas viser
python scripts/p1_2_post_processing_viewer.py `
  --raw raw_keypoints/1118.npz `
  --baseline processed/1118_target.npy `
  --reviewed processed/1118_reviewed.npy
```

실행 후 브라우저에서 `http://localhost:8080`을 엽니다. **F67**에서는 손목 좌표가 튀는 구간을, **F361**에서는 다리 관절의 좌우가 뒤바뀐 구간을 비교할 수 있습니다.

---

## 과제 2 · MNIST 손글씨 숫자 분류

MNIST 데이터로 MLP와 CNN을 학습하고 두 모델의 분류 성능을 비교했습니다. 직접 쓴 손글씨는 두 모델의 예측을 확인하는 데 사용했으며, 학습에는 사용하지 않았습니다.

- [`mnist_final.py`](./mnist_final.py) — 두 모델의 학습과 성능 비교, 직접 쓴 손글씨의 예측을 수행하는 코드
- [`handwritten_mnist.npz`](./handwritten_mnist.npz) — 직접 쓴 숫자를 변환한 별도 예측용 데이터

두 파일이 같은 폴더에 있는 상태에서 `python mnist_final.py`를 실행하면 결과가 `mnist_results/`에 저장됩니다.
