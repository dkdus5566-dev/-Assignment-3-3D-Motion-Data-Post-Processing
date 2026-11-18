````markdown
# AI & Machine Learning 과제 저장소

이 저장소는 과제 3의 3D 모션 데이터 후처리 코드와 결과를 담고 있습니다. 이전 과제인 과제 2의 MNIST 손글씨 숫자 분류 파일은 아래에 별도로 정리했습니다.

## 과제 3 · 3D 모션 데이터 후처리

오버워치 댄스 영상에서 MediaPipe BlazePose로 3D 관절 좌표를 추출했습니다. 수업에서 제공한 바닥 높이 보정을 기본으로 적용한 뒤, 일부 프레임에서 뒤바뀐 다리 관절을 바로잡고 갑자기 튀는 좌표를 보정했습니다. 마지막으로 3프레임 스무딩을 적용했습니다.

| 폴더 | 내용 |
| --- | --- |
| [scripts](./scripts/) | 관절 좌표 추출, 기본 후처리, 전후 비교 뷰어 코드 |
| [raw_videos](./raw_videos/) | 관절 좌표 추출에 사용한 원본 영상 |
| [raw_keypoints](./raw_keypoints/) | 영상에서 추출한 보정 전 관절 좌표 |
| [processed](./processed/) | 기본 보정과 최종 보정 데이터 |
| [motion_1118](./motion_1118/) | `1118` 클립의 기본 보정과 최종 보정 결과 CSV |

### 전후 비교 뷰어 실행

저장소의 최상위 폴더에서 아래 명령을 실행합니다.

```bash
python -m pip install numpy pandas viser
python scripts/p1_2_post_processing_viewer.py --raw raw_keypoints/1118.npz --baseline processed/1118_target.npy --reviewed processed/1118_reviewed.npy
```

실행 후 브라우저에서 `http://localhost:8080`을 열면 같은 프레임에서 기본 보정과 최종 보정 결과를 비교할 수 있습니다.

## 과제 2 · MNIST 손글씨 숫자 분류

MNIST 데이터로 MLP와 CNN을 학습하고 두 모델의 분류 성능을 비교했습니다. 학습에 사용한 MNIST 데이터와 별도로, 직접 작성한 손글씨 이미지에도 두 모델을 적용해 예측 결과를 확인했습니다.

| 파일 | 내용 |
| --- | --- |
| [mnist_final.py](./mnist_final.py) | MLP와 CNN의 학습, 성능 비교, 직접 작성한 손글씨의 예측을 수행하는 코드 |
| [handwritten_mnist.npz](./handwritten_mnist.npz) | 직접 작성한 손글씨를 변환한 별도 예측용 데이터. 모델 학습에는 사용하지 않았습니다. |

두 파일을 같은 폴더에 두고 `python mnist_final.py`를 실행하면 결과가 `mnist_results/` 폴더에 저장됩니다.
````
