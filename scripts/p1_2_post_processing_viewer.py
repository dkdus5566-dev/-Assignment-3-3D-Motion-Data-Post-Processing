"""기존 높이 보정 + Viser 전후 비교.

기본 실행: python p1_2_post_processing_viewer.py
제공된 높이 보정 파일만 다시 생성: python p1_2_post_processing_viewer.py --process
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from glob import glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")



def load_raw_keypoints(npz_path):
    """Load BlazePose raw data (T,33,3)."""
    with np.load(npz_path, allow_pickle=False) as file:
        data = file["data"]
    df = pd.DataFrame(data, columns=["frame","landmark","x","y","z","visibility"])

    pts_seq = []

    for f, group in df.groupby("frame"):
        group = group.sort_values("landmark")
        pts = group[["x","y","z"]].values  # (33,3)
        pts_seq.append(pts)

    return np.array(pts_seq)   # (T,33,3)


def create_height_corrected_target(raw_seq):
    """
    raw_seq: (T,33,3)
    Return target height corrected (T,33,3)
    """

    seq = raw_seq.copy()

    # BlazePose foot joints
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 모든 프레임에서 발의 y값 중 최소값 → ground level
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    
    # reshape to (T,1) → broadcast to (T,33)
    frame_ground = foot_y[:, None]

    # 전체 시퀀스를 ground=0으로 맞춤
    seq[:, :, 1] -= frame_ground

    # 땅 아래(-)로 떨어지지 않게 clamp
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)

    return seq

def export_unity_csv(motion, output_path):
    """
    motion: (T, 33, 3)
    Export Unity-compatible CSV.
    """

    T, n_landmarks, _ = motion.shape

    rows = []

    for t in range(T):
        for landmark in range(n_landmarks):

            x, y, z = motion[t, landmark]

            rows.append([
                t,
                landmark,
                float(x),
                float(y),
                float(z),
                1.0
            ])

    df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "landmark",
            "x",
            "y",
            "z",
            "visibility"
        ]
    )

    # 문자열 경로 처리
    output_path = os.path.abspath(output_path)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Unity CSV saved: {output_path}")
    print(f"Frames: {T}")
    print(f"Landmarks: {n_landmarks}")
    print(f"Total rows: {len(df)}")

def process_raw_files():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = glob(f"{RAW_KEYPOINTS_DIR}/*.npz")
    print(f"Found {len(paths)} raw keypoint files")

    for path in paths:
        base = os.path.basename(path).replace(".npz", "")

        # -------------------------------
        # 1) Load raw BlazePose sequence
        # -------------------------------
        raw_seq = load_raw_keypoints(path)      # (T,33,3)
        raw_out_path = os.path.join(OUT_DIR, f"{base}_raw.npy")
        np.save(raw_out_path, raw_seq)
        print(f"Saved raw → {raw_out_path}")

        # ---------------------------------------
        # 2) Create height-corrected target (GT)
        # ---------------------------------------
        target_seq = create_height_corrected_target(raw_seq)
        target_out_path = os.path.join(OUT_DIR, f"{base}_target.npy")
        np.save(target_out_path, target_seq)
        print(f"Saved target → {target_out_path}")

        OUTPUT_PATH = (
                os.path.join(OUT_DIR,
                f"{base}_target.csv")
            )
        export_unity_csv(target_seq,OUTPUT_PATH)

    print("\nDone processing all keypoint files.")


# ----------------------------------------------------------
# 추가 시각화: 같은 프레임을 좌우에서 동시에 재생
# ----------------------------------------------------------
BONES = np.array([
    (0,11),(0,12),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
])
FPS = 30
FOCUS = {
    "왼쪽 손목 (15)": (15,),
    "양쪽 발끝 (31, 32)": (31, 32),
    "오른쪽 손목 (16)": (16,),
}
BASELINE_MODE = "수업 높이 보정"
RAW_MODE = "추출 원본"


def load_comparison(stem, baseline_path=None, reviewed_path=None, raw_path=None):
    """Load outputs without rewriting any data. Raw pose is optional."""
    baseline_path = baseline_path or os.path.join(OUT_DIR, f"{stem}_target.npy")
    reviewed_path = reviewed_path or os.path.join(OUT_DIR, f"{stem}_reviewed.npy")
    raw_path = raw_path or os.path.join(RAW_KEYPOINTS_DIR, f"{stem}.npz")

    if not os.path.isfile(reviewed_path):
        raise FileNotFoundError(f"최종 보정 NPY를 찾을 수 없습니다: {reviewed_path}")
    reviewed = np.load(reviewed_path, allow_pickle=False)
    raw_cached = os.path.join(OUT_DIR, f"{stem}_raw.npy")
    if os.path.isfile(raw_path):
        raw = load_raw_keypoints(raw_path)
    elif os.path.isfile(raw_cached):
        raw = np.load(raw_cached, allow_pickle=False)
    else:
        raw = None

    if os.path.isfile(baseline_path):
        baseline = np.load(baseline_path, allow_pickle=False)
    elif raw is not None:
        # 재생용으로만 계산한다. NPY/CSV로 다시 저장하지 않는다.
        baseline = create_height_corrected_target(raw)
    else:
        raise FileNotFoundError(f"높이 보정 NPY 또는 원본 NPZ가 필요합니다: {baseline_path}")

    for name, data in [("기본 보정", baseline), ("최종 보정", reviewed), ("원본", raw)]:
        if data is None:
            continue
        if (data.ndim != 3 or data.shape[1:] != (33, 3)
                or data.shape[0] < 2 or not np.isfinite(data).all()):
            raise ValueError(f"{name}은 유효한 (프레임 수, 33, 3) 배열이어야 합니다.")
        if data.shape != reviewed.shape:
            raise ValueError(f"{name}과 최종본의 프레임 수가 다릅니다. 같은 클립을 선택하세요.")
    return baseline, reviewed, raw


def run_comparison(baseline, reviewed, raw=None, port=8080, book_marks=True):
    try:
        import viser
    except ImportError as exc:
        raise SystemExit("기존 Viser를 실행하던 Python 환경을 사용하세요. viser가 없으면 설치가 필요합니다.") from exc

    # 골반 상대 좌표는 그대로 두고 화면에서 좌우 간격만 더한다.
    # y = 위, x = 좌우, z = 깊이. 원본 좌표 파일에는 오프셋을 저장하지 않는다.
    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.scene.set_up_direction((0.0, 1.0, 0.0))
    @server.on_client_connect
    def setup_camera(client):
        client.camera.up_direction = (0.0, 1.0, 0.0)
        client.camera.position = (0.7, 1.55, 5.7)
        client.camera.look_at = (0.0, 0.72, 0.0)

    ground = np.array([[[x,0,-.72],[x,0,.72]] for x in (-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.)]
                      + [[[-2.,0,z],[2.,0,z]] for z in (-.72,-.36,0.,.36,.72)], dtype=np.float32)
    server.scene.add_line_segments("/ground", points=ground, colors=(174,190,201), line_width=1.0)

    sides = [(-1.05, (55,95,212), baseline), (1.05, (0,154,128), reviewed)]
    rigs = []
    for index, (x, color, seq) in enumerate(sides):
        offset = np.array([x, 0.0, 0.0], dtype=np.float32)
        pose = (seq[0] + offset).astype(np.float32)
        lines = server.scene.add_line_segments(
            f"/pose{index}/bones", points=pose[BONES], colors=color, line_width=3.0)
        points = server.scene.add_point_cloud(
            f"/pose{index}/joints", points=pose,
            colors=np.tile(np.array(color, dtype=np.uint8), (33, 1)), point_size=.024)
        highlight = [
            server.scene.add_icosphere(f"/pose{index}/focus{j}", radius=.043,
                                      color=c, position=(0.0, 0.0, 0.0))
            for j, c in enumerate(((245,145,36), (205,65,101)))
        ]
        label = server.scene.add_label(
            f"/pose{index}/label", "수업 높이 보정" if index == 0 else "최종 보정",
            position=(x, 1.93, 0.0))
        rigs.append((offset, lines, points, highlight, label))

    gui = server.gui
    gui.add_markdown("### 1118 모션 전후 비교\n"
                     "왼쪽은 비교 기준, 오른쪽은 최종본입니다. "
                     "정지한 뒤 프레임 번호를 바꾸면 관절 위치를 자세히 볼 수 있어요.")
    options = [BASELINE_MODE] + ([RAW_MODE] if raw is not None else [])
    left_mode = gui.add_dropdown("왼쪽 데이터", options=options, initial_value=BASELINE_MODE)
    play = gui.add_checkbox("재생", initial_value=False)
    speed = gui.add_dropdown("재생 속도", options=["0.25×","0.5×","1×"], initial_value="1×")
    frame = gui.add_slider("프레임 (0부터)", min=0, max=len(reviewed)-1, step=1, initial_value=0)
    previous = gui.add_button("이전 프레임")
    following = gui.add_button("다음 프레임")
    focus = gui.add_dropdown("강조할 관절", options=list(FOCUS), initial_value="왼쪽 손목 (15)")
    status = gui.add_text("원본 시간", initial_value="", disabled=True)
    jump_readout = gui.add_text("직전 프레임부터 3D 이동량 (좌 / 우)", initial_value="", disabled=True)
    if book_marks and len(reviewed) > 362:
        wrist = gui.add_button("F67 · 왼쪽 손목 튐")
        arms = gui.add_button("F287 · 빠른 팔 전환")
        toes = gui.add_button("F361 · 다리 좌우 뒤바뀜")

    def seek(index):
        play.value = False
        frame.value = max(0, min(len(reviewed)-1, index))

    @previous.on_click
    def _(_event):
        seek(int(frame.value)-1)

    @following.on_click
    def _(_event):
        seek(int(frame.value)+1)

    if book_marks and len(reviewed) > 362:
        @wrist.on_click
        def _(_event):
            focus.value = "왼쪽 손목 (15)"
            seek(67)

        @arms.on_click
        def _(_event):
            focus.value = "오른쪽 손목 (16)"
            seek(287)

        @toes.on_click
        def _(_event):
            focus.value = "양쪽 발끝 (31, 32)"
            seek(361)

    def draw(index, mode, focus_name):
        left = raw if mode == RAW_MODE and raw is not None else baseline
        joints = FOCUS[focus_name]
        for rig, seq in zip(rigs, (left, reviewed)):
            offset, lines, points, highlights, _label = rig
            pose = np.ascontiguousarray(seq[index] + offset, dtype=np.float32)
            lines.points = pose[BONES]
            points.points = pose
            for h, obj in enumerate(highlights):
                obj.visible = h < len(joints)
                if h < len(joints):
                    obj.position = tuple(pose[joints[h]].tolist())
        rigs[0][-1].text = mode
        status.value = f"F{index:03d} / {index/FPS:.2f}초"
        if index == 0:
            jump_readout.value = "—"
            return
        summary = []
        for j in joints:
            d1 = np.linalg.norm(left[index,j] - left[index-1,j])
            d2 = np.linalg.norm(reviewed[index,j] - reviewed[index-1,j])
            summary.append(f"{j}번  {d1:.3f} / {d2:.3f}")
        jump_readout.value = "   |   ".join(summary)

    pos = 0.0
    last_slider = int(frame.value)
    shown = None
    last_time = time.monotonic()
    print(f"Viser 비교 뷰어: http://127.0.0.1:{server.get_port()}  (종료: Ctrl+C)")
    try:
        while True:
            now = time.monotonic()
            elapsed = min(now - last_time, 0.12)
            last_time = now
            requested = int(frame.value)
            if requested != last_slider:
                pos = float(requested)
            elif play.value:
                pos = (pos + elapsed * FPS * float(speed.value[:-1])) % len(reviewed)
            index = int(pos)
            if index != requested:
                frame.value = index
            last_slider = index
            state = (index, left_mode.value, focus.value)
            if state != shown:
                draw(*state)
                shown = state
            time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description="수업 기본 높이 보정과 Viser 모션 비교")
    parser.add_argument("--process", action="store_true", help="기존 방식으로 raw/target NPY와 CSV만 생성")
    parser.add_argument("--clip", default="1118")
    parser.add_argument("--baseline", help="수업 높이 보정 NPY 경로")
    parser.add_argument("--reviewed", help="최종 보정 NPY 경로")
    parser.add_argument("--raw", help="추출 원본 NPZ 경로")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if args.process:
        process_raw_files()
        return
    try:
        baseline, reviewed, raw = load_comparison(args.clip, args.baseline, args.reviewed, args.raw)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{len(reviewed)} 프레임 × 33 관절, 30 FPS. 원본 파일을 수정하지 않습니다.")
    run_comparison(baseline, reviewed, raw, args.port, args.clip == "1118")


if __name__ == "__main__":
    main()
