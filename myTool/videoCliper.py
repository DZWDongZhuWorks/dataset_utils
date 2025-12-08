import cv2
import time
from tqdm import tqdm
import os
import argparse
import numpy as np
from typing import Optional, Tuple, List
import math
import cv2
import numpy as np

def draw_label_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int] = (5, 25),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """
    在 frame 上畫「黑底白字」標籤。
    org 是文字左下角的位置（與 cv2.putText 相同定義）。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 先取得文字尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = org
    pad = 5  # 背景留一點邊距

    # 計算黑底矩形的兩個角
    # y 是 baseline（左下角），上方要扣掉 text_h
    rect_tl = (max(0, x - pad), max(0, y - text_h - pad))
    rect_br = (x + text_w + pad, y + baseline + pad)

    # 畫黑色實心底
    cv2.rectangle(
        frame,
        rect_tl,
        rect_br,
        (0, 0, 0),
        cv2.FILLED,
    )

    # 再畫白字
    cv2.putText(
        frame,
        text,
        org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def interactive_preview_tui(
    frames: List[np.ndarray],
    init_positions: List[Tuple[int,int]],
    init_canvas_size: Tuple[int,int]
) -> Tuple[List[Tuple[int,int]], Tuple[int,int]]:
    """
    TUI 互動預覽：
    [ ] 切換影片
    Enter 切換移動模式
    wasd (小寫) 微移 1px
    WASD (大寫) 大移 100px
    R/r 重設所有移動 (還原初始位置)
    Q/q 取消操作 (還原初始位置並退出)
    Esc 結束並確定位置
    回傳最終 positions 與 canvas_size。
    """
    # 保存一份初始狀態，以供重設或取消時還原
    orig_positions = init_positions.copy()
    positions = orig_positions.copy()
    sel = 0
    move_mode = False

    win = "Preview ([ ] sselect | Enter change mode | w/a/s/d move | W/A/S/D move 100px | R reset | Q cancel | Esc confirm)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        # 動態計算畫布大小
        canvas_w = max(x + f.shape[1] for (x,y), f in zip(positions, frames))
        canvas_h = max(y + f.shape[0] for (x,y), f in zip(positions, frames))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 貼上並畫紅框標示選中
        for i, frame in enumerate(frames):
            x,y = positions[i]
            h, w = frame.shape[:2]
            canvas[y:y+h, x:x+w] = frame
            if i == sel:
                cv2.rectangle(canvas, (x,y), (x+w-1, y+h-1), (0,0,255), 2)

        # 顯示目前模式
        mode_text = "MOVE" if move_mode else "SELECT"
        cv2.putText(canvas, f"Mode: {mode_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(0)

        # Q / q -> 取消操作，還原初始位置並退出
        if key in (ord('q'), ord('Q')):
            positions = orig_positions.copy()
            return None, None

        # R / r -> 重設所有移動，回到初始位置
        if key in (ord('r'), ord('R')):
            positions = orig_positions.copy()
            move_mode = False
            sel = 0
            continue

        # Esc -> 確定並退出
        if key == 27:
            break

        if not move_mode:
            if key == ord('['):       # 切到上一支
                sel = (sel - 1) % len(frames)
            elif key == ord(']'):     # 切到下一支
                sel = (sel + 1) % len(frames)
            elif key == 13:           # Enter 進入移動模式
                move_mode = True
        else:
            # 判定移動量：小寫 => 1px，大寫 => 100px
            delta = 100 if key in (ord('W'), ord('A'), ord('S'), ord('D')) else 1
            dx = dy = 0
            if key in (ord('w'), ord('W')):  dy = -delta
            elif key in (ord('s'), ord('S')): dy = +delta
            elif key in (ord('a'), ord('A')): dx = -delta
            elif key in (ord('d'), ord('D')): dx = +delta
            elif key == 13:                  # Enter 退出移動模式
                move_mode = False

            # 更新座標（不低於0）
            x, y = positions[sel]
            positions[sel] = (max(0, x + dx), max(0, y + dy))

    cv2.destroyWindow(win)
    return positions, (canvas_w, canvas_h)

def compose_videos(
    input_paths: List[str],
    output_path: str,
    *,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    skip_frame: int = 0,
    output_size: Optional[Tuple[int, int]] = None,
    rotation_angles: Optional[List[int]] = None,
    positions: Optional[List[Tuple[int,int]]] = None,
    preview: bool = False,
    output_fps: Optional[float] = None,
    concat_mode: str = "overlay",  # "overlay"（原本行為）或 "sequential"（首尾相連）
    reverse: bool = False,         # 倒轉輸出：由後往前寫入，不需緩存整段影片
    auto_grid: bool = False,       # ⭐ 新增：自動以 sqrt(n) 計算 grid 行列數
    label: bool = False,
    show: bool = False,            # ⭐ 新增：處理時即時顯示畫面
) -> None:

    # 1. 開啟所有影片
    caps = [cv2.VideoCapture(p) for p in input_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise OSError(f"無法開啟第 {i} 支影片: {input_paths[i]}")
      

    # 2. 擷取各影片基本參數
    file_names = [os.path.basename(p) for p in input_paths] # ⭐ 每支影片的檔名
    src_fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    fps = output_fps if output_fps is not None else min(src_fps_list)
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    if concat_mode == "overlay":
        end_frame = min(frame_counts) if end_frame is None else min(end_frame, min(frame_counts))
    else:
        # sequential：各片段各自有 end
        per_video_end = [fc if end_frame is None else min(end_frame, fc) for fc in frame_counts]

    # 3. 讀取第一張影格以供預覽或計算尺寸
    first_frames = []
    sizes = []
    for i, cap in enumerate(caps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"無法讀取第 {i} 支影片的第一張影格")
        angle = rotation_angles[i] if rotation_angles else 0
        # 旋轉
        if angle:
            h0, w0 = frame.shape[:2]
            rad = np.deg2rad(angle)
            c, s = abs(np.cos(rad)), abs(np.sin(rad))
            rot_w, rot_h = int(h0*s + w0*c), int(h0*c + w0*s)
            M = cv2.getRotationMatrix2D((w0/2,h0/2), angle, 1.0)
            M[0,2] += (rot_w/2 - w0/2)
            M[1,2] += (rot_h/2 - h0/2)
            frame = cv2.warpAffine(frame, M, (rot_w, rot_h))
        # Resize
        if output_size:
            frame = cv2.resize(frame, output_size)
        first_frames.append(frame)
        sizes.append(frame.shape[:2][::-1])  # (w,h)

    # 4. 計算輸出尺寸
    if concat_mode == "sequential":
        # sequential：整支影片輸出尺寸固定。若未指定 output_size，沿用第一支影片處理後的尺寸。
        if output_size:
            canvas_size = tuple(output_size)
        else:
            canvas_size = sizes[0]  # (w, h)
        # sequential 模式下，positions/preview/auto_grid 無效
        positions = None
    else:
        # overlay 模式

        def make_horizontal_positions(sizes):
            positions = []
            x_offset = 0
            max_h = 0
            for w, h in sizes:
                positions.append((x_offset, 0))
                x_offset += w
                max_h = max(max_h, h)
            canvas_size = (x_offset, max_h)
            return positions, canvas_size

        def make_grid_positions(sizes):
            n = len(sizes)
            if n == 0:
                return [], (0, 0)
            # 以 sqrt(n) 決定 cols / rows
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            # 每格採用最大寬高，避免重疊
            max_w = max(w for w, h in sizes)
            max_h = max(h for w, h in sizes)

            positions = []
            for i, (w, h) in enumerate(sizes):
                r = i // cols
                c = i % cols
                x = c * max_w
                y = r * max_h
                positions.append((x, y))
            canvas_size = (cols * max_w, rows * max_h)
            return positions, canvas_size

        # 根據是否給定 positions / auto_grid / preview 來決定初始排列
        if positions is None:
            if auto_grid:
                # 自動九宮格（或 2x2, 3x4 類似）
                positions, canvas_size = make_grid_positions(sizes)
            else:
                # 預設橫向並排
                positions, canvas_size = make_horizontal_positions(sizes)
        else:
            # 使用者有手動給 positions，就依照 positions 計算畫布大小
            canvas_w = max(x + w for (x, y), (w, h) in zip(positions, sizes))
            canvas_h = max(y + h for (x, y), (w, h) in zip(positions, sizes))
            canvas_size = (canvas_w, canvas_h)

    # 5. 若開啟互動預覽，要求使用者調整
    if preview and concat_mode == "overlay":
        # 若前面沒算好 positions（理論上不會發生，但保險一下）
        if positions is None:
            if auto_grid:
                # 仍然用 grid 當起始點
                def make_grid_positions_for_preview(sizes):
                    n = len(sizes)
                    cols = math.ceil(math.sqrt(n))
                    rows = math.ceil(n / cols)
                    max_w = max(w for w, h in sizes)
                    max_h = max(h for w, h in sizes)
                    pos = []
                    for i, (w, h) in enumerate(sizes):
                        r = i // cols
                        c = i % cols
                        x = c * max_w
                        y = r * max_h
                        pos.append((x, y))
                    canvas_size = (cols * max_w, rows * max_h)
                    return pos, canvas_size
                positions, canvas_size = make_grid_positions_for_preview(sizes)
            else:
                # 預設橫向並排
                positions = []
                x_off = 0
                max_h = 0
                for w, h in sizes:
                    positions.append((x_off, 0))
                    x_off += w
                    max_h = max(max_h, h)
                canvas_size = (x_off, max_h)

        positions, canvas_size = interactive_preview_tui(first_frames, positions, canvas_size)

    if concat_mode == "overlay" and positions is None and canvas_size is None:
        print("使用者退出")
        return None
    
    # ⭐ 若需要即時顯示，先建立視窗
    preview_win = "Live Preview"
    if show:
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)

    # 6. 建立 VideoWriter
    ext = os.path.splitext(output_path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*{'.avi':'XVID', '.mp4':'mp4v', '.mov':'avc1'}.get(ext,'mp4v'))
    out = cv2.VideoWriter(output_path, fourcc, fps, canvas_size)

    # 7. 正式處理每一張影格
    if concat_mode == "overlay":
        total_frames = ((end_frame - start_frame)//(skip_frame+1))
    else:
        total_frames = 0
        for i, fc in enumerate(frame_counts):
            end_i = per_video_end[i]
            start_i = min(start_frame, end_i)
            total_frames += max(0, (end_i - start_i)//(skip_frame+1))
    pbar = tqdm(total=total_frames, unit="frame", desc="Processing")
    t0 = time.time()
    stop_flag = False  # ⭐ 新增：給 show 用的中斷旗標

    if concat_mode == "overlay":
        step = skip_frame + 1
        if reverse:
            # 由後往前的索引序列（對齊 skip 間隔，不需緩存整段）
            last = end_frame - 1
            first_idx = last - ((last - start_frame) % step)
            frame_iter = range(first_idx, start_frame - 1, -step)
        else:
            # 由前往後
            frame_iter = (f for f in range(start_frame, end_frame) if (f - start_frame) % step == 0)

        for f_idx in frame_iter:
            if stop_flag:
                break

            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            for i, cap in enumerate(caps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                # 旋轉
                angle = rotation_angles[i] if rotation_angles else 0
                if angle:
                    h0, w0 = frame.shape[:2]
                    rad = np.deg2rad(angle)
                    c, s = abs(np.cos(rad)), abs(np.sin(rad))
                    rot_w, rot_h = int(h0*s + w0*c), int(h0*c + w0*s)
                    M = cv2.getRotationMatrix2D((w0/2,h0/2), angle, 1.0)
                    M[0,2] += (rot_w/2 - w0/2)
                    M[1,2] += (rot_h/2 - h0/2)
                    frame = cv2.warpAffine(frame, M, (rot_w, rot_h))
                # Resize
                if output_size:
                    frame = cv2.resize(frame, output_size)
                # ⭐ 標註檔名（畫在這支影片自己的左上角）
                if label:
                    draw_label_with_bg(frame, file_names[i], org=(5, 25))
                x, y = positions[i]
                h, w = frame.shape[:2]
                canvas[y:y+h, x:x+w] = frame

            out.write(canvas)
            pbar.update(1)

            # ⭐ show 即時預覽
            if show:
                cv2.imshow(preview_win, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    print("🔁 使用者中斷合成")
                    stop_flag = True
                    break
    else:
        # sequential：逐支影片依序寫入
        base_w, base_h = canvas_size
        iter_vids = range(len(caps)-1, -1, -1) if reverse else range(len(caps))
        for i in iter_vids:
            if stop_flag:
                break

            cap = caps[i]
            start_i = start_frame
            end_i = per_video_end[i]
            step = skip_frame + 1

            if reverse:
                # 倒放：由該段最後一幀往前取，對齊 skip 間隔，不需緩存整段
                last = end_i - 1
                first_idx = last - ((last - start_i) % step)
                idx_iter = range(first_idx, start_i - 1, -step)
                for f_idx in idx_iter:
                    if stop_flag:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    # 旋轉
                    angle = rotation_angles[i] if rotation_angles else 0
                    if angle:
                        h0, w0 = frame.shape[:2]
                        rad = np.deg2rad(angle)
                        c, s = abs(np.cos(rad)), abs(np.sin(rad))
                        rot_w, rot_h = int(h0*s + w0*c), int(h0*c + w0*s)
                        M = cv2.getRotationMatrix2D((w0/2,h0/2), angle, 1.0)
                        M[0,2] += (rot_w/2 - w0/2)
                        M[1,2] += (rot_h/2 - h0/2)
                        frame = cv2.warpAffine(frame, M, (rot_w, rot_h))
                    # 統一輸出大小
                    if output_size:
                        frame = cv2.resize(frame, (base_w, base_h))

                    # ⭐ 標註檔名
                    if label:
                        draw_label_with_bg(frame, file_names[i], org=(5, 25))
                    else:
                        if frame.shape[1] != base_w or frame.shape[0] != base_h:
                            frame = cv2.resize(frame, (base_w, base_h))

                    out.write(frame)
                    pbar.update(1)

                    if show:
                        cv2.imshow(preview_win, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q'), ord('Q')):
                            print("🔁 使用者中斷合成")
                            stop_flag = True
                            break
            else:
                # 正放：維持原本逐幀讀取＋跳幀
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_i)
                for f_idx in range(start_i, end_i):
                    if stop_flag:
                        break

                    if (f_idx - start_i) % step != 0:
                        # 仍需移動讀取指標
                        _ = cap.read()
                        continue
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # 旋轉
                    angle = rotation_angles[i] if rotation_angles else 0
                    if angle:
                        h0, w0 = frame.shape[:2]
                        rad = np.deg2rad(angle)
                        c, s = abs(np.cos(rad)), abs(np.sin(rad))
                        rot_w, rot_h = int(h0*s + w0*c), int(h0*c + w0*s)
                        M = cv2.getRotationMatrix2D((w0/2,h0/2), angle, 1.0)
                        M[0,2] += (rot_w/2 - w0/2)
                        M[1,2] += (rot_h/2 - h0/2)
                        frame = cv2.warpAffine(frame, M, (rot_w, rot_h))
                    # 統一輸出大小
                    if output_size:
                        frame = cv2.resize(frame, (base_w, base_h))
                    else:
                        if frame.shape[1] != base_w or frame.shape[0] != base_h:
                            frame = cv2.resize(frame, (base_w, base_h))
                    # ⭐ 標註檔名
                    if label:
                        draw_label_with_bg(frame, file_names[i], org=(5, 25))

                    out.write(frame)
                    pbar.update(1)

                    if show:
                        cv2.imshow(preview_win, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q'), ord('Q')):
                            print("🔁 使用者中斷合成")
                            stop_flag = True
                            break

    pbar.close()

    # 8. 釋放
    for cap in caps:
        cap.release()
    out.release()
    if show:
        cv2.destroyWindow(preview_win)
    print(f"✅ 合併輸出完成 (耗時 {time.time()-t0:.2f}s) │ 輸出 FPS {fps:.2f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多影片合併剪輯，可同時輸入多支影片並指定相對位置，支援預覽模式。"
    )
    parser.add_argument('-i', '--inputs', nargs='+', required=True, help='多支輸入影片路徑')
    parser.add_argument('-o', '--output', required=True, help='輸出影片路徑')
    parser.add_argument('--start', type=int, default=0, help='起始幀 (含)')
    parser.add_argument('--end', type=int, default=None, help='結束幀 (含)')
    parser.add_argument('--skip', type=int, default=0, help='跳幀量')
    parser.add_argument('--output-size', type=int, nargs=2, metavar=('W','H'),
                        help='輸出解析度 (寬 高)')
    parser.add_argument('--rotation-angles', type=int, nargs='+',
                        help='每支影片的旋轉角度，順序對應 inputs')
    parser.add_argument('--positions', type=int, nargs='+',
                        help='指定位置 (x1 y1 x2 y2 ...)，數量需為 2×影片數')
    parser.add_argument('--preview', action='store_true',
                        help='啟用互動式預覽來調整各影片位置')
    parser.add_argument('--fps', type=float, default=None, help='輸出 FPS')
    parser.add_argument('--concat',
                        choices=['overlay','sequential'],
                        default='overlay',
                        help='合併模式：overlay（疊加/拼貼，多片同時出現在畫面）或 sequential（首尾相連，依序播放各影片）')
    parser.add_argument('-r', '--reverse', action='store_true',
                        help='啟用倒轉輸出：合成每幀後由後往前的索引順序寫入，不需緩存整段影片')
    parser.add_argument('--auto-grid', action='store_true',
                        help='overlay 模式下，根據影片數量自動以 sqrt(n) 計算網格行列數，排成九宮格風格')
    parser.add_argument('--label', action='store_true',
                        help='在每支影片左上角顯示檔名')
    parser.add_argument('--show', action='store_true',
                        help='處理時即時顯示畫面（按 q / Q / Esc 可中斷）')  # ⭐ 新增
    return parser.parse_args()

def main():
    args = parse_args()
    size = tuple(args.output_size) if args.output_size else None
    rots = args.rotation_angles if args.rotation_angles else [0]*len(args.inputs)
    pos = None
    if args.positions:
        assert len(args.positions) == 2 * len(args.inputs), "positions 長度需為 2×影片數"
        pos = [(args.positions[i*2], args.positions[i*2+1]) for i in range(len(args.inputs))]

    compose_videos(
        input_paths=args.inputs,
        output_path=args.output,
        start_frame=args.start,
        end_frame=args.end,
        skip_frame=args.skip,
        output_size=size,
        rotation_angles=rots,
        positions=pos,
        preview=args.preview,
        output_fps=args.fps,
        concat_mode=args.concat,
        reverse=args.reverse,
        auto_grid=args.auto_grid,
        label=args.label,
        show=args.show,  # ⭐ 新增
    )

if __name__ == '__main__':
    main()
