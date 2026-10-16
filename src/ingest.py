import argparse, os, time, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="0 = webcam, or a video path / RTSP URL")
ap.add_argument("--fps", type=float, default=1.0, help="sampling rate")
ap.add_argument("--threshold", type=float, default=1.5, help="% of pixels changed to count as motion")
ap.add_argument("--out", default="data/images/motion")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise SystemExit("cannot open source (macOS: allow camera access for Terminal)")

prev, last, seen, passed = None, 0.0, 0, 0
print("running, Ctrl+C to stop")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - last < 1.0 / args.fps:
            continue
        last = now
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        seen += 1
        if prev is not None:
            diff = cv2.threshold(cv2.absdiff(prev, gray), 25, 255, cv2.THRESH_BINARY)[1]
            pct = 100.0 * cv2.countNonZero(diff) / diff.size
            if pct >= args.threshold:
                passed += 1
                cv2.imwrite(f"{args.out}/{int(now)}.jpg", frame)
            print(f"motion {pct:5.2f}% | {'PASS' if pct >= args.threshold else 'skip'} | "
                  f"gated out {100*(1-passed/seen):.0f}% of {seen} frames")
        prev = gray
except KeyboardInterrupt:
    pass
cap.release()