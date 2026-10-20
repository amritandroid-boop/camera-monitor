import argparse, json, os, sqlite3, time, uuid, cv2
from vlm import encode, evaluate
from actions import fire
from actions import fire

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="0 = webcam, or a video path / RTSP URL")
ap.add_argument("--fps", type=float, default=1.0)
ap.add_argument("--threshold", type=float, default=1.5, help="% of pixels changed to count as motion")
ap.add_argument("--rules", default="rules.json")
ap.add_argument("--model", default="qwen2.5vl:3b")
args = ap.parse_args()

rules = [r for r in json.load(open(args.rules)) if r.get("enabled", True)]
os.makedirs("data/thumbs", exist_ok=True)
db = sqlite3.connect("data/events.db")
db.execute("""CREATE TABLE IF NOT EXISTS events(
    ts REAL, trace TEXT, rule_id TEXT, triggered INT, confidence REAL,
    caption TEXT, extra TEXT, latency REAL, thumb TEXT)""")

src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise SystemExit("cannot open source (macOS: allow camera access for Terminal)")

prev, last, seen, passed = None, 0.0, 0, 0
print(f"{len(rules)} rules active: {[r['id'] for r in rules]} | Ctrl+C to stop")
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
        if prev is None:
            prev = gray
            continue
        diff = cv2.threshold(cv2.absdiff(prev, gray), 25, 255, cv2.THRESH_BINARY)[1]
        pct = 100.0 * cv2.countNonZero(diff) / diff.size
        prev = gray
        if pct < args.threshold:
            continue
        passed += 1
        trace = uuid.uuid4().hex[:8]
        img = encode(frame)
        for rule in rules:
            t = time.time()
            try:
                out = evaluate(img, rule, args.model)
            except Exception as e:
                print(f"[{trace}] {rule['id']} ERROR {e}")
                continue
            lat = time.time() - t
            trig = bool(out.get("triggered"))
            thumb = ""
            if trig:
                thumb = f"data/thumbs/{trace}_{rule['id']}.jpg"
                cv2.imwrite(thumb, frame)
            extra = {k: out.get(k) for k in rule.get("extract", {})}
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                       (now, trace, rule["id"], int(trig), out.get("confidence"),
                        out.get("caption"), json.dumps(extra), lat, thumb))
            db.commit()
            if trig:
                print(f"    action: {fire(db, rule, out, trace, now)}")
            if trig:
                print(f"    action: {fire(db, rule, out, trace, now)}")
            print(f"[{trace}] {rule['id']:10} trig={trig!s:5} {lat:.2f}s {extra} | {out.get('caption')}")
except KeyboardInterrupt:
    pass
cap.release()
print(f"\nframes sampled {seen}, sent to VLM {passed} ({100*(1-passed/max(seen,1)):.0f}% gated out)")