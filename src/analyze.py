import argparse, base64, csv, glob, json, time, requests

URL = "http://localhost:11434/api/chat"
ENVELOPE = {
    "triggered": {"type": "boolean"},
    "confidence": {"type": "number"},
    "caption": {"type": "string"},
}

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--extract", default="", help="e.g. count:integer,color:string")
ap.add_argument("--dir", default="data/images/motion")
ap.add_argument("--model", default="qwen2.5vl:3b")
ap.add_argument("--out", default="data/analysis.csv")
args = ap.parse_args()

props = dict(ENVELOPE)
for pair in filter(None, args.extract.split(",")):
    k, t = pair.split(":")
    props[k] = {"type": t}
schema = {"type": "object", "properties": props, "required": list(props)}
prompt = f"{args.question} triggered is true if the answer is yes or the condition is met."

files = sorted(glob.glob(f"{args.dir}/*.jpg"))
print(f"{len(files)} frames | {args.question}\n")

rows, times = [], []
for i, f in enumerate(files, 1):
    img = base64.b64encode(open(f, "rb").read()).decode()
    body = {"model": args.model, "stream": False, "keep_alive": "30m", "format": schema,
            "messages": [{"role": "user", "content": prompt, "images": [img]}]}
    t = time.time()
    try:
        r = requests.post(URL, json=body, timeout=120)
        out = json.loads(r.json()["message"]["content"])
    except Exception as e:
        out = {"triggered": None, "caption": f"ERROR {e}"}
    dt = time.time() - t
    times.append(dt)
    out.update(file=f.split("/")[-1], latency=round(dt, 2))
    rows.append(out)
    print(f"{i:3}/{len(files)} {out['file']} trig={out.get('triggered')!s:5} {dt:.2f}s | {out.get('caption')}")

keys = ["file", "latency"] + list(props)
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

trig = sum(1 for r in rows if r.get("triggered"))
print(f"\ntriggered {trig}/{len(rows)} | avg {sum(times)/max(len(times),1):.2f}s | saved {args.out}")