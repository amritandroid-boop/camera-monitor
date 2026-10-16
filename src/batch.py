import argparse, base64, glob, json, time, requests, cv2

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--dir", default="data/images/motion")
ap.add_argument("--max", type=int, default=8, help="frames per call")
ap.add_argument("--width", type=int, default=512)
ap.add_argument("--model", default="qwen2.5vl:3b")
args = ap.parse_args()

files = sorted(glob.glob(f"{args.dir}/*.jpg"))[: args.max]
imgs = []
for f in files:
    im = cv2.imread(f)
    h, w = im.shape[:2]
    im = cv2.resize(im, (args.width, int(h * args.width / w)))
    imgs.append(base64.b64encode(cv2.imencode(".jpg", im)[1]).decode())

schema = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "triggered": {"type": "boolean"},
                    "caption": {"type": "string"},
                },
                "required": ["index", "triggered", "caption"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["frames", "summary"],
}
prompt = (f"You are given {len(imgs)} images in order, numbered 1 to {len(imgs)}. "
          f"For each image: {args.question} Then give a one-line summary across all images.")

print(f"{len(imgs)} frames in ONE call | {args.question}")
t = time.time()
r = requests.post("http://localhost:11434/api/chat", timeout=600, json={
    "model": args.model, "stream": False, "format": schema,
    "options": {"num_ctx": 32768},
    "messages": [{"role": "user", "content": prompt, "images": imgs}]})
data = r.json()
if "message" not in data:
    raise SystemExit(f"Ollama error (HTTP {r.status_code}): {data}")
out = json.loads(data["message"]["content"])
print(f"{time.time()-t:.1f}s\n")
for fr in out["frames"]:
    i = fr["index"]
    name = files[i - 1].split("/")[-1] if 1 <= i <= len(files) else "?"
    print(f"{i:2} {name} trig={fr['triggered']!s:5} | {fr['caption']}")
print("\nsummary:", out["summary"])
print(f"frames returned {len(out['frames'])} of {len(imgs)}")