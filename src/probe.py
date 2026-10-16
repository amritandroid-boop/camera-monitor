import argparse, base64, json, time, requests

URL = "http://localhost:11434/api/chat"

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--extract", default="", help="e.g. count:integer,color:string")
ap.add_argument("--image", default="data/images/test.jpg")
ap.add_argument("--model", default="qwen2.5vl:3b")
ap.add_argument("-n", type=int, default=10)
args = ap.parse_args()

ENVELOPE = {
    "triggered": {"type": "boolean"},
    "confidence": {"type": "number"},
    "caption": {"type": "string"},
}
EXTRACT = {}
for pair in filter(None, args.extract.split(",")):
    k, t = pair.split(":")
    EXTRACT[k] = {"type": t}
PROPS = {**ENVELOPE, **EXTRACT}
SCHEMA = {"type": "object", "properties": PROPS, "required": list(PROPS)}

img = base64.b64encode(open(args.image, "rb").read()).decode()


def call(prompt, fmt=None):
    body = {
        "model": args.model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [img]}],
    }
    if fmt:
        body["format"] = fmt
    t = time.time()
    r = requests.post(URL, json=body, timeout=120)
    return r.json()["message"]["content"], time.time() - t


def valid(text):
    try:
        d = json.loads(text)
    except Exception:
        return False, False
    return True, isinstance(d, dict) and all(k in d for k in PROPS)


base = f"{args.question} triggered is true if the answer is yes or the condition is met."
free_prompt = f"{base} Reply ONLY as JSON with keys: {', '.join(PROPS)}."

call(base, SCHEMA)  # warmup, not timed

for name, prompt, fmt in [("free-text", free_prompt, None), ("schema   ", base, SCHEMA)]:
    parsed = keys_ok = 0
    times, last = [], ""
    for _ in range(args.n):
        text, dt = call(prompt, fmt)
        p, k = valid(text)
        parsed += p
        keys_ok += k
        times.append(dt)
        last = text
    print(f"{name} | parsed {parsed}/{args.n} | keys ok {keys_ok}/{args.n} | avg {sum(times)/args.n:.2f}s")
    print("  sample:", last.strip()[:200])