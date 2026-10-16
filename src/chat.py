from contextlib import contextmanager
import itertools
import sys
import threading
import argparse, base64, json, time, requests

URL = "http://localhost:11434/api/chat"

ap = argparse.ArgumentParser()
ap.add_argument("--image", default="data/images/test.jpg")
ap.add_argument("--model", default="qwen2.5vl:3b")
args = ap.parse_args()

ENVELOPE = {
    "triggered": {"type": "boolean"},
    "confidence": {"type": "number"},
    "caption": {"type": "string"},
}


def load(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def build_schema(extract):
    props = dict(ENVELOPE)
    for pair in filter(None, (p.strip() for p in extract.split(","))):
        k, t = pair.split(":")
        props[k.strip()] = {"type": t.strip()}
    return {"type": "object", "properties": props, "required": list(props)}


def ask(img, question, schema):
    prompt = f"{question} triggered is true if the answer is yes or the condition is met."
    body = {
        "model": args.model,
        "stream": False,
        "keep_alive": "30m",
        "format": schema,
        "messages": [{"role": "user", "content": prompt, "images": [img]}],
    }
    t = time.time()
    r = requests.post(URL, json=body, timeout=120)
    return r.json()["message"]["content"], time.time() - t

@contextmanager
def spinner(msg="thinking"):
    stop = threading.Event()

    def run():
        for c in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop.is_set():
                break
            sys.stdout.write(f"\r{c} {msg}...")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(msg) + 6) + "\r")
        sys.stdout.flush()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        
img = load(args.image)
print(f"image: {args.image} | model: {args.model}")
print("ask: <question>  or  <question> | count:integer,color:string")
print("cmds: :image <path>   :quit\n")

while True:
    try:
        line = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue
    if line == ":quit":
        break
    if line.startswith(":image "):
        try:
            path = line.split(" ", 1)[1].strip()
            img = load(path)
            print(f"image set: {path}")
        except Exception as e:
            print("error:", e)
        continue
    question, _, extract = line.partition("|")
    try:
        with spinner():
            text, dt = ask(img, question.strip(), build_schema(extract))
        text, dt = ask(img, question.strip(), build_schema(extract))
        print(json.dumps(json.loads(text), indent=2), f"\n({dt:.2f}s)\n")
    except Exception as e:
        print("error:", e)