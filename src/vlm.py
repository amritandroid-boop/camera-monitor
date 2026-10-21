import base64, json, requests, cv2

URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:3b"
ENVELOPE = {
    "triggered": {"type": "boolean"},
    "confidence": {"type": "number"},
    "caption": {"type": "string"},
}


def build_schema(extract):
    props = {**ENVELOPE, **{k: {"type": t} for k, t in extract.items()}}
    return {"type": "object", "properties": props, "required": list(props)}


def encode(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


def evaluate(img_b64, rule, model=MODEL, retries=1):
    prompt = f"{rule['question']} Set triggered to true only if the condition is met. In caption, write one sentence describing what you see that is relevant to the question."
    body = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "format": build_schema(rule.get("extract", {})),
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(URL, json=body, timeout=120)
            return json.loads(r.json()["message"]["content"])
        except Exception:
            if attempt == retries:
                raise