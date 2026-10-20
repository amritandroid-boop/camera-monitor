import subprocess, time, requests

_last = {}


def _notify(rule, out):
    msg = (out.get("caption") or "")[:120].replace('"', "'")
    subprocess.run(["osascript", "-e",
                    f'display notification "{msg}" with title "Camera: {rule["id"]}"'], check=True)


def _webhook(rule, out, url):
    requests.post(url, json={"rule": rule["id"], "caption": out.get("caption"),
                             "ts": time.time()}, timeout=5).raise_for_status()


def fire(db, rule, out, trace, now):
    db.execute("""CREATE TABLE IF NOT EXISTS actions(
        ts REAL, trace TEXT, rule_id TEXT, action TEXT, status TEXT)""")
    if now - _last.get(rule["id"], 0) < rule.get("cooldown", 0):
        return "cooldown"
    _last[rule["id"]] = now
    results = []
    for a in rule.get("actions", []):
        try:
            if a == "notify":
                _notify(rule, out)
            elif a.startswith("webhook:"):
                _webhook(rule, out, a.split(":", 1)[1])
            status = "ok"
        except Exception as e:
            status = f"error: {e}"
        db.execute("INSERT INTO actions VALUES (?,?,?,?,?)", (now, trace, rule["id"], a.split(":")[0], status))
        results.append(f"{a.split(':')[0]}={status}")
    db.commit()
    return ", ".join(results) or "no actions"