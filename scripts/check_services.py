import urllib.request
for url in ["http://127.0.0.1:5173/", "http://127.0.0.1:8000/health", "http://127.0.0.1:8002/health", "http://127.0.0.1:8003/health"]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            print(url, r.status, r.read(120).decode("utf-8", "replace").replace("\n", " "))
    except Exception as exc:
        print(url, "ERROR", repr(exc))
