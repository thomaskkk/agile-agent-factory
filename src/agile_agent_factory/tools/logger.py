from datetime import datetime


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)
