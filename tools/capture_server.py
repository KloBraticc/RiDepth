import argparse
import ctypes
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mss
import numpy as np
from PIL import Image

MAGENTA = np.array([255, 0, 255])
MAGENTA_TOLERANCE = 12
IMAGE_WIDTH = 640
STUDIO_TITLE = "Roblox Studio"


class Paused(Exception):
    pass


def foreground_title():
    user32 = ctypes.windll.user32
    window = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(window)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(window, buffer, length + 1)
    return buffer.value


def require_studio():
    if STUDIO_TITLE not in foreground_title():
        raise Paused("Roblox Studio is not the active window, capture is paused")


class Capture:
    def __init__(self, root: Path):
        self.root = root
        self.rect = None
        self.session = None
        self.frames = {}
        self.lock = threading.Lock()

    def grab(self, region):
        with mss.MSS() as screen:
            shot = screen.grab(region)
        return np.asarray(shot)[:, :, 2::-1]

    def calibrate(self, payload):
        require_studio()
        with mss.MSS() as screen:
            monitor = screen.monitors[0]
            shot = np.asarray(screen.grab(monitor))[:, :, 2::-1].astype(np.int16)
        mask = np.all(np.abs(shot - MAGENTA) <= MAGENTA_TOLERANCE, axis=2)
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or columns.size == 0:
            raise ValueError("the magenta calibration overlay was not found on screen")
        self.rect = {
            "left": int(monitor["left"] + columns[0]),
            "top": int(monitor["top"] + rows[0]),
            "width": int(columns[-1] - columns[0] + 1),
            "height": int(rows[-1] - rows[0] + 1),
        }
        place = str(payload.get("place", 0))
        self.session = f"{place}_{int(time.time())}"
        (self.root / self.session).mkdir(parents=True, exist_ok=True)
        print(f"calibrated viewport {self.rect} for session {self.session}")
        return {"session": self.session, "rect": self.rect}

    def frame(self, payload):
        if self.rect is None:
            raise ValueError("calibrate before capturing frames")
        require_studio()
        image = self.grab(self.rect)
        with self.lock:
            self.frames[payload["id"]] = image
        return {"ok": True}

    def depth(self, payload):
        sample = payload["id"]
        with self.lock:
            image = self.frames.pop(sample, None)
        if image is None:
            raise ValueError(f"no frame was captured for {sample}")
        width = int(payload["width"])
        height = int(payload["height"])
        depth = np.asarray(payload["depth"], dtype=np.float32).reshape(height, width) / 100.0
        normal = np.stack(
            [np.asarray(payload[key], dtype=np.float32).reshape(height, width) / 1000.0 for key in ("nx", "ny", "nz")]
        )
        target = self.root / self.session / sample
        target.mkdir(parents=True, exist_ok=True)
        picture = Image.fromarray(image)
        picture = picture.resize((IMAGE_WIDTH, round(IMAGE_WIDTH * height / width)), Image.Resampling.LANCZOS)
        picture.save(target / "rgb.png")
        np.save(target / "depth.npy", depth)
        np.save(target / "normal.npy", normal.astype(np.float16))
        (target / "meta.json").write_text(json.dumps({"fov": payload.get("fov"), "width": width, "height": height}))
        print(f"saved {sample}")
        return {"ok": True}


def make_handler(capture: Capture):
    routes = {"/calibrate": capture.calibrate, "/frame": capture.frame, "/depth": capture.depth}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            route = routes.get(self.path)
            if route is None:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                result = route(json.loads(self.rfile.read(length)))
                body = json.dumps(result).encode()
                self.send_response(200)
            except Paused as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_response(409)
            except Exception as error:
                body = json.dumps({"error": str(error)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    arguments.data.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), make_handler(Capture(arguments.data)))
    print(f"RiDepth capture server listening on 127.0.0.1:{arguments.port}, saving to {arguments.data.resolve()}")
    server.serve_forever()


if __name__ == "__main__":
    main()
