import argparse
import ctypes
import ctypes.wintypes
import time
from pathlib import Path

import mss
import numpy as np
from PIL import Image

WINDOW_TITLE = "Roblox"
IMAGE_WIDTH = 640
PROBE_SIZE = (64, 36)


def make_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        ctypes.windll.user32.SetProcessDPIAware()


def roblox_client_rect():
    user32 = ctypes.windll.user32
    window = user32.FindWindowW(None, WINDOW_TITLE)
    if not window or user32.GetForegroundWindow() != window or user32.IsIconic(window):
        return None
    rect = ctypes.wintypes.RECT()
    user32.GetClientRect(window, ctypes.byref(rect))
    origin = ctypes.wintypes.POINT(0, 0)
    user32.ClientToScreen(window, ctypes.byref(origin))
    if rect.right < 320 or rect.bottom < 180:
        return None
    return {"left": origin.x, "top": origin.y, "width": rect.right, "height": rect.bottom}


def probe(image: Image.Image):
    return np.asarray(image.resize(PROBE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gameplay"))
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--change", type=float, default=0.04)
    parser.add_argument("--limit", type=int, default=20000)
    arguments = parser.parse_args()
    make_dpi_aware()
    session = arguments.output / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    print(f"recording Roblox gameplay into {session.resolve()}, press Ctrl+C to stop")
    previous = None
    saved = 0
    with mss.MSS() as screen:
        while saved < arguments.limit:
            started = time.monotonic()
            rect = roblox_client_rect()
            if rect is not None:
                shot = np.asarray(screen.grab(rect))[:, :, 2::-1]
                image = Image.fromarray(shot)
                current = probe(image)
                brightness = current.mean()
                if 0.03 < brightness < 0.97 and (previous is None or np.abs(current - previous).mean() > arguments.change):
                    height = round(IMAGE_WIDTH * image.height / image.width)
                    image.resize((IMAGE_WIDTH, height), Image.Resampling.LANCZOS).save(session / f"{saved:06d}.png")
                    previous = current
                    saved += 1
                    if saved % 25 == 0:
                        print(f"saved {saved} frames")
            time.sleep(max(0.0, arguments.interval - (time.monotonic() - started)))
    print(f"finished with {saved} frames")


if __name__ == "__main__":
    main()
