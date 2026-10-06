"""Frame listing, in one place.

Archives produced on macOS carry AppleDouble sidecars (`._000001.jpg`) that sort
*before* the real frames and are not decodable images. A plain `glob("*.jpg")`
therefore silently shifts every frame index by one and poisons the entire evaluation,
so all frame listing goes through here.
"""
from pathlib import Path


def list_frames(frames_dir, ext=(".jpg", ".jpeg", ".png")):
    d = Path(frames_dir)
    if not d.is_dir():
        raise SystemExit(f"not a directory: {d}")
    out = [p for p in d.iterdir()
           if p.is_file() and p.suffix.lower() in ext and not p.name.startswith("._")]
    if not out:
        raise SystemExit(f"no frames in {d}")
    return sorted(out)
