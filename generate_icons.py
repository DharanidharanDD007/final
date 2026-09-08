"""
Extension Icon Generator for Real-Time Deepfake Detector
========================================================
Generates security badge icons at 16x16, 48x48, and 128x128 in PNG format.
Works with Pillow if installed, or falls back to pure Python PNG encoding.
"""

import os
import math
import struct
import zlib

ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
os.makedirs(ICONS_DIR, exist_ok=True)


def generate_png_pure_python(size, filepath):
    """
    Renders a cybersecurity shield badge with biometric scanner reticle
    using pure Python and saves it as a valid PNG using standard zlib.
    """
    w = h = size
    canvas = [[(15, 23, 42, 0) for _ in range(w)] for _ in range(h)]  # transparent base

    cx, cy = w / 2.0, h / 2.0
    scale = size / 128.0

    for y in range(h):
        for x in range(w):
            dx = (x - cx) / scale
            dy = (y - cy) / scale

            # Shield formula:
            # Top flat with curved corners, sides taper down to a point
            in_shield = False
            if -48 <= dx <= 48 and -48 <= dy <= 10:
                in_shield = True
            elif -48 <= dx <= 48 and 10 < dy <= 54:
                # Triangular taper
                taper = (54 - dy) / 44.0 * 48.0
                if abs(dx) <= taper:
                    in_shield = True

            if in_shield:
                # Background of shield: Dark Slate Blue #0f172a
                canvas[y][x] = (15, 23, 42, 255)

                # Outer border (Emerald / Cyan glow #00ff88 & #00d2ff)
                dist_border = min(
                    abs(dx - 48), abs(dx + 48),
                    abs(dy + 48),
                    abs(abs(dx) - ((54 - dy) / 44.0 * 48.0)) if dy > 10 else 999
                )
                if dist_border < 4 * scale or (dy < -44 and abs(dx) < 46):
                    canvas[y][x] = (0, 255, 136, 255)

                # Biometric reticle in center (Face oval)
                face_dist = math.sqrt((dx / 22.0) ** 2 + ((dy + 4) / 28.0) ** 2)
                if 0.75 <= face_dist <= 1.05:
                    canvas[y][x] = (0, 210, 255, 255)  # Cyan #00d2ff

                # Scanner line across middle
                if abs(dy + 4) < 2.5 * scale and abs(dx) < 28:
                    canvas[y][x] = (0, 255, 136, 255)  # Neon green

                # Center authentic checkmark/dot
                if math.sqrt(dx ** 2 + (dy + 4) ** 2) < 4.0 * scale:
                    canvas[y][x] = (255, 255, 255, 255)

    # Encode to PNG format
    raw_rows = []
    for row in canvas:
        raw_rows.append(b'\x00' + b''.join(bytes([p[0], p[1], p[2], p[3]]) for p in row))
    raw_data = b''.join(raw_rows)
    compressed = zlib.compress(raw_data, 9)

    def png_chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png_bytes = b'\x89PNG\r\n\x1a\n'
    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    png_bytes += png_chunk(b'IHDR', ihdr_data)
    # IDAT chunk
    png_bytes += png_chunk(b'IDAT', compressed)
    # IEND chunk
    png_bytes += png_chunk(b'IEND', b'')

    with open(filepath, "wb") as f:
        f.write(png_bytes)
    print(f"Generated: {filepath} ({size}x{size})")


def generate_all_icons():
    sizes = [16, 48, 128]
    for s in sizes:
        out_path = os.path.join(ICONS_DIR, f"icon{s}.png")
        generate_png_pure_python(s, out_path)
    print("[SUCCESS] All icons generated successfully.")


if __name__ == "__main__":
    generate_all_icons()
