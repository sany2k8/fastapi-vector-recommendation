"""Procedural product imagery for the demo catalogue.

Real product photos would need a download; these are drawn locally with Pillow so the
project seeds itself offline. Each category gets a distinct silhouette and each product
its palette colour, which gives the image vectors genuine structure to cluster on:
same-category items share a shape, same-colour items share a histogram.
"""

import colorsys
import hashlib
import io

from PIL import Image, ImageDraw

from app.seeding.catalog_data import COLORS

CANVAS = 512


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in color)  # type: ignore[return-value]


def _background(color: tuple[int, int, int]) -> Image.Image:
    """Vertical gradient from near-white to a faint tint of the product colour."""
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    for y in range(CANVAS):
        t = y / CANVAS
        r, g, b = (int(248 * (1 - t * 0.35) + c * t * 0.25) for c in color)
        draw.line(((0, y), (CANVAS, y)), fill=(r, g, b))
    return img


def _sneaker(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.polygon(
        [(90, 330), (150, 230), (250, 220), (330, 280), (420, 300), (430, 340), (90, 340)],
        fill=c,
        outline=_shade(c, 0.6),
    )
    draw.rounded_rectangle((80, 330, 435, 375), radius=18, fill=_shade(c, 0.55))
    draw.line([(150, 250), (300, 300)], fill=_shade(c, 1.5), width=10)
    draw.ellipse((180, 245, 215, 275), outline=_shade(c, 1.4), width=5)


def _jacket(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.polygon(
        [
            (190, 130),
            (322, 130),
            (400, 195),
            (365, 245),
            (345, 215),
            (345, 400),
            (167, 400),
            (167, 215),
            (147, 245),
            (112, 195),
        ],
        fill=c,
        outline=_shade(c, 0.6),
    )
    draw.line([(256, 140), (256, 400)], fill=_shade(c, 0.5), width=6)
    draw.polygon(((216, 130), (256, 190), (296, 130)), fill=_shade(c, 0.75))


def _watch(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((225, 110, 287, 220), radius=14, fill=_shade(c, 0.7))
    draw.rounded_rectangle((225, 292, 287, 402), radius=14, fill=_shade(c, 0.7))
    draw.ellipse((176, 176, 336, 336), fill=c, outline=_shade(c, 0.5), width=8)
    draw.ellipse((200, 200, 312, 312), fill=_shade(c, 1.45))
    draw.line([(256, 256), (256, 210)], fill=_shade(c, 0.4), width=6)
    draw.line([(256, 256), (295, 275)], fill=_shade(c, 0.4), width=5)


def _backpack(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((150, 160, 362, 420), radius=44, fill=c, outline=_shade(c, 0.6))
    draw.arc((190, 96, 322, 220), start=180, end=360, fill=_shade(c, 0.55), width=14)
    draw.rounded_rectangle((180, 300, 332, 380), radius=18, fill=_shade(c, 0.75))
    draw.line([(180, 285), (332, 285)], fill=_shade(c, 0.5), width=9)


def _headphones(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.arc((136, 130, 376, 370), start=180, end=360, fill=_shade(c, 0.65), width=26)
    draw.rounded_rectangle((120, 240, 190, 360), radius=30, fill=c, outline=_shade(c, 0.5))
    draw.rounded_rectangle((322, 240, 392, 360), radius=30, fill=c, outline=_shade(c, 0.5))
    draw.ellipse((136, 262, 174, 338), fill=_shade(c, 1.35))
    draw.ellipse((338, 262, 376, 338), fill=_shade(c, 1.35))


def _lamp(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.polygon(((196, 210), (316, 210), (346, 300), (166, 300)), fill=c, outline=_shade(c, 0.6))
    draw.line([(256, 300), (256, 400)], fill=_shade(c, 0.55), width=12)
    draw.ellipse((186, 386, 326, 424), fill=_shade(c, 0.5))
    draw.polygon(((200, 300), (312, 300), (350, 380), (162, 380)), fill=(255, 246, 214))


def _chair(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((160, 130, 352, 280), radius=40, fill=c, outline=_shade(c, 0.6))
    draw.rounded_rectangle((146, 270, 366, 330), radius=24, fill=_shade(c, 1.15))
    draw.line([(176, 330), (166, 420)], fill=_shade(c, 0.5), width=14)
    draw.line([(336, 330), (346, 420)], fill=_shade(c, 0.5), width=14)


def _mug(draw: ImageDraw.ImageDraw, c: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((170, 170, 330, 380), radius=26, fill=c, outline=_shade(c, 0.6))
    draw.arc((300, 210, 400, 320), start=270, end=90, fill=_shade(c, 0.6), width=22)
    draw.ellipse((170, 150, 330, 196), fill=_shade(c, 1.35), outline=_shade(c, 0.6))


_SHAPES = {
    "sneakers": _sneaker,
    "jackets": _jacket,
    "watches": _watch,
    "backpacks": _backpack,
    "headphones": _headphones,
    "lamps": _lamp,
    "chairs": _chair,
    "mugs": _mug,
}


def colour_for(name: str) -> tuple[int, int, int]:
    """Resolve a colour word, deriving a stable one for anything unknown.

    Offline generation has to cope with colours and categories the built-in palette has
    never seen, because admins import their own catalogues. Hashing the word into a hue
    keeps it deterministic — the same word always renders the same colour — while keeping
    distinct words visually distinct, which is what the image vectors key on.
    """
    known = COLORS.get(name.strip().lower())
    if known:
        return known
    digest = hashlib.blake2b(name.strip().lower().encode(), digest_size=4).digest()
    hue = int.from_bytes(digest, "big") % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.45, 0.55)
    return int(r * 255), int(g * 255), int(b * 255)


def _generic(draw: ImageDraw.ImageDraw, c: tuple[int, int, int], category: str) -> None:
    """Fallback silhouette for an unknown category — varied by name, not identical."""
    variant = hashlib.blake2b(category.encode(), digest_size=2).digest()[0] % 3
    if variant == 0:
        draw.rounded_rectangle((150, 160, 362, 372), radius=34, fill=c, outline=_shade(c, 0.6))
    elif variant == 1:
        draw.ellipse((150, 150, 362, 362), fill=c, outline=_shade(c, 0.6), width=6)
    else:
        draw.polygon([(256, 140), (372, 330), (140, 330)], fill=c, outline=_shade(c, 0.6))
    draw.line(((170, 400), (342, 400)), fill=_shade(c, 0.55), width=12)


def render(category: str, color: str | None = None) -> bytes:
    """Draw a product image for a category/colour pair and return JPEG bytes."""
    rgb = colour_for(color or category)
    img = _background(rgb)
    draw = ImageDraw.Draw(img)

    shape = _SHAPES.get(category.strip().lower())
    if shape is None:
        _generic(draw, rgb, category)
    else:
        shape(draw, rgb)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()
