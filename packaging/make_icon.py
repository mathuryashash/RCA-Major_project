"""Generate the application icon.

Kept as a script rather than a checked-in binary alone, so the mark can be
adjusted without a drawing tool and re-rendered at every size Windows asks
for.

The mark: a telemetry trace running flat, dipping once, with that dip marked.
That is the whole product in one shape -- normal behaviour, one departure
from it, and the point being identified. It reads at 16px because the dip is
the only feature; detail was deliberately left out.

Colours are muted on purpose. The app runs on a dark navy surface and this
sits in a taskbar for days at a time, so the palette is low-saturation teal
on deep navy rather than a saturated accent that fights everything near it.

Run: python packaging/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

NAVY = (18, 24, 43)             # surface, matches the app's background family
NAVY_EDGE = (30, 40, 66)        # subtle lift at the border
TRACE = (94, 214, 197)          # muted teal, the calm state
MARK = (232, 121, 133)          # desaturated coral for the anomaly only
RENDER = 1024                   # draw large, downsample for smooth edges


def draw_icon(size: int = RENDER) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    unit = size / 100

    # Rounded square rather than a circle: it stays square-ish in the taskbar
    # and does not shrink as much as a disc at the same nominal size.
    draw.rounded_rectangle(
        [(2 * unit, 2 * unit), (98 * unit, 98 * unit)],
        radius=22 * unit, fill=NAVY, outline=NAVY_EDGE, width=int(1.5 * unit),
    )

    # The trace: flat, one dip, flat again. Coordinates are in the same 0-100
    # space as everything else so the shape survives any render size.
    points = [
        (16, 56), (28, 56), (36, 54), (44, 57),
        (52, 78),                                   # the departure
        (60, 42), (68, 55), (76, 56), (84, 56),
    ]
    draw.line(
        [(x * unit, y * unit) for x, y in points],
        fill=TRACE, width=int(5.5 * unit), joint="curve",
    )

    # One marker, on the dip. Drawn with a navy ring so it stays legible where
    # it overlaps the trace at small sizes.
    cx, cy, r = 52 * unit, 78 * unit, 9 * unit
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=NAVY)
    r = 6.5 * unit
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=MARK)

    return image


def main() -> None:
    here = Path(__file__).resolve().parent
    assets = here.parent / "assets"
    assets.mkdir(exist_ok=True)

    master = draw_icon()
    master.resize((512, 512), Image.LANCZOS).save(assets / "logo.png")

    # Windows picks whichever size it needs; supplying them all avoids the
    # blurry rescale it does otherwise.
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    master.save(assets / "logo.ico", sizes=sizes)

    print(f"wrote {assets / 'logo.png'} and {assets / 'logo.ico'}")


if __name__ == "__main__":
    main()
