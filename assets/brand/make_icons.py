# Builds Welder's app icons, splash, manual banner and SVG from the logo's shapes.
# The shapes are traced from welder-logo-source.png (1388 x 1088); they are drawn as polygons at
# 4x size and scaled down, so every size is sharp. Run from assets/brand:  python3 make_icons.py
# (the wordmark needs Inter Display Bold; the manual banner goes to docs/manual/banner.jpg).
import os
from PIL import Image, ImageDraw, ImageFont

BG = (40, 40, 40)
WHITE = (255, 255, 255)
GOLD = (255, 198, 0)
SHAPES = [  # (colour, polygon) in source pixels
    (WHITE, [(158, 241), (325, 241), (654, 855), (489, 855)]),
    (WHITE, [(421, 240), (588, 240), (917, 854), (751, 854)]),
    (GOLD, [(689, 240), (1235, 240), (1181, 355), (750, 355)]),
    (GOLD, [(803, 456), (1132, 456), (1078, 572), (865, 572)]),
    (GOLD, [(919, 673), (1029, 673), (993, 750), (960, 750)]),
]
X0 = min(x for _, p in SHAPES for x, _ in p); X1 = max(x for _, p in SHAPES for x, _ in p)
Y0 = min(y for _, p in SHAPES for _, y in p); Y1 = max(y for _, p in SHAPES for _, y in p)
MW, MH = X1 - X0, Y1 - Y0  # the mark's own size (no margin)
SS = 4  # supersampling


def draw_mark(img, x, y, w):
    """Draws the mark w pixels wide with its top-left at (x, y), on an image at SS scale."""
    s = w / MW
    d = ImageDraw.Draw(img)
    for col, poly in SHAPES:
        d.polygon([(x + (px - X0) * s, y + (py - Y0) * s) for px, py in poly], fill=col)


def mark_canvas(size, frac, bg=BG):
    """A square image with the mark centred, frac of its width."""
    big = Image.new("RGB", (size * SS, size * SS), bg)
    w = size * SS * frac
    h = w * MH / MW
    draw_mark(big, (size * SS - w) / 2, (size * SS - h) / 2, w)
    return big.resize((size, size), Image.LANCZOS)


def rounded(img, size, inset, radius_frac):
    full = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    inner = size - 2 * inset
    im = img.resize((inner, inner), Image.LANCZOS).convert("RGBA")
    mk = Image.new("L", (inner * 4, inner * 4), 0)
    ImageDraw.Draw(mk).rounded_rectangle((0, 0, inner * 4 - 1, inner * 4 - 1), radius=int(inner * 4 * radius_frac), fill=255)
    im.putalpha(mk.resize((inner, inner), Image.LANCZOS))
    full.alpha_composite(im, (inset, inset))
    return full


def lockup(W, H, mark_h_frac, bg=BG):
    """The mark with the 'Welder' wordmark beside it, centred on a W x H image."""
    big = Image.new("RGB", (W * SS, H * SS), bg)
    mh = H * SS * mark_h_frac
    mw = mh * MW / MH
    path = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
    cap = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), "W", font=ImageFont.truetype(path, 1000))
    font = ImageFont.truetype(path, int(mh * 0.44 * 1000 / (cap[3] - cap[1])))  # cap height = 44% of the mark
    d = ImageDraw.Draw(big)
    tb = d.textbbox((0, 0), "Welder", font=font)
    gap = mh * 0.16
    total = mw + gap + (tb[2] - tb[0])
    x = (W * SS - total) / 2
    y = (H * SS - mh) / 2
    draw_mark(big, x, y, mw)
    d.text((x + mw + gap - tb[0], y + (mh - (tb[3] - tb[1])) / 2 - tb[1]), "Welder", font=font, fill=WHITE)  # centred on the mark
    return big.resize((W, H), Image.LANCZOS)


# app icons: big mark, dark rounded square (the mark's own white and gold carry it at 16 px)
full = mark_canvas(1720, 0.78)
app = rounded(full, 1024, 0, 0.18)
app.save("welder-icon-1024.png")
rounded(full, 1024, 100, 0.2237).save("welder-icon-macos-1024.png")
app.resize((256, 256), Image.LANCZOS).save("welder-icon-256.png")
app.resize((64, 64), Image.LANCZOS).save("welder-icon-64.png")
# small sizes drawn directly (sharper than scaling 1024 down)
smalls = {s: rounded(mark_canvas(s * 4, 0.84), s, 0, 0.18) for s in (16, 24, 32, 48)}
icons = [smalls[s] for s in (16, 24, 32, 48)] + [app.resize((s, s), Image.LANCZOS) for s in (64, 128, 256)]
icons[-1].save("welder.ico", sizes=[(i.width, i.height) for i in icons], append_images=icons[:-1])
rounded(full, 1024, 100, 0.2237).save("Welder.icns")

lockup(1100, 622, 0.30).save("welder-splash.png", optimize=True)
os.makedirs("../../docs/manual", exist_ok=True)
lockup(1600, 600, 0.34).save("../../docs/manual/banner.jpg", quality=92)

# vector source
s = 1.0
with open("welder-logo.svg", "w") as f:
    f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MW} {MH}">\n')
    for col, poly in SHAPES:
        pts = " ".join(f"{px - X0},{py - Y0}" for px, py in poly)
        f.write(f'  <polygon fill="#{col[0]:02x}{col[1]:02x}{col[2]:02x}" points="{pts}"/>\n')
    f.write("</svg>\n")
