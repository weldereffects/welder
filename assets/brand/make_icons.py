# Builds Welder's app icons and splash from the two source images in assets/brand.
from PIL import Image, ImageDraw, ImageFilter
src = Image.open("welder-logo-square.png").convert("RGB")
bg = (16, 17, 20)
box = (290, 360, 1760, 1330)
piece = src.crop(box)
m = Image.new("L", piece.size, 0); ImageDraw.Draw(m).rectangle((60,60,piece.width-61,piece.height-61), fill=255)
m = m.filter(ImageFilter.GaussianBlur(30))
S = 1720
canvas = Image.new("RGB", (S,S), bg)
cx, cy = 1023, 843
canvas.paste(piece, (S//2-(cx-box[0]), S//2-(cy-box[1])), m)
def rounded(img, size, inset, radius_frac=0.2237):
    full = Image.new("RGBA", (size,size), (0,0,0,0))
    inner = size-2*inset
    im = img.resize((inner,inner), Image.LANCZOS).convert("RGBA")
    mk = Image.new("L", (inner*4,inner*4), 0)
    ImageDraw.Draw(mk).rounded_rectangle((0,0,inner*4-1,inner*4-1), radius=int(inner*4*radius_frac), fill=255)
    im.putalpha(mk.resize((inner,inner), Image.LANCZOS))
    if inset:
        sh = Image.new("RGBA", (size,size), (0,0,0,0)); shm = Image.new("L", (size,size), 0)
        ImageDraw.Draw(shm).rounded_rectangle((inset,inset+inset//6,size-inset,size-inset+inset//6), radius=int(inner*radius_frac), fill=110)
        sh.putalpha(shm.filter(ImageFilter.GaussianBlur(inset/4)))
        full = Image.alpha_composite(full, sh)
    full.alpha_composite(im, (inset,inset))
    return full
app = rounded(canvas, 1024, 0, 0.18)
app.save("welder-icon-1024.png")
mac = rounded(canvas, 1024, 100); mac.save("welder-icon-macos-1024.png")
app.resize((256,256), Image.LANCZOS).save("welder-icon-256.png")
app.save("welder.ico", sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
mac.save("Welder.icns")
app.resize((64,64), Image.LANCZOS).save("welder-icon-64.png")
ban = Image.open("welder-banner.png").convert("RGB")
ban.resize((1100, round(1100*ban.height/ban.width)), Image.LANCZOS).save("welder-splash.png", optimize=True)
