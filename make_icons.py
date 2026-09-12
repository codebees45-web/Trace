from PIL import Image, ImageDraw

for size in [192, 512]:
    img = Image.new("RGB", (size, size), color="#1a1a1a")
    draw = ImageDraw.Draw(img)
    draw.text((size * 0.15, size * 0.4), "MaskID", fill="white")
    img.save(f"frontend/public/icon-{size}.png")

print("Icons created.")