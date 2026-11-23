from PIL import Image, ImageOps

image = Image.open("pic1.png").convert("RGB")

image = ImageOps.fit(
    image,
    (150, 150),
    method=Image.Resampling.LANCZOS,
    centering=(0.5, 0.5),
)

image.save("pic1_1.png", quality=95)