from PIL import Image, ImageDraw, ImageFont

# canvas.save("unicode-text.png", "PNG")
# canvas.show()

def drawString(text):
	unicode_text = text
	font = ImageFont.truetype("fonts/VAGRundschrift.ttf", 28, encoding="unic")
	text_width, text_height = font.getsize(unicode_text)
	canvas = Image.new("RGBA", (text_width + 10, text_height + 10), "white")
	draw = ImageDraw.Draw(canvas)
	draw.text((5, 5), text, "black", font)
	canvas.save("unicode-text.gif", "GIF", transparency = 0)
	# canvas.show()
	return canvas

def makeString(text):
	unicode_text = text
	font = ImageFont.truetype("fonts/VAGRundschrift.ttf", 40)
	text_width, text_height = font.getsize(unicode_text)
	image = Image.new('RGBA', (text_width + 10, text_height + 10), (255,255,255,0))
	draw = ImageDraw.Draw(image)
	draw.text((5, 5), text, "black", font)
	image.save("lalala.gif", "GIF", transparency=0, fill=(255,255,255,255))
	return draw

makeString("Hohoho")