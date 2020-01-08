import predict as pred
from tkinter import *
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageTk as itk

def display_result(apparel_name):
	# Window
	window = Toplevel()
	window.title("Image Result")
	window.geometry('360x640')
	window.resizable(0, 0)

	# Fonts 
	fontpath = "C:\\Users\\ACER\\Desktop\\COMP6065-Artificial_Intelligence\\ProjectAI\\apparel-classifier\\fonts\\VAGRundschrift.ttf"
	font = ImageFont.truetype(fontpath, 28)
	b, g, r, a = 255, 255, 255, 0

	image = Image.open("assets/image-result.png")
	draw = ImageDraw.Draw(image)
	draw.text((200, 210), apparel_name[0], font = font, fill = (b, g, r, a))

	newImage = itk.PhotoImage(image)
	label = Label(window, image = newImage)
	label.pack()

	window.mainloop()

def live():
    return pred.live_processing()

def result(file):
	apparel_name = pred.getprediction(file, 'image')
	return display_result(apparel_name)