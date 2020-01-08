import directory_management as dm
import result as rs
import numpy as np
from tkinter import *
from displayconfig import *

# Window
window = Toplevel()
window.title("Apparel Classifier")
window.geometry('360x640')
window.resizable(0, 0)

# Frame & Canvas
frame = Frame(window)
frame.pack()
canvas = Canvas(frame, bg="blue", width=360, height=640)
canvas.pack()

# Assets
bg = PhotoImage(file="assets/bg.png")
button_pressed = PhotoImage(file="assets/button-pressed.png")
button_unpressed = PhotoImage(file="assets/button-unpressed.png")

# Variables
cursor_x = 0
cursor_y = 0

# initial
canvas.create_image(180, 320, image=bg)
canvas.create_image(180, 400, image=button_unpressed)
canvas.create_text(180, 400, text="Choose File", fill="white", font=("arial", 16))
canvas.create_image(180, 500, image=button_unpressed)
canvas.create_text(180, 500, text="Live Camera", fill="white", font=("arial", 16))

def upload():
	file = dm.getfile("Select apparel image")
	if (file == ''): return
	return rs.result(file)

def livecamera():
	apparel_name = rs.live()
	# return rs.display_result(apparel_name)

def leftclick(event):
	# Button 1 Click Event
	if cursor_x >= 75 and cursor_x <= 285 and cursor_y >= 370 and cursor_y <= 430:
		upload()
	# Button 2 Click Event
	if cursor_x >= 75 and cursor_x <= 285 and cursor_y >= 470 and cursor_y <= 530:
		livecamera()

# updates cursor position
def motion(event):
	canvas.delete('all')
	canvas.create_image(180, 320, image=bg)
	global cursor_x, cursor_y
	cursor_x = event.x
	cursor_y = event.y
	# print('{}, {}'.format(cursor_x, cursor_y))
	tick()

def tick():
	# Check Button 1 Area
	if cursor_x >= 75 and cursor_x <= 285 and cursor_y >= 370 and cursor_y <= 430:
		canvas.create_image(180, 400, image=button_pressed)
		canvas.create_text(180, 400, text="Choose File", fill="blue", font=("arial", 16))
	else:
		canvas.create_image(180, 400, image=button_unpressed)
		canvas.create_text(180, 400, text="Choose File", fill="white", font=("arial", 16))

	# Check Button 2 Area
	if cursor_x >= 75 and cursor_x <= 285 and cursor_y >= 470 and cursor_y <= 530:
		canvas.create_image(180, 500, image=button_pressed)
		canvas.create_text(180, 500, text="Live Camera", fill="blue", font=("arial", 16))
	else:
		canvas.create_image(180, 500, image=button_unpressed)
		canvas.create_text(180, 500, text="Live Camera", fill="white", font=("arial", 16))

window.bind('<Motion>', motion)
window.bind('<Button-1>', leftclick)
window.mainloop()