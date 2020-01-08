import tkinter as tk
from tkinter import filedialog
import os.path

root = tk.Tk()
root.withdraw()

def getdir(title): 
	dir = filedialog.askdirectory(parent=root,initialdir="//",title=title)
	return dir

def getfilefromdirectory(dir): 
	files = sum([len(files) for r, d, files in os.walk(dir)])
	return files

def getfile(title):
	return filedialog.askopenfilename(parent=root,initialdir="//",title=title)