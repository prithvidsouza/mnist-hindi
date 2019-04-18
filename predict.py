from tkinter import *

class MainWindow():
    def __init__(self, width, height, title):
        self.window = Tk()
        self.window.bind("<B1-Motion>", self.draw)
        self.window.geometry("%sx%s"%(width,height))
        self.window.title(title)
        self.canvas = Canvas(self.window,width=width,height=height-50,bg="#fff")
        self.canvas.pack()
        self.predictbtn = Button(self.window,text="Predict",command=self.get_image)
        self.predictbtn.pack()
        self.clearbtn = Button(self.window,text="Clear",command=self.clear_canvas)
        self.clearbtn.pack()
        self.thickness = 20
    def draw(self,event):
        print("Drawing over {} and {}".format(event.x,event.y))
        self.canvas.create_oval(event.x, event.y, event.x + self.thickness, event.y + self.thickness, fill="black")

    def clear_canvas(self):
        print("Clearing canvas")

    def get_image(self):
        print("You pressed the button")

    def start(self):
        self.window.mainloop()

window = MainWindow(400,400,"Hindi MNIST")
window.start()
