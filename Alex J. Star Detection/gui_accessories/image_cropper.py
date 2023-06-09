import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

class ImageCropper:
    def __init__(self, image_path):
        self.root = ctk.CTk()
        self.root.title("Image Cropper")
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        self.image = Image.open(image_path)
        self.image.thumbnail((500, 500))
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)

        self.crop_button = tk.Button(self.root, text="Crop", command=self.crop_image)
        self.crop_button.pack()

        self.rect = None
        self.start_x = None
        self.start_y = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.root.mainloop()

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red')

    def on_move_press(self, event):
        cur_x = event.x
        cur_y = event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        cur_x = event.x
        cur_y = event.y

        # Normalize coordinates
        if cur_x < self.start_x:
            cur_x, self.start_x = self.start_x, cur_x
        if cur_y < self.start_y:
            cur_y, self.start_y = self.start_y, cur_y

        # Crop the image
        cropped_image = self.image.crop((self.start_x, self.start_y, cur_x, cur_y))
        cropped_image.show()

    def crop_image(self):
        # Disable the crop button
        self.crop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    image_path = "C:\\Users\\S\\Pictures\\Alexander_Jannaeus.png"
    image_cropper = ImageCropper(image_path)
