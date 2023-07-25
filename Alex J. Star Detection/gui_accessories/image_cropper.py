import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

dimension = 500
ctk.set_default_color_theme("green")
"""
A module for cropping an image
"""


class ImageCropper(ctk.CTkToplevel):
    def __init__(self, image_path):
        super().__init__()
        self.create_window(image_path)

    def create_window(self, image_path):
        self.title("Image Cropper")
        self.focus()
        fg_color = self.cget("fg_color")[1]
        self.canvas = tk.Canvas(self, width=dimension, height=dimension, bg=fg_color)
        self.canvas.pack()

        self.image = None
        og_image = Image.open(image_path)
        self.set_image(og_image)
        # width = self.canvas.winfo_width()
        # height = self.canvas.winfo_height()
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        self.crop_button = ctk.CTkButton(self, text="Crop", command=self.crop_image)
        self.crop_button.pack()
        self.confirm_button = ctk.CTkButton(self, text="Confirm", command=self.destroy)
        self.confirm_button.pack()

        self.rect = None
        self.start_x = None
        self.start_y = None

        # commands to draw a rectangle using the mouse
        self.canvas.bind("<ButtonPress-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.resize_rect)
        self.canvas.bind("<ButtonRelease-1>", self.confirm_rect)

        # self.mainloop()
    def set_image(self, image):
        self.image = image
        temp = self.image
        temp.thumbnail((dimension, dimension))
        self.photo_image = ImageTk.PhotoImage(temp)

    def start_rect(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red')

    def resize_rect(self, event):
        cur_x = event.x
        cur_y = event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def confirm_rect(self, event):
        cur_x = event.x
        cur_y = event.y

        # Normalize coordinates
        if cur_x < self.start_x:
            cur_x, self.start_x = self.start_x, cur_x
        if cur_y < self.start_y:
            cur_y, self.start_y = self.start_y, cur_y

        # Crop the image
        self.cropped = self.image.crop((self.start_x, self.start_y, cur_x, cur_y))
        self.cropped_imagetk = ImageTk.PhotoImage(self.cropped)

    def crop_image(self):
        self.canvas.delete(self.rect)
        self.set_image(self.cropped)
        self.photo_image = self.cropped_imagetk
        self.canvas.itemconfig(self.img_on_canvas, image=self.cropped_imagetk)


    def confirm_image(self):
        # self.deiconify()
        self.wm_protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)
        return self.image

# if __name__ == "__main__":
#     image_path = "D:\\UNI\\FinalProject\\data\\star_side\\6\\K16068.JPG"
#
#     image_cropper = ImageCropper(image_path, ctk.CTk)
#     image_cropper.image.show()
