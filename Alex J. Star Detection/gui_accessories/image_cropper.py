import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

ctk.set_default_color_theme("green")


class ImageCropper:
    def __init__(self, image_path, root):
        self.root = ctk.CTkToplevel(root)
        # self.root = ctk.CTk()
        self.root.title("Image Cropper")
        self.root.focus()
        fg_color = self.root.cget("fg_color")[1]
        self.canvas = tk.Canvas(self.root, width=500, height=500, bg=fg_color)
        self.canvas.pack()

        self.image = Image.open(image_path)
        self.image.thumbnail((500, 500))
        self.photo_image = ImageTk.PhotoImage(self.image)
        # width = self.canvas.winfo_width()
        # height = self.canvas.winfo_height()
        self.img_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        self.crop_button = ctk.CTkButton(self.root, text="Crop", command=self.crop_image)
        self.crop_button.pack()
        self.confirm_button = ctk.CTkButton(self.root, text="Confirm", command=self.confirm_image)
        self.confirm_button.pack()

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
        self.cropped = self.image.crop((self.start_x, self.start_y, cur_x, cur_y))
        self.cropped_imagetk = ImageTk.PhotoImage(self.cropped)
        # self.canvas.itemconfig(self.photo_image, image=self.cropped_image)
        # self.canvas.create_image(250, 250, anchor=tk.CENTER, image=cropped_image)

    def crop_image(self):
        self.canvas.delete(self.rect)
        self.canvas.itemconfig(self.img_on_canvas, image=self.cropped_imagetk)
        self.image = self.cropped
        # Disable the crop button
        # self.crop_button.configure(state=tk.DISABLED)

    def confirm_image(self):
        self.root.withdraw()

# if __name__ == "__main__":
#     image_path = "D:\\UNI\\FinalProject\\data\\star_side\\6\\K16068.JPG"
#
#     image_cropper = ImageCropper(image_path, ctk.CTk)
#     image_cropper.image.show()
