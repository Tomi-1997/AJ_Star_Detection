import customtkinter as ctk
from PIL import Image


class Guide:
    def __init__(self, root):
        self.root = root

    def help_window(self):
        help_win = ctk.CTkToplevel(self.root)

        help_win.title("Help")
        help_win.geometry("400x350")
        help_win.focus()
        help_win.grid_rowconfigure(0, weight=1)
        help_win.grid_rowconfigure(1, weight=4)
        help_win.grid_rowconfigure(2, weight=2)
        # help_win.grid_rowconfigure((0, 1, 2), weight=1)
        help_win.grid_columnconfigure((0, 1), weight=1)

        help_title = ctk.CTkLabel(help_win, text="Here's a guide how to use the program")
        help_title.grid(column=0, row=0, columnspan=2, padx=10, pady=(5, 0), sticky="nsew")

        text = ctk.CTkTextbox(help_win, wrap="word", height=50)
        text.insert("0.0", "1. Choose a picture for inspection by one of the options- \n")
        text.insert("end", "\t • Dragging it onto the window \n \t • file > open\n")
        text.insert("end", "2. Edit the picture by cropping the star side like below if needed\n")
        text.insert("end", "3. Press the classify button \n")
        text.insert("end", "4. A label will appear with the approximate probability\n")
        text.configure(state="disabled")
        text.grid(row=1, column=0, padx=10, columnspan=2, pady=(5, 0), sticky="nsew")

        file_name = "gui_accessories/cropped_eg.jpg"
        image_path = Image.open(file_name)
        image = ctk.CTkImage(light_image=image_path, size=(100, 100))
        image_label = ctk.CTkLabel(help_win, image=image, text="")
        image_label.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="ew")
        eg_label = ctk.CTkLabel(help_win, text="cropped example")
        eg_label.grid(row=2, column=0, padx=10, pady=(5, 0), sticky="e")
