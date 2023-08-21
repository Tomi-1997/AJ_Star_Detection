import os.path

import customtkinter as ctk
from PIL import Image


# class MyTabView(customtkinter.CTkTabview):
#     def __init__(self, master, **kwargs):
#         super().__init__(master, **kwargs)
def write_text(root, name_file):
    text = ctk.CTkTextbox(root, wrap="word", height=50)
    with open(name_file) as f:
        for line in f.readlines():
            text.insert("end", line)
    text.configure(state="disabled")
    return text


class Guide(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        self.title("Help")
        self.geometry("400x350")
        self.focus()
        self.grid_rowconfigure(0, weight=5)
        self.grid_rowconfigure(1, weight=2)
        self.grid_columnconfigure((0, 1), weight=1)

        # create tabs
        self.tabView = ctk.CTkTabview(self)
        self.general_info = self.tabView.add("General")
        self.crop_info = self.tabView.add("Crop")

        # tab content
        self.general_win = self.general_help_window()
        self.crop_win = self.crop_help_window()
        self.crop_win.pack(fill=ctk.BOTH, expand=True)
        self.general_win.pack(fill=ctk.BOTH, expand=True)

        self.tabView.grid(column=0, row=0, columnspan=2, padx=10, pady=(5, 0), sticky="nsew")

        # example pic
        file_name = "gui_accessories/cropped_eg.jpg"
        image_path = Image.open(file_name)
        image = ctk.CTkImage(light_image=image_path, size=(100, 100))
        image_label = ctk.CTkLabel(self, image=image, text="")
        image_label.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="ew")
        eg_label = ctk.CTkLabel(self, text="cropped example")
        eg_label.grid(row=2, column=0, padx=10, pady=(5, 0), sticky="e")

    def general_help_window(self):
        help_win = ctk.CTkFrame(master=self.general_info)
        help_win.grid_rowconfigure(0, weight=1)
        help_win.grid_rowconfigure(1, weight=4)
        help_win.grid_columnconfigure((0, 1), weight=1)

        help_title = ctk.CTkLabel(help_win, text="Here's a guide how to use the program")
        help_title.grid(column=0, row=0, padx=10, pady=(5, 0), sticky="nsew")

        text = write_text(help_win, "gui_accessories\\general_help.txt")
        text.grid(row=1, column=0, padx=10, columnspan=2, pady=(5, 0), sticky="nsew")

        return help_win

    def crop_help_window(self):
        help_win = ctk.CTkFrame(master=self.crop_info)
        help_win.grid_rowconfigure(0, weight=1)
        help_win.grid_rowconfigure(1, weight=4)
        help_win.grid_columnconfigure((0, 1), weight=1)

        help_title = ctk.CTkLabel(help_win, text="Here's a guide how to use the crop")
        help_title.grid(column=0, row=0, columnspan=2, padx=10, pady=(5, 0), sticky="nsew")

        text = write_text(help_win, "gui_accessories\\crop_help.txt")
        text.grid(row=1, column=0, padx=10, columnspan=2, pady=(5, 0), sticky="nsew")

        return help_win
