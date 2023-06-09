import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import ttk
from Cons import tf

import tkinter.dnd
import customtkinter as ctk
# import tkinterDnD
from tkinterdnd2 import TkinterDnD, DND_FILES
from Cons import MODELS_PATH
from PIL import ImageTk, Image
from Predictor import pred_conf

import threading


class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


ctk.set_default_color_theme("green")


class CoinApp(Tk):
    def __init__(self):
        super().__init__()
        self.title('Alexander Jannaeus Coin Classifier')
        self.geometry('700x600')
        # self.config(bg='gold')

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.__init_widgets()

    def __init_widgets(self):
        self.loaded_models = []
        self.__init_menu()
        self.__init_input_frame()
        self.__init_classifier_frame()
        self.__init_models()

    def __init_input_frame(self):
        input_frame = ctk.CTkFrame(master=self)
        input_frame.grid(column=0, row=0, columnspan=2, padx=20, pady=(20, 0), sticky="nsew")
        input_frame.grid_rowconfigure(0, weight=1)
        input_frame.grid_rowconfigure(1, weight=5)
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=3)
        self.entry_var = StringVar()

        title_label = ctk.CTkLabel(input_frame, text='drop the file below')
        filename = ctk.CTkEntry(input_frame, textvariable=self.entry_var, width=80, state="disabled")
        self.picture_frame = ctk.CTkFrame(input_frame, height=200)

        self.picture_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")
        title_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        filename.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        filename.drop_target_register(DND_FILES)
        filename.dnd_bind('<<Drop>>', self.drop_image)
        self.picture_frame.drop_target_register(DND_FILES)
        self.picture_frame.dnd_bind('<<Drop>>', self.drop_image)

        self.entry_var.trace('w', self.__reset_prediction)

    def drop_image(self, event):
        self.clear_image()
        self.entry_var.set(event.data)
        self.show_image()

    def clear_image(self):
        for pic in self.picture_frame.winfo_children():
            pic.destroy()

    def show_image(self):
        file_name = self.entry_var.get()

        image_path = Image.open(str(file_name))
        image = ctk.CTkImage(light_image=image_path, size=(200, 200))
        image_label = ctk.CTkLabel(self.picture_frame, image=image, text="")
        # image_label.image = image
        image_label.pack(anchor='center')

    def __init_classifier_frame(self):
        classifier_frame = ctk.CTkFrame(self)
        classifier_frame.grid(column=0, row=1, columnspan=2, padx=20, pady=20, sticky="nsew")
        classifier_frame.grid_rowconfigure(0, weight=1)
        classifier_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.button_var = StringVar()
        self.classifier_button = ctk.CTkButton(classifier_frame, textvariable=self.button_var, command=self.__predict)

        self.label_var = StringVar(value="<- please click here to start")
        classifier_label = ctk.CTkLabel(classifier_frame, textvariable=self.label_var)

        self.classifier_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        classifier_label.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.md_progress = ctk.CTkProgressBar(
            classifier_frame,
            orientation='horizontal',
            mode='determinate',
            width=240
        )
        self.md_progress.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

    def __init_models(self):
        self.label_var.set("Loading Models...")
        self.button_var.set("Loading...")

        # Start a new thread to load the models
        load_thread = threading.Thread(target=self.load_models_thread)
        load_thread.start()

    def load_models_thread(self):
        import keras
        path = MODELS_PATH
        md_list = os.listdir(path)
        loaded_models = []
        for md in md_list:
            loaded_models.append(keras.models.load_model(path + str(md)))
            self.md_progress.step()
        # Update the loaded models in the main GUI thread
        self.update_loaded_models(loaded_models)

    def update_loaded_models(self, loaded_models):
        self.loaded_models = loaded_models
        # Update the GUI elements to reflect the loaded state

        self.button_var.set("Classify")
        self.classifier_button.configure(state=NORMAL, command=self.__predict)
        self.label_var.set("<- press to classify")

        self.md_progress.configure(mode="indeterminate")
        # self.md_progress.grid_forget()

    def __predict(self):
        if len(self.loaded_models) < 24:
            messagebox.showerror('Error', 'The models need to finish to load')
            return
        if not self.entry_var.get():
            messagebox.showerror('Error', 'No picture has been uploaded')
            return
        pred_thread = threading.Thread(target=self.get_pred)
        pred_thread.start()

    def get_pred(self):
        self.md_progress.start()
        image_path = self.entry_var.get()
        guess, conf = pred_conf(image_path, self.loaded_models)
        res = f'{guess} rayed star, Confidence = {conf * 100:.2f}%'
        self.label_var.set(res)
        self.md_progress.stop()

    def __reset_prediction(self, *args):
        self.label_var.set("<- press to classify")
        if self.entry_var.get() != "" and self.classifier_button.cget("state") == "disabled":
            self.classifier_button.config(state=NORMAL)

    def __init_menu(self):
        menubar = Menu(self)
        self.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open', command=self.__open_file)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.quit)
        menubar.add_cascade(label='Help', command=self.help_window)

    def __open_file(self):
        file_path = filedialog.askopenfilename(title='Select Image File',
                                               filetypes=[('Image Files', '*.png *.jpg *.jpeg')])
        if file_path:
            self.clear_image()
            self.entry_var.set(file_path)
            self.show_image()

    def help_window(self):
        help_win = ctk.CTkToplevel(self)

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

        text = ctk.CTkTextbox(help_win, wrap=WORD, height=50)
        text.insert("0.0", "1. Choose a picture for inspection by one of the options- \n")
        text.insert("end", "\t • Dragging it onto the window \n \t • file > open\n")
        text.insert("end", "2. Edit the picture by cropping the star side like below if needed\n")
        text.insert("end", "3. Press the classify button \n")
        text.insert("end", "4. A label will appear with the approximate probability\n")
        text.configure(state="disabled")
        text.grid(row=1, column=0, padx=10, columnspan=2, pady=(5, 0), sticky="nsew")

        file_name = "cropped_eg.jpg"
        image_path = Image.open(file_name)
        image = ctk.CTkImage(light_image=image_path, size=(100, 100))
        image_label = ctk.CTkLabel(help_win, image=image, text="")
        image_label.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="ew")
        eg_label = ctk.CTkLabel(help_win, text="cropped example")
        eg_label.grid(row=2, column=0, padx=10, pady=(5, 0), sticky="e")


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    app = CoinApp()
    app.mainloop()
