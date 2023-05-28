import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk

import tkinterDnD as tkD
import tkinterdnd2
from Cons import MODELS_PATH
from PIL import ImageTk, Image
from Predictor import pred_conf


# https://www.pythontutorial.net/tkinter/tkinter-object-oriented-frame/


class InputFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        # self.config(bg='gold')
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=2)
        self.master = master
        self.__create_widgets()

    def __create_widgets(self):
        self.testvariable = StringVar()
        self.labelframe = LabelFrame(self, bg='white', height=200)
        self.entrybox = Entry(self, bg='gold', textvar=self.testvariable, width=80)
        self.textlabel = Label(self, bg='gold', text='drop the file below')
        self.textlabel.grid(column=0, row=0)
        self.entrybox.grid(column=1, row=0, sticky=tk.W)
        self.labelframe.grid(column=0, row=1, padx=20, pady=20, columnspan=2, sticky=tk.NSEW)
        self.__dnd_config()

    def __dnd_config(self):
        self.entrybox.drop_target_register(tkinterdnd2.DND_FILES)
        self.entrybox.dnd_bind('<<Drop>>', self.drop_image)
        self.labelframe.drop_target_register(tkinterdnd2.DND_FILES)
        self.labelframe.dnd_bind('<<Drop>>', self.drop_image)

    def drop_image(self, event):
        self.testvariable.set(event.data)
        self.show_image()

    def show_image(self, filepath=""):
        if filepath:
            self.master.file_name = filepath
        else:
            # get the value from string variable
            self.master.file_name = self.testvariable.get()

        image_path = Image.open(str(self.master.file_name))
        # resize image
        reside_image = image_path.resize((200, 200))
        # displays an image
        self.master.image = ImageTk.PhotoImage(reside_image)
        image_label = Label(self.labelframe, image=self.master.image)
        image_label.pack(anchor='center')


class PredictFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=2)
        self.master = master
        self.loaded_models = []
        self.__create_widgets()
        self.__init_models()

    def __create_widgets(self):
        self.predict_variable = StringVar()
        self.predition_label = Label(self, text=self.predict_variable, bg='gold')
        self.predict_btn = Button(self, text="Classify", command=self.predict)
        self.predict_btn.config(state=DISABLED)
        self.predict_btn.grid(column=0, row=0, sticky=tk.W)
        self.predition_label.grid(column=1, row=0, sticky=tk.W)

        self.md_progress = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=210
        )
        self.md_progress.grid()

    def __init_models(self):
        import keras
        self.predict_variable.set("Loading Models...")
        path = MODELS_PATH+"\\H05\\"
        md_list =  os.listdir(path)
        for md in md_list:
            self.loaded_models.append(keras.models.load_model(path + str(md)))
            progress = len(self.loaded_models) / len(md_list)
            self.md_progress['value'] = progress * 210

        if (self.master.image):
            self.predict_btn.config(state=NORMAL)

    def predict(self):
        image_path = str(self.master.file_name)
        guess, conf = pred_conf(image_path, self.loaded_models)
        res = f'Label - {guess}, Confidence = {conf * 100:.2f}%'
        self.predict_variable.set(res)


class CoinApp(tkD.Tk):
    def __init__(self):
        super().__init__()

        self.title('Alexander Jannaeus Coin Classifier')
        self.geometry('700x600')
        self.config(bg='gold')

        self.rowconfigure(0, weight=4)
        self.rowconfigure(1, weight=1)
        # main frame

        # self.main_frame_config()

        # menubar
        self.init_menu()
        self.__create_widgets()

    def __create_widgets(self):
        self.init_menu()
        self.inputframe = InputFrame(self)
        self.inputframe.grid(column=0, row=0, columnspan=2, sticky=tk.NSEW)
        self.classifier = PredictFrame(self)
        self.classifier.grid(column=0, row=1, columnspan=2, sticky=tk.NSEW)

    def init_menu(self):
        menubar = Menu(self)
        self.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open', command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.quit)
        menubar.add_cascade(label='Help', command=self.help_window)

    def open_file(self):
        file_path = filedialog.askopenfilename(title='Select Image File',
                                               filetypes=[('Image Files', '*.png *.jpg *.jpeg')])
        if file_path:
            # self.testvariable.set(file_path)
            self.inputframe.show_image(file_path)

    def help_window(self):
        help_win = Toplevel(self.root)

        help_win.title("Help")
        help_win.geometry("500x300")
        Label(help_win,
              text="Here's a guide how to use the program").pack()
        text = Text(help_win, wrap=WORD)
        text.insert(INSERT, "1. Choose a picture for inspection by one of the options- \n")
        text.insert(INSERT, "\t • Dragging it onto the window \n \t • file > open\n")
        text.insert(INSERT, "2. Edit the picture by cropping the star side if needed\n")
        text.insert(INSERT, "3. Press the classify button \n")
        text.insert(INSERT, "4. A label will appear with the approximate probability\n")
        text.config(state=DISABLED)
        text.pack()

    def run(self):
        self.mainloop()


if __name__ == '__main__':
    app = CoinApp()
    app.run()
