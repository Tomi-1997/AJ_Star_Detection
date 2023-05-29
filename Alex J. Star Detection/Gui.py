import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from Cons import tf

import tkinterDnD as tkD
import tkinterdnd2
from Cons import MODELS_PATH
from PIL import ImageTk, Image
from Predictor import pred_conf

import threading

# https://www.pythontutorial.net/tkinter/tkinter-object-oriented-frame/

class CoinApp(tkD.Tk):
    def __init__(self):
        super().__init__()

        self.title('Alexander Jannaeus Coin Classifier')
        self.geometry('700x600')
        self.config(bg='gold')

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=5)
        self.rowconfigure(2, weight=1)

        self.__init_widgets()

    def __init_widgets(self):
        self.loaded_models = []
        self.__init_menu()
        self.__init_input_frame()
        self.__init_classifier_frame()

    def __init_input_frame(self):
        input_frame = tk.Frame(self, bg='gold')
        input_frame.grid(column=0, row=0, sticky=NE)
        self.entry_var = StringVar()

        title_label = Label(input_frame, bg='gold', text='drop the file below')
        filename = Entry(input_frame, bg='gold', textvar=self.entry_var, width=80, state=DISABLED)
        self.picture_frame = Frame(input_frame, bg='white', height=200)

        self.picture_frame.pack(anchor=S, side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        title_label.pack(anchor=N, side=tk.LEFT, padx=10)
        filename.pack(anchor=N, side=tk.LEFT, padx=10)

        filename.drop_target_register(tkinterdnd2.DND_FILES)
        filename.dnd_bind('<<Drop>>', self.drop_image)
        self.picture_frame.drop_target_register(tkinterdnd2.DND_FILES)
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
        # resize image
        reside_image = image_path.resize((200, 200))
        # displays an image
        image = ImageTk.PhotoImage(reside_image)
        image_label = Label(self.picture_frame, image=image, bg='white')
        image_label.image = image
        image_label.pack(anchor='center')

    def __init_classifier_frame(self):
        classifier_frame = tk.Frame(self, bg='red')
        classifier_frame.grid(column=0, row=1, sticky=S)

        self.button_var = StringVar(value="Load Models")
        self.classifier_button = Button(classifier_frame, textvar=self.button_var, command=self.__init_models)

        self.label_var = StringVar(value="<- please click here to start")
        classifier_label = Label(classifier_frame, textvar=self.label_var, bg='gold')

        self.classifier_button.pack(anchor=N, side=tk.LEFT, padx=10)
        classifier_label.pack(anchor=N, side=tk.LEFT, padx=10)

        self.md_progress = ttk.Progressbar(
            classifier_frame,
            orient='horizontal',
            mode='determinate',
            length=240
        )
        self.md_progress.pack()

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
        total = len(md_list)
        loaded_models = []
        for md in md_list:
            loaded_models.append(keras.models.load_model(path + str(md)))
            self.md_progress['value'] += (1/total)*100
        # Update the loaded models in the main GUI thread
        self.update_loaded_models(loaded_models)

    def update_loaded_models(self, loaded_models):
        self.loaded_models = loaded_models
        # Update the GUI elements to reflect the loaded state

        self.button_var.set("Classify")
        self.classifier_button.config(state=NORMAL, command=self.__predict)
        self.label_var.set("<- press to classify")

        # self.md_progress.grid_forget()

    def __predict(self):
        image_path = self.entry_var.get()
        guess, conf = pred_conf(image_path, self.loaded_models)
        res = f'Label - {guess}, Confidence = {conf * 100:.2f}%'
        self.label_var.set(res)

    def __reset_prediction(self, *args):
        self.label_var.set("<- press to classify")
        if self.entry_var.get() != "" and self.classifier_button['state'] == tk.DISABLED:
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


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    app = CoinApp()
    app.mainloop()
