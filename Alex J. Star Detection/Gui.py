from tkinter import *
from tkinter import filedialog

# from Predictor import multi_predict
import tkinterDnD as tkD
import tkinterdnd2
from PIL import ImageTk, Image


class CoinApp:
    def __init__(self):
        self.root = tkD.Tk()
        self.config_root()
        # main frame
        self.testvariable = StringVar()
        self.labelframe = LabelFrame(self.root, bg='gold')
        self.entrybox = Entry(self.root, textvar=self.testvariable, width=80)
        self.textlabel = Label(self.root, text='drop the file below')
        self.predition_label = Label(self.root, text='Label: ')
        self.predict_btn = Button(self.root, text="predict", command=self.predict)
        self.main_frame_config()
        # menubar
        self.init_menu()

    def config_root(self):
        self.root.title('Alexander Jannaeus Coin Classifier')
        self.root.geometry('700x600')
        self.root.config(bg='gold')

    def init_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open', command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.root.quit)
        menubar.add_cascade(label='Help', command=self.help_window)

    def open_file(self):
        file_path = filedialog.askopenfilename(title='Select Image File',
                                               filetypes=[('Image Files', '*.png *.jpg *.jpeg')])
        if file_path:
            self.testvariable.set(file_path)
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

    def main_frame_config(self):
        self.textlabel.pack(anchor=NW, padx=10)
        self.entrybox.pack(fill=X, padx=10)
        self.entrybox.drop_target_register(tkinterdnd2.DND_FILES)
        self.entrybox.dnd_bind('<<Drop>>', self.drop_image)
        self.labelframe.drop_target_register(tkinterdnd2.DND_FILES)
        self.labelframe.dnd_bind('<<Drop>>', self.drop_image)
        self.labelframe.pack(fill=BOTH, expand=True, padx=9, pady=9)
        self.predict_btn.config(state=DISABLED)
        self.predict_btn.pack()
        self.predition_label.pack()

    def predict(self):
        image_path = str(self.root.file_name)
        res = "ass"
        self.predition_label.config(text="Label: " + res)

    def drop_image(self, event):
        self.testvariable.set(event.data)
        self.show_image()

    def show_image(self):
        # get the value from string variable
        self.root.file_name = self.testvariable.get()

        image_path = Image.open(str(self.root.file_name))
        # resize image
        # reside_image = image_path.resize((200, 200), Image.ANTIALIAS)
        # displays an image
        self.root.image = ImageTk.PhotoImage(image_path)
        image_label = Label(self.labelframe, image=self.root.image).pack()
        self.predict_btn.config(state=NORMAL)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = CoinApp()
    app.run()
