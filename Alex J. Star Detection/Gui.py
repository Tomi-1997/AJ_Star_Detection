from tkinter import *
from tkinter import filedialog

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
        self.textlabel = Label(self.root, text='drop the file below', bg='#fcba03')
        self.main_frame_config()
        # menubar
        self.init_menu()

    def config_root(self):
        self.root.title('Alexander Jannaeus Coin Classifier')
        self.root.geometry('400x300')
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
        # Toplevel object which will
        # be treated as a new window
        help_win = Toplevel(self.root)

        # sets the title of the
        # Toplevel widget
        help_win.title("Help")

        # sets the geometry of toplevel
        help_win.geometry("400x300")

        # A Label widget to show in toplevel
        Label(help_win,
              text="Here's a guide how to use the program").pack()
        text = Text(help_win, wrap=WORD)
        text.insert(INSERT, "1. Choose a picture for inspection by- \n")
        text.insert(INSERT, "\t • Dragging it onto the window \n \t • By file > open\n")
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

    def drop_image(self, event):
        self.testvariable.set(event.data)
        self.show_image()

    def show_image(self):
        # get the value from string variable
        self.root.file_name = self.testvariable.get()

        image_path = Image.open(str(self.root.file_name))
        # resize image
        reside_image = image_path.resize((200, 200), Image.ANTIALIAS)
        # displays an image
        self.root.image = ImageTk.PhotoImage(reside_image)
        image_label = Label(self.labelframe, image=self.root.image).pack()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = CoinApp()
    app.run()
