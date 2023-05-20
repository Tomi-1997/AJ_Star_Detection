from tkinter import *
import tkinterDnD as tkD
import tkinterdnd2
from PIL import ImageTk, Image


def DropImage(event):
    testvariable.set(event.data)
    # get the value from string variable
    window.file_name = testvariable.get()
    # takes path using dragged file
    image_path = Image.open(str(window.file_name))
    # resize image
    reside_image = image_path.resize((200, 200), Image.ANTIALIAS)
    # displays an image
    window.image = ImageTk.PhotoImage(reside_image)
    image_label = Label(labelframe, image=window.image).pack()


window = tkD.Tk()
window.title('Alexander Jannaeus Coin Classifier')
window.geometry('400x300')
window.config(bg='gold')

testvariable = StringVar()
textlabel = Label(window, text='drop the file below', bg='#fcba03')
textlabel.pack(anchor=NW, padx=10)
entrybox = Entry(window, textvar=testvariable, width=80)
entrybox.pack(fill=X, padx=10)
entrybox.drop_target_register(tkinterdnd2.DND_FILES)
entrybox.dnd_bind('<<Drop>>', DropImage)

labelframe = LabelFrame(window, bg='gold')

labelframe.pack(fill=BOTH, expand=True, padx=9, pady=9)

window.mainloop()
