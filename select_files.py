import tkinter as tk
from tkinter import filedialog

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    with open("selected_files.txt", "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")

if __name__ == "__main__":
    select_files()
