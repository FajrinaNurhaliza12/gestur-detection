import keyboard
import os
import json


print("Key Listener aktif")
print("Tekan angka 1-0 untuk menampilkan angka.")
print("Tekan ENTER untuk reset.")
print("Tekan ESC untuk keluar.\n")


def save_data(value):
    data = {
        "number": value
    }

    with open("data.json", "w", encoding="utf-8") as file:
        json.dump(data, file)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def on_key(event):
    key = event.name

    # Tombol angka 1-0
    if key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:

        # Simpan angka ke JSON
        save_data(key)

        print(key)

    # Tombol ENTER = reset / blank
    elif key == "enter":

        # Kosongkan data
        save_data("")

        clear_screen()

        print("Key Listener aktif")
        print("Tekan angka 1-0 untuk menampilkan angka.")
        print("Tekan ENTER untuk reset.")
        print("Tekan ESC untuk keluar.\n")


# Pastikan data awal kosong
save_data("")


keyboard.on_press(on_key)

# Program tetap berjalan sampai ESC
keyboard.wait("esc")

print("\nProgram berhenti.")