import math
import os

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QColor, QImage, QIntValidator, QPainter, QPolygon
from PySide6.QtWidgets import (QApplication, QFileDialog, QMainWindow,
                               QMessageBox, QDialog, QVBoxLayout, QLineEdit, QPushButton, QLabel, QHBoxLayout)
from ui import Ui_MainWindow
import numpy as np
from PIL import Image
import json
from random import randint
import hashlib


class CoordinatesInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ведіть розміри:")

        layout = QVBoxLayout(self)

        # Create a QHBoxLayout to hold the "x =" label and its input field
        layout_x = QHBoxLayout()

        # Create QLabel to display "x = "
        label_x = QLabel("x = ")

        # Create QLineEdit for inputting x value
        self.x_input = QLineEdit()
        self.x_input.setValidator(QIntValidator())

        # Add the "x =" label and its input field to the QHBoxLayout
        layout_x.addWidget(label_x)
        layout_x.addWidget(self.x_input)

        # Add the QHBoxLayout to the overall QVBoxLayout
        layout.addLayout(layout_x)

        # Create a QHBoxLayout to hold the "y =" label and its input field
        layout_y = QHBoxLayout()

        # Create QLabel to display "y = "
        label_y = QLabel("y = ")

        # Create QLineEdit for inputting y value
        self.y_input = QLineEdit()
        self.y_input.setValidator(QIntValidator())

        # Add the "y =" label and its input field to the QHBoxLayout
        layout_y.addWidget(label_y)
        layout_y.addWidget(self.y_input)

        # Add the QHBoxLayout to the overall QVBoxLayout
        layout.addLayout(layout_y)

        self.confirm_button = QPushButton("Застосувати")
        layout.addWidget(self.confirm_button)

        self.confirm_button.clicked.connect(self.on_confirm_clicked)

    def on_confirm_clicked(self):
        # Retrieve the values from the input fields
        x_value = self.x_input.text()
        y_value = self.y_input.text()

        if 100 > int(x_value) or int(x_value) > 1000:
            QMessageBox.information(self, "От халепа", f"X має бути в проміжку від 100 до 1000")
            return
        if 100 > int(y_value) or int(y_value) > 1000:
            QMessageBox.information(self, "От халепа", f"Y має бути в проміжку від 100 до 1000")
            return
        # Assign the values to x and y variables in the main window
        self.parent().y = x_value
        self.parent().x = y_value

        # Close the dialog
        self.accept()


class RegistrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.FILE_NAME = "password.txt"
        self.setWindowTitle("Вхід")

        layout = QVBoxLayout(self)

        # Create QLabel to display "Username: "
        label_username = QLabel("Ім'я користувача:")
        self.username_input = QLineEdit()

        # Create QLabel to display "Password: "
        label_password = QLabel("Пароль:")
        self.password_input = QLineEdit()
        # self.password_input.setEchoMode(QLineEdit.Password)

        layout.addWidget(label_username)
        layout.addWidget(self.username_input)
        layout.addWidget(label_password)
        layout.addWidget(self.password_input)

        # Create QPushButton for registration confirmation
        self.confirm_button = QPushButton("Підтвердити")
        layout.addWidget(self.confirm_button)

        self.confirm_button.clicked.connect(self.on_confirm_clicked)

    def on_confirm_clicked(self):
        if_exist = self.check_file()
        if if_exist:
            self.accept()  # Close the registration dialog
        else:
            QMessageBox.warning(self, "От халепа", "Неправильний пароль. Спробуйте ще раз.")

    def check_file(self) -> bool:
        info = None
        if os.path.exists(self.FILE_NAME):
            try:
                with open(self.FILE_NAME, 'r+') as file:
                    info = file.read()
                    if not info:
                        info = {
                            "passwords": [],
                            "usernames": []
                        }
                    else:
                        info = json.loads(info)
            except FileNotFoundError:
                pass

            with open(self.FILE_NAME, 'w') as file:
                if self.username_input.text() in info["usernames"]:
                    if hashlib.sha256(self.password_input.text().encode("utf-8")).hexdigest() == info["passwords"][
                        info["usernames"].index(self.username_input.text())]:
                        info = json.dumps(info)
                        file.write(info)
                        return True
                    info = json.dumps(info)
                    file.write(info)
                    return False
                else:
                    info["usernames"].append(self.username_input.text())
                    info["passwords"].append(hashlib.sha256(self.password_input.text().encode("utf-8")).hexdigest())
                    info = json.dumps(info)
                    file.write(info)
                    return True
        # If the file doesn't exist or if there's an error reading it, create it
        with open(self.FILE_NAME, 'w+') as file:
            info = {
                "passwords": [hashlib.sha256(self.password_input.text().encode("utf-8")).hexdigest()],
                "usernames": [self.username_input.text()]
            }
            # print(info)
            info = json.dumps(info)
            file.write(info)
            return True


class InfoWindow(QDialog):
    def __init__(self, info_text, title_text,parent=None):
        super().__init__(parent)

        self.setWindowTitle(title_text)
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout(self)

        # Create QLabel to display the information text
        self.info_label = QLabel(info_text)
        layout.addWidget(self.info_label)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.registration_dialog = RegistrationDialog()
        if self.registration_dialog.exec() == QDialog.Rejected:
            self.close()
            return

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.FILE_NAME = "info.txt"
        self.username = self.registration_dialog.username_input.text()
        self.info = self.get_info()

        self.load_recent_encrypted_text()
        self.load_recent_decrypted_text()

        self.ui.radioButton_color_1.setText("Монохром")
        self.ui.radioButton_color_2.setText("Рандом")
        self.ui.radioButton_color_3.setText("Градієнт")

        self.ui.radioButton_bitmap_1.setText("Квадратики")
        self.ui.radioButton_bitmap_2.setText("Трикутник Серпінського")
        self.ui.radioButton_bitmap_3.setText("Хвильки")

        self.ui.btn_download_image.clicked.connect(self.on_btn_download_image_clicked)
        self.ui.btn_encrypt_text.clicked.connect(self.on_btn_encrypt_text_clicked)
        self.ui.btn_decrypt_text_2.clicked.connect(self.on_btn_decrypt_text_clicked)
        self.ui.btn_create_new_image.clicked.connect(self.open_coordinates_dialog)
        self.ui.btn_save_image.clicked.connect(self.save_image)
        self.ui.btn_apply_changes.clicked.connect(self.get_radio_btn_val)
        self.ui.btn_exit.clicked.connect(self.account_log_out)
        self.ui.btn_recent_image_1.clicked.connect(self.recent_img_1)
        self.ui.btn_recent_image_2.clicked.connect(self.recent_img_2)
        self.ui.btn_recent_image_3.clicked.connect(self.recent_img_3)
        self.ui.btn_about_creators.clicked.connect(self.open_about_creators)
        self.ui.btn_about_programme.clicked.connect(self.open_about_programme)
        self.ui.btn_user_instruction.clicked.connect(self.open_user_instruction)
        self.ui.btn_revert_changes.clicked.connect(self.revert_changes)

        self.image = None
        self.image_copy = None
        self.image_width = None
        self.image_height = None

        self.text_to_encrypt = ""

        self.symbol_end = "000111000111000111"
        self.text_end = "000001111100000111110000011111"

        self.x = None
        self.y = None

        self.chosen_bitmap = None
        self.chosen_color = None

    def on_btn_download_image_clicked(self):
        # Open file dialog with BMP filter (case-insensitive)
        file, _ = QFileDialog.getOpenFileName(self, "Виберіть файл", "", "*.bmp;*.BMP")

        if file:
            # Check if the selected file is a BMP image
            if not file.lower().endswith((".bmp")):
                error_msg = "Виберіть bmp файл."
                QMessageBox.warning(self, "От халепа", error_msg)
                return

            # Load the image
            self.open_image_in_label(file)

        else:
            # Inform the user if no file was selected
            QMessageBox.information(self, "От халепа", "Виберіть вайл.")

    def open_image_in_label(self, file):
        pixmap = QPixmap(file)
        self.image = pixmap.toImage()
        self.image_copy = self.image.copy()
        self.image_width = self.image.width()
        self.image_height = self.image.height()
        # Resize the image to fit the image widget while maintaining aspect ratio
        self.ui.image.setPixmap(pixmap.scaled(self.ui.image.width(),
                                              self.ui.image.height(),
                                              Qt.KeepAspectRatio))

    def on_btn_encrypt_text_clicked(self):
        if not self.image:
            QMessageBox.warning(self, "От халепа", "Немає фото для шифрування.")
            return
        # Get the text from plainTextEdit_encrypt_text
        self.text_to_encrypt = self.ui.plainTextEdit_encrypt_text.toPlainText()
        self.update_info(self.text_to_encrypt, "encode_text")
        self.load_recent_encrypted_text()
        # Check if the text is empty
        if not self.text_to_encrypt:
            QMessageBox.warning(self, "От халепа", "Введіть текст для шифрування.")
        else:
            QMessageBox.warning(self, "Успіх!", "Текст успішно зашифровано!")
            self.encrypt_text()
            # Do something with the text, for example, print it
            # print("Text to Encrypt:", self.text_to_encrypt)

    def on_btn_decrypt_text_clicked(self):
        if not self.image:
            QMessageBox.warning(self, "От халепа", "Немає фото для розшифрування.")
            return
        self.decrypt_text()

    def decrypt_text(self):
        img_rgb_array = self.convert_img_to_rgb_array()
        img_byte_array = self.convert_int_to_bite_array(img_rgb_array)
        last_bit_of_byte = self.convert_bytes_to_last_bits(img_byte_array)
        decrypted_text = self.get_text_from_bits(last_bit_of_byte)
        if decrypted_text == "":
            QMessageBox.warning(self, "От халепа", "Немає зашифрованих даних в фото.")
            return
        self.update_info(decrypted_text, "decode_text")
        self.load_recent_decrypted_text()
        self.ui.plainTextEdit_decrypted_text.setPlainText(decrypted_text)

    def encrypt_text(self):
        img_rgb_array = self.convert_img_to_rgb_array()
        img_byte_array = self.convert_int_to_bite_array(img_rgb_array)

        text_byte_array = self.convert_text_to_bite_array()
        # print(img_byte_array[0], img_byte_array[1], img_byte_array[2])
        index = 0
        for symbol in text_byte_array:
            for bit in symbol:
                a = list(img_byte_array[index])
                a[-1] = bit
                img_byte_array[index] = ''.join(a)
                index += 1

        a = self.byte_to_int_array(img_byte_array)
        new_img_byte_array = [a[i:i + 3] for i in range(0, len(a), 3)]
        new_img_byte_array = [new_img_byte_array[i:i + self.image_width] for i in
                              range(0, len(new_img_byte_array), self.image_width)]
        self.create_img_from_np_array(np.array(new_img_byte_array, dtype=np.uint8), False)

    def create_img_from_np_array(self, arr, copy_check):
        img = Image.fromarray(arr, 'RGB')
        img.save("test.bmp")

        pil_image = Image.open("test.bmp")
        qt_image = QPixmap.fromImage(
            QImage(pil_image.tobytes(), pil_image.size[0], pil_image.size[1], QImage.Format_RGB888))
        self.image = qt_image.toImage()
        if copy_check:
            self.image_copy = self.image.copy()
        self.image_width = self.image.width()
        self.image_height = self.image.height()

        self.ui.image.setPixmap(qt_image.scaled(self.ui.image.width(),
                                                self.ui.image.height(),
                                                Qt.KeepAspectRatio))


    def convert_img_to_rgb_array(self) -> list[list[int]]:
        rgb_values = []
        for i in range(self.image.height()):
            for j in range(self.image.width()):
                color = QColor(self.image.pixel(j, i))
                red = color.red()
                green = color.green()
                blue = color.blue()
                rgb_values.append([red, green, blue])
        return rgb_values

    def convert_int_to_bite_array(self, rgb_array) -> list[str]:
        bite_values = []
        for i in range(len(rgb_array)):
            for j in range(3):
                bite_values.append(f"{rgb_array[i][j]:08b}")
        return bite_values

    def convert_text_to_bite_array(self):
        a = ' '.join(format(ord(x), 'b') for x in self.text_to_encrypt)
        a = a.split()
        b = []
        for i in range(len(a) - 1):
            b.append(a[i])
            b.append(self.symbol_end)
        b.append(a[-1])
        b.append(self.text_end)
        return b

    def byte_to_int_array(self, arr):
        out = []
        for i in arr:
            out.append(int(i, 2))
        return out

    def convert_bytes_to_last_bits(self, arr):
        out = ""
        for i in arr:
            out += i[-1]
        return out

    def get_text_from_bits(self, bits):
        if not bits.find(self.text_end):
            return ""
        bits = bits[0:bits.find(self.text_end)]
        while self.symbol_end in bits:
            bits = bits[0:bits.find(self.symbol_end)] + ' ' + bits[bits.find(self.symbol_end) + len(self.symbol_end):]
        bits = bits.split()
        chars = [chr(int(i, 2)) for i in bits]
        return ''.join(chars)

    def open_coordinates_dialog(self):
        dialog = CoordinatesInputDialog(self)
        dialog.exec()

        # After the dialog is closed, check if x and y have been updated
        if self.x is not None and self.y is not None:
            # print("X:", self.x)
            # print("Y:", self.y)

            self.create_blank_img()

    def create_blank_img(self):
        rgb_img = np.full((int(self.x), int(self.y), 3), 255, dtype=np.uint8)
        self.create_img_from_np_array(rgb_img, True)

    def save_image(self):
        if self.image is None:
            QMessageBox.information(self, "От халепа", f"Немає фото для збереження.")
            return
        # Open a file dialog to get the filename and directory where the user wants to save the image
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "BMP Files (*.bmp);;All Files (*)")
        self.update_info(file_name, "images_url")
        if file_name:
            # Save the image
            self.image.save(file_name)
            QMessageBox.information(self, "Успіх", f"Фото збережено: {file_name}.")

    def get_radio_btn_val(self):
        # Монохромний: Всі пікселі одного кольору, наприклад, чорного або білого.
        # Градієнт: Кольори змінюються поступово від одного кінця малюнка до іншого.
        # Рандомний: Кольори генеруються випадковим чином для кожної точки малюнка.
        # Check which color radio button is selected

        if self.ui.radioButton_color_1.isChecked():
            self.chosen_color = "monochrome"
        elif self.ui.radioButton_color_2.isChecked():
            self.chosen_color = "random"
        elif self.ui.radioButton_color_3.isChecked():
            self.chosen_color = "gradient"
        else:
            # Handle the case when no color is selected
            self.chosen_color = None

        # Check which bitmap radio button is selected

        if self.ui.radioButton_bitmap_1.isChecked():
            self.chosen_bitmap = 1
        elif self.ui.radioButton_bitmap_2.isChecked():
            self.chosen_bitmap = 2
        elif self.ui.radioButton_bitmap_3.isChecked():
            self.chosen_bitmap = 3
        else:
            # Handle the case when no bitmap is selected
            self.chosen_bitmap = None

        if self.chosen_bitmap is None or self.chosen_color is None:
            QMessageBox.information(self, "От халепа", f"Виберіть колірну комбінацію і ботову карту.")
        elif self.image is None:
            QMessageBox.information(self, "От халепа", f"Створіть нову або відкрийте існуючу фотографію.")
        else:
            if self.chosen_bitmap == 1:
                self.draw_symbol(self.chosen_color)
            elif self.chosen_bitmap == 2:
                self.draw_sierpinski_triangle(self.chosen_color)
            else:
                self.draw_symbol_2(self.chosen_color)

    def draw_sierpinski_triangle(self, palette):
        # Get image dimensions
        img_data = self.convert_img_to_correct_rgb_array()
        width = len(img_data[0])
        height = len(img_data)

        # Starting triangle coordinates (center of the image)
        p1_x = width // 2
        p1_y = height // 2

        # Calculate coordinates for other two points of the initial triangle
        offset = width // 4
        p2_x = p1_x + offset
        p2_y = p1_y - offset
        p3_x = p1_x - offset
        p3_y = p1_y - offset

        # Recursive function to draw the triangle
        def sierpinski_triangle_recursive(x1, y1, x2, y2, x3, y3, depth):
            # Base case: stop recursion when depth reaches a limit
            if depth == 0:
                return

            # Calculate midpoints of each side
            mid_x1_2 = (x1 + x2) // 2
            mid_y1_2 = (y1 + y2) // 2
            mid_x2_3 = (x2 + x3) // 2
            mid_y2_3 = (y2 + y3) // 2
            mid_x1_3 = (x1 + x3) // 2
            mid_y1_3 = (y1 + y3) // 2

            # Draw the center triangle based on the chosen palette
            center_color = self.get_color(palette, mid_x1_2, mid_y1_2)

            # Update RGB values in the array for the center triangle pixels
            for x in range(max(0, mid_x1_2 - 1), min(width, mid_x1_2 + 2)):
                for y in range(max(0, mid_y1_2 - 1), min(height, mid_y1_2 + 2)):
                    if abs(x - mid_x1_2) + abs(
                            y - mid_y1_2) <= 1:  # Check if within triangle boundaries (adjust for anti-aliasing if needed)
                        img_data[y][x] = center_color

            # Recursively draw triangles on each sub-section
            sierpinski_triangle_recursive(x1, y1, mid_x1_2, mid_y1_2, mid_x1_3, mid_y1_3, depth - 1)
            sierpinski_triangle_recursive(mid_x1_2, mid_y1_2, x2, y2, mid_x2_3, mid_y2_3, depth - 1)
            sierpinski_triangle_recursive(mid_x2_3, mid_y2_3, x3, y3, mid_x1_3, mid_y1_3, depth - 1)

        # Call the recursive function with initial parameters and desired depth
        depth = 10  # Adjust this value to control the complexity of the triangle
        sierpinski_triangle_recursive(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, depth)

        # Update the image data with the modified RGB array
        self.create_img_from_np_array(np.array(img_data, dtype=np.uint8), False)

    def draw_symbol(self, palette):
        img_rgb_array = self.convert_img_to_correct_rgb_array()
        for i in range(self.image_height // 20):
            for j in range(self.image_width // 20):
                for n in range(10):
                    img_rgb_array[i * 20][j * 20 + n] = self.get_color(palette, i * 20, j * 20 + n)
                    img_rgb_array[i * 20 + 9][j * 20 + n] = self.get_color(palette, i * 20 + 9, j * 20 + n)
                    img_rgb_array[i * 20 + 18][j * 20 + n] = self.get_color(palette, i * 20 + 18, j * 20 + n)

                    img_rgb_array[i * 20 + n][j * 20] = self.get_color(palette, i * 20 + n, j * 20)
                    img_rgb_array[i * 20 + n][j * 20 + 9] = self.get_color(palette, i * 20 + n, j * 20 + 9)
                    img_rgb_array[i * 20 + n][j * 20 + 18] = self.get_color(palette, i * 20 + n, j * 20 + 18)
        self.create_img_from_np_array(np.array(img_rgb_array, dtype=np.uint8), False)

    def draw_symbol_2(self, palette):
        img_rgb_array = self.convert_img_to_correct_rgb_array()
        for i in range(self.image_height):
            for j in range(self.image_width):
                if i % 7 == 0 or i % 8 == 0 and i != 0:
                    if j % 7 == 0 or j % 9 == 0:
                        img_rgb_array[i - 1][j] = self.get_color(palette, i - 1, j)
                    elif j % 8 == 0:
                        img_rgb_array[i - 2][j] = self.get_color(palette, i - 2, j)
                    else:
                        img_rgb_array[i][j] = self.get_color(palette, i, j)
        self.create_img_from_np_array(np.array(img_rgb_array, dtype=np.uint8), False)

    def get_color(self, palette, x, y):
        if palette == "monochrome":
            return [0, 0, 0]
        elif palette == "random":
            return [randint(0, 255), randint(0, 255), randint(0, 255)]
        else:
            # Define gradient colors
            start_color = [255, 0, 0]  # Red
            end_color = [0, 0, 255]  # Blue

            # Calculate position-based RGB values
            distance = math.sqrt(x ** 2 + y ** 2)  # Euclidean distance from upper-left corner
            max_distance = math.sqrt(self.image_width ** 2 + self.image_height ** 2)
            progress = distance / max_distance

            r = int(start_color[0] * (1 - progress) + end_color[0] * progress)
            g = int(start_color[1] * (1 - progress) + end_color[1] * progress)
            b = int(start_color[2] * (1 - progress) + end_color[2] * progress)

            return [r, g, b]

    def draw_fractal(self):
        img_rgb_array = self.convert_img_to_correct_rgb_array()
        # paste code here
        for i in range(len(img_rgb_array)):
            for j in range(len(img_rgb_array[i])):
                # Calculate fractal value based on pixel coordinates
                fractal_value = self.calculate_fractal(i, j)

                # Modify RGB values based on fractal value
                img_rgb_array[i][j][0] = fractal_value  # Red component
                img_rgb_array[i][j][1] = fractal_value  # Green component
                img_rgb_array[i][j][2] = fractal_value  # Blue component

        # end
        self.create_img_from_np_array(np.array(img_rgb_array, dtype=np.uint8), False)

    def convert_img_to_correct_rgb_array(self) -> list[list[list[int]]]:
        rgb_values = []
        for i in range(self.image.height()):
            a = []
            for j in range(self.image.width()):
                color = QColor(self.image.pixel(j, i))
                red = color.red()
                green = color.green()
                blue = color.blue()
                a.append([red, green, blue])
            rgb_values.append(a)
        return rgb_values

    def calculate_fractal(self, x, y):
        max_iterations = 100
        real = x * 3.5 / self.image_width - 2.5
        imaginary = y * 2.0 / self.image_height - 1.0
        c_real = real
        c_imaginary = imaginary
        z_real = 0
        z_imaginary = 0
        iteration = 0

        while (z_real * z_real + z_imaginary * z_imaginary) < 4 and iteration < max_iterations:
            temp_real = z_real * z_real - z_imaginary * z_imaginary + c_real
            z_imaginary = 2 * z_real * z_imaginary + c_imaginary
            z_real = temp_real
            iteration += 1

        if iteration == max_iterations:
            return 0  # Black for points inside the Mandelbrot set
        else:
            return 255  # White for points outside the Mandelbrot set

    def account_log_out(self):
        self.save_info()
        self.registration_dialog.username_input.clear()
        self.registration_dialog.password_input.clear()
        self.registration_dialog.exec()
        self.username = self.registration_dialog.username_input.text()
        self.info = self.get_info()
        self.load_recent_decrypted_text()
        self.load_recent_encrypted_text()
        self.image = None
        self.image_copy = None
        self.image_width = None
        self.image_height = None
        self.ui.image.clear()
        self.ui.plainTextEdit_encrypt_text.clear()
        self.ui.plainTextEdit_decrypted_text.clear()

        if self.registration_dialog.result() == QDialog.Rejected:
            # Close the application if registration fails
            self.close()
            return

    def closeEvent(self, event):
        if self.registration_dialog.username_input.text():
            self.save_info()
        # This method is called when the window is about to close
        # print("Exit button pressed")
        event.accept()  # Accept the close event

    def get_info(self):
        info = None
        if os.path.exists(self.FILE_NAME):
            try:
                with open(self.FILE_NAME, 'r+') as file:
                    info = file.read()
                    if not info:
                        info = {
                            self.username: {
                                "images_url": [],
                                "encode_text": [],
                                "decode_text": []
                            }
                        }
                    else:
                        info = json.loads(info)
                        if self.username not in info:
                            info[self.username] = {
                                "images_url": [],
                                "encode_text": [],
                                "decode_text": []
                            }
                    return info
            except FileNotFoundError:
                pass

        with open(self.FILE_NAME, 'w') as file:
            pass

        info = {
            self.username: {
                "images_url": [],
                "encode_text": [],
                "decode_text": []
            }
        }
        return info

    def update_info(self, data, data_type):
        self.info[self.username][data_type].insert(0, data)
        if len(self.info[self.username][data_type]) > 3:
            self.info[self.username][data_type] = self.info[self.username][data_type][:3]
        # print(self.info)

    def save_info(self):
        with open(self.FILE_NAME, 'w') as file:
            info = json.dumps(self.info)
            file.write(info)

    def load_recent_encrypted_text(self):
        text = ""
        for i in self.info[self.username]["encode_text"]:
            text += i + '\n'
        self.ui.plainTextEdit_recent_encrypted_text.setPlainText(text)

    def load_recent_decrypted_text(self):
        text = ""
        for i in self.info[self.username]["decode_text"]:
            text += i + '\n'
        self.ui.plainTextEdit_recent_decrypted_text.setPlainText(text)

    def recent_img_1(self):
        if len(self.info[self.username]["images_url"]) > 0:
            self.open_image_in_label(self.info[self.username]["images_url"][0])

    def recent_img_2(self):
        if len(self.info[self.username]["images_url"]) > 1:
            self.open_image_in_label(self.info[self.username]["images_url"][1])

    def recent_img_3(self):
        if len(self.info[self.username]["images_url"]) > 2:
            self.open_image_in_label(self.info[self.username]["images_url"][2])

    def open_about_creators(self):
        info_text = ("<div align=center><h2>Данилишин Андрій Любомирович<br>"
                     "Винничук Максим Петрович<br>"
                     "Караневич Владислав Олександрович</h2></div>")
        self.open_info_window(info_text, "Розробники")

    def open_about_programme(self):
        info_text = """<h3>Програма створена для роботи з bmp файлами.<br> 
Операції які можна з ними проводити:<br>
-Створення нового файлу<br>
-Відкриття існуючого файлу<br>
-Можливість "непомітно" записати інформацію у файл<br>
-Можливість зчитати інформацію з файлу<br>
-Можливість на вибір задати три бітмапи та три колірні схеми для файлу<br>
-Можливість повернути назад незмінний файл після застосування бітмапи<br>

При виході з програми останні три записи зберігаються<br>

Програма написана на мові програмування Python з використанням бібліотеки PySide6<br></h3>"""
        self.open_info_window(info_text, "Про програму")

    def open_user_instruction(self):
        info_text = """<div alighn=center>
<h1>Інструкція користування</h1><br></div>
<div align=left><h4>
<p>У верхній панелі розташовано 7 кнопок<br>
Завантажити фото- при натисканні відкриється вікно windows для вибору bmp файлу для подальшої роботи з ним<br>
Зберегти фото - при натисканні відкривається вікно windows для збереження фотографії<br>
Створити нове фото - при натисканні відкривається вікно в якому потрібно задати розміри фотографії, розміри можна задати від 100 до 1000px<br>
Інструкція користувача - те, що ви зараз переглядаєте<br>
Про програму - короткі відомості про програму<br>
Про авторів - імена тих, хто причепний до створення цієї програми<br>
Вийти з акаунту - вихід з поточного акауну з можливістю переключитись на інший</p>
<p>У лівій панелі розташовано панелі для вибору кольорової комбінації та бітової карти, а також можливість зашифрувати і зчитати повідомлення з файлу<br>
Застовувати - при натисканні відбувається застосовування вибраної бітової карти та колірної комбінації<br>
Повернути оригінал - при натисканні фото відкатується до стану перед змінами бітової карти<br>
Зашифрувати текст - при натисканні весь текст з верхнього поля буде зашифровано у файл<br>
Розшифрувати текст - при натисканні у нижнє поле буде записано раніше зашифроване повідомлення з файлу</p>
<p>У правій панелі розташо кнопки які повертають 3 останні фотографії з якими працював користувач а також 3 останні зашифровані та розшифровані повідомлення</p></h4>"""
        self.open_info_window(info_text, "Інструкція користувача")

    def open_info_window(self, info_text, title_text):
        info_window = InfoWindow(info_text, title_text)
        info_window.exec()

    def revert_changes(self):
        if self.image is None:
            return
        rgb_values = []
        for i in range(self.image.height()):
            a = []
            for j in range(self.image.width()):
                color = QColor(self.image_copy.pixel(j, i))
                red = color.red()
                green = color.green()
                blue = color.blue()
                a.append([red, green, blue])
            rgb_values.append(a)
        self.create_img_from_np_array(np.array(rgb_values, dtype=np.uint8), False)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
