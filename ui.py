# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledjPeiIh.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGroupBox, QLabel,
    QMainWindow, QMenuBar, QPlainTextEdit, QPushButton,
    QRadioButton, QSizePolicy, QStatusBar, QWidget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1300, 800)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.top_frame = QFrame(self.centralwidget)
        self.top_frame.setObjectName(u"top_frame")
        self.top_frame.setGeometry(QRect(0, 0, 1050, 60))
        self.top_frame.setFrameShape(QFrame.StyledPanel)
        self.top_frame.setFrameShadow(QFrame.Raised)
        self.btn_download_image = QPushButton(self.top_frame)
        self.btn_download_image.setObjectName(u"btn_download_image")
        self.btn_download_image.setGeometry(QRect(10, 15, 150, 30))
        font = QFont()
        font.setPointSize(9)
        self.btn_download_image.setFont(font)
        self.btn_save_image = QPushButton(self.top_frame)
        self.btn_save_image.setObjectName(u"btn_save_image")
        self.btn_save_image.setGeometry(QRect(180, 15, 150, 30))
        self.btn_save_image.setFont(font)
        self.btn_create_new_image = QPushButton(self.top_frame)
        self.btn_create_new_image.setObjectName(u"btn_create_new_image")
        self.btn_create_new_image.setGeometry(QRect(350, 15, 150, 30))
        self.btn_create_new_image.setFont(font)
        self.btn_user_instruction = QPushButton(self.top_frame)
        self.btn_user_instruction.setObjectName(u"btn_user_instruction")
        self.btn_user_instruction.setGeometry(QRect(520, 15, 150, 30))
        self.btn_user_instruction.setFont(font)
        self.btn_about_programme = QPushButton(self.top_frame)
        self.btn_about_programme.setObjectName(u"btn_about_programme")
        self.btn_about_programme.setGeometry(QRect(690, 15, 150, 30))
        self.btn_about_programme.setFont(font)
        self.btn_about_creators = QPushButton(self.top_frame)
        self.btn_about_creators.setObjectName(u"btn_about_creators")
        self.btn_about_creators.setGeometry(QRect(860, 15, 150, 30))
        self.btn_about_creators.setFont(font)
        self.image = QLabel(self.centralwidget)
        self.image.setObjectName(u"image")
        self.image.setGeometry(QRect(250, 100, 800, 600))
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(0, 60, 2000, 5))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.left_frame = QFrame(self.centralwidget)
        self.left_frame.setObjectName(u"left_frame")
        self.left_frame.setGeometry(QRect(0, 60, 200, 740))
        self.left_frame.setFrameShape(QFrame.StyledPanel)
        self.left_frame.setFrameShadow(QFrame.Raised)
        self.frame = QFrame(self.left_frame)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(0, 20, 200, 160))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.label_1 = QLabel(self.frame)
        self.label_1.setObjectName(u"label_1")
        self.label_1.setGeometry(QRect(0, 0, 200, 30))
        self.label_1.setAlignment(Qt.AlignCenter)
        self.groupBox = QGroupBox(self.frame)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(10, 40, 180, 100))
        self.radioButton_color_1 = QRadioButton(self.groupBox)
        self.radioButton_color_1.setObjectName(u"radioButton_color_1")
        self.radioButton_color_1.setGeometry(QRect(10, 0, 180, 30))
        self.radioButton_color_2 = QRadioButton(self.groupBox)
        self.radioButton_color_2.setObjectName(u"radioButton_color_2")
        self.radioButton_color_2.setGeometry(QRect(10, 30, 180, 30))
        self.radioButton_color_3 = QRadioButton(self.groupBox)
        self.radioButton_color_3.setObjectName(u"radioButton_color_3")
        self.radioButton_color_3.setGeometry(QRect(10, 60, 180, 30))
        self.frame_2 = QFrame(self.left_frame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setGeometry(QRect(0, 170, 200, 160))
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.label_2 = QLabel(self.frame_2)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(0, 0, 200, 30))
        self.label_2.setAlignment(Qt.AlignCenter)
        self.groupBox_2 = QGroupBox(self.frame_2)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 40, 180, 100))
        self.radioButton_bitmap_1 = QRadioButton(self.groupBox_2)
        self.radioButton_bitmap_1.setObjectName(u"radioButton_bitmap_1")
        self.radioButton_bitmap_1.setGeometry(QRect(10, 0, 180, 30))
        self.radioButton_bitmap_2 = QRadioButton(self.groupBox_2)
        self.radioButton_bitmap_2.setObjectName(u"radioButton_bitmap_2")
        self.radioButton_bitmap_2.setGeometry(QRect(10, 30, 180, 30))
        self.radioButton_bitmap_3 = QRadioButton(self.groupBox_2)
        self.radioButton_bitmap_3.setObjectName(u"radioButton_bitmap_3")
        self.radioButton_bitmap_3.setGeometry(QRect(10, 60, 180, 30))
        self.frame_3 = QFrame(self.left_frame)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setGeometry(QRect(0, 410, 200, 410))
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)
        self.label_3 = QLabel(self.frame_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(0, 0, 200, 30))
        self.label_3.setAlignment(Qt.AlignCenter)
        self.plainTextEdit_encrypt_text = QPlainTextEdit(self.frame_3)
        self.plainTextEdit_encrypt_text.setObjectName(u"plainTextEdit_encrypt_text")
        self.plainTextEdit_encrypt_text.setGeometry(QRect(0, 50, 200, 60))
        self.plainTextEdit_encrypt_text.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.btn_encrypt_text = QPushButton(self.frame_3)
        self.btn_encrypt_text.setObjectName(u"btn_encrypt_text")
        self.btn_encrypt_text.setGeometry(QRect(25, 130, 150, 30))
        self.btn_encrypt_text.setFont(font)
        self.btn_decrypt_text_2 = QPushButton(self.frame_3)
        self.btn_decrypt_text_2.setObjectName(u"btn_decrypt_text_2")
        self.btn_decrypt_text_2.setGeometry(QRect(25, 180, 150, 30))
        self.btn_decrypt_text_2.setFont(font)
        self.plainTextEdit_decrypted_text = QPlainTextEdit(self.frame_3)
        self.plainTextEdit_decrypted_text.setObjectName(u"plainTextEdit_decrypted_text")
        self.plainTextEdit_decrypted_text.setGeometry(QRect(0, 230, 200, 60))
        self.plainTextEdit_decrypted_text.setTextInteractionFlags(Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)
        self.btn_apply_changes = QPushButton(self.left_frame)
        self.btn_apply_changes.setObjectName(u"btn_apply_changes")
        self.btn_apply_changes.setGeometry(QRect(25, 330, 150, 30))
        self.btn_apply_changes.setFont(font)
        self.btn_revert_changes = QPushButton(self.left_frame)
        self.btn_revert_changes.setObjectName(u"btn_revert_changes")
        self.btn_revert_changes.setGeometry(QRect(25, 380, 150, 30))
        self.btn_revert_changes.setFont(font)
        self.btn_exit = QPushButton(self.centralwidget)
        self.btn_exit.setObjectName(u"btn_exit")
        self.btn_exit.setGeometry(QRect(1135, 15, 150, 30))
        self.btn_exit.setFont(font)
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(200, 61, 7, 1200))
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(1095, 61, 7, 1200))
        self.line_3.setFrameShape(QFrame.VLine)
        self.line_3.setFrameShadow(QFrame.Sunken)
        self.right_frame = QFrame(self.centralwidget)
        self.right_frame.setObjectName(u"right_frame")
        self.right_frame.setGeometry(QRect(1100, 60, 200, 740))
        self.right_frame.setFrameShape(QFrame.StyledPanel)
        self.right_frame.setFrameShadow(QFrame.Raised)
        self.label_4 = QLabel(self.right_frame)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(0, 20, 200, 30))
        self.label_4.setAlignment(Qt.AlignCenter)
        self.btn_recent_image_1 = QPushButton(self.right_frame)
        self.btn_recent_image_1.setObjectName(u"btn_recent_image_1")
        self.btn_recent_image_1.setGeometry(QRect(20, 60, 40, 30))
        self.btn_recent_image_2 = QPushButton(self.right_frame)
        self.btn_recent_image_2.setObjectName(u"btn_recent_image_2")
        self.btn_recent_image_2.setGeometry(QRect(80, 60, 40, 30))
        self.btn_recent_image_3 = QPushButton(self.right_frame)
        self.btn_recent_image_3.setObjectName(u"btn_recent_image_3")
        self.btn_recent_image_3.setGeometry(QRect(140, 60, 40, 30))
        self.label_5 = QLabel(self.right_frame)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(0, 120, 200, 50))
        self.label_5.setAlignment(Qt.AlignCenter)
        self.plainTextEdit_recent_encrypted_text = QPlainTextEdit(self.right_frame)
        self.plainTextEdit_recent_encrypted_text.setObjectName(u"plainTextEdit_recent_encrypted_text")
        self.plainTextEdit_recent_encrypted_text.setGeometry(QRect(0, 190, 200, 90))
        self.plainTextEdit_recent_encrypted_text.setTextInteractionFlags(Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)
        self.label_6 = QLabel(self.right_frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(0, 300, 200, 50))
        self.label_6.setAlignment(Qt.AlignCenter)
        self.plainTextEdit_recent_decrypted_text = QPlainTextEdit(self.right_frame)
        self.plainTextEdit_recent_decrypted_text.setObjectName(u"plainTextEdit_recent_decrypted_text")
        self.plainTextEdit_recent_decrypted_text.setGeometry(QRect(0, 370, 200, 90))
        self.plainTextEdit_recent_decrypted_text.setTextInteractionFlags(Qt.TextSelectableByKeyboard|Qt.TextSelectableByMouse)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1300, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.btn_download_image.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0432\u0430\u043d\u0442\u0430\u0436\u0438\u0442\u0438 \u0444\u043e\u0442\u043e", None))
        self.btn_save_image.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0431\u0435\u0440\u0435\u0433\u0442\u0438 \u0444\u043e\u0442\u043e", None))
        self.btn_create_new_image.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0442\u0432\u043e\u0440\u0438\u0442\u0438 \u043d\u043e\u0432\u0435 \u0444\u043e\u0442\u043e", None))
        self.btn_user_instruction.setText(QCoreApplication.translate("MainWindow", u"\u0406\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0456\u044f \u043a\u043e\u0440\u0438\u0441\u0442\u0443\u0432\u0430\u0447\u0430", None))
        self.btn_about_programme.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u043e \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u0443", None))
        self.btn_about_creators.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u043e \u0430\u0432\u0442\u043e\u0440\u0456\u0432", None))
        self.image.setText("")
        self.label_1.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0438\u0431\u0435\u0440\u0456\u0442\u044c \u043a\u043e\u043b\u044c\u043e\u0440\u043e\u0432\u0443 \u043a\u043e\u043c\u0431\u0456\u043d\u0430\u0446\u0456\u044e:", None))
        self.groupBox.setTitle("")
        self.radioButton_color_1.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.radioButton_color_2.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.radioButton_color_3.setText(QCoreApplication.translate("MainWindow", u"3", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0438\u0431\u0435\u0440\u0456\u0442\u044c \u0431\u0456\u0442\u043e\u0432\u0443 \u043a\u0430\u0440\u0442\u0443:", None))
        self.groupBox_2.setTitle("")
        self.radioButton_bitmap_1.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.radioButton_bitmap_2.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.radioButton_bitmap_3.setText(QCoreApplication.translate("MainWindow", u"3", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0432\u0435\u0434\u0456\u0442\u044c \u0442\u0435\u043a\u0441\u0442 \u0434\u043b\u044f \u0448\u0438\u0444\u0440\u0443\u0432\u0430\u043d\u043d\u044f:", None))
        self.btn_encrypt_text.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0448\u0438\u0444\u0440\u0443\u0432\u0430\u0442\u0438 \u0442\u0435\u043a\u0441\u0442", None))
        self.btn_decrypt_text_2.setText(QCoreApplication.translate("MainWindow", u"\u0420\u043e\u0437\u0448\u0438\u0444\u0440\u0443\u0432\u0430\u0442\u0438 \u0442\u0435\u043a\u0441\u0442", None))
        self.plainTextEdit_decrypted_text.setPlainText("")
        self.btn_apply_changes.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0441\u0442\u043e\u0441\u0443\u0432\u0430\u0442\u0438", None))
        self.btn_revert_changes.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0432\u0435\u0440\u043d\u0443\u0442\u0438 \u043e\u0440\u0438\u0433\u0456\u043d\u0430\u043b", None))
        self.btn_exit.setText(QCoreApplication.translate("MainWindow", u"\u0412\u0438\u0439\u0442\u0438 \u0437 \u0430\u043a\u0430\u0443\u043d\u0442\u0443", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0435\u0449\u043e\u0434\u0430\u0432\u043d\u0456 \u0444\u043e\u0442\u043e:", None))
        self.btn_recent_image_1.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.btn_recent_image_2.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.btn_recent_image_3.setText(QCoreApplication.translate("MainWindow", u"3", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>\u041e\u0441\u0442\u0430\u043d\u043d\u0456 \u0437\u0430\u0448\u0438\u0444\u0440\u043e\u0432\u0430\u043d\u0456</p><p>\u043f\u043e\u0432\u0456\u0434\u043e\u043c\u043b\u0435\u043d\u043d\u044f:</p></body></html>", None))
        self.plainTextEdit_recent_encrypted_text.setPlainText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>\u041e\u0441\u0442\u0430\u043d\u043d\u0456 \u0440\u043e\u0437\u0448\u0438\u0444\u0440\u043e\u0432\u0430\u043d\u0456</p><p>\u043f\u043e\u0432\u0456\u0434\u043e\u043c\u043b\u0435\u043d\u043d\u044f:</p></body></html>", None))
        self.plainTextEdit_recent_decrypted_text.setPlainText("")
    # retranslateUi

