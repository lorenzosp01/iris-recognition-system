import json
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QMessageBox, QGroupBox, QStackedWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QComboBox)

from db import initialize_database, insert_user, find_all_matches_under_threshold
from src.demo.image_processing import process_image


class LoginPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        login_group = QGroupBox("Login Form")
        login_layout = QVBoxLayout()

        switch_layout = QHBoxLayout()
        self.switch_label = QLabel('Select Eye:')
        self.switch_input = QComboBox(self)
        self.switch_input.addItems(["Left", "Right"])
        switch_layout.addWidget(self.switch_label)
        switch_layout.addWidget(self.switch_input)
        login_layout.addLayout(switch_layout)

        image_layout = QHBoxLayout()
        self.image_label = QLabel('Upload image:')
        self.image_button = QPushButton('Upload Image', self)
        self.image_button.clicked.connect(self.upload_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_button)
        login_layout.addLayout(image_layout)

        self.image_display = QLabel(self)
        self.image_display.setAlignment(Qt.AlignCenter)
        login_layout.addWidget(self.image_display)

        self.identify_button = QPushButton('Identify', self)
        self.identify_button.clicked.connect(self.identify_user)
        login_layout.addWidget(self.identify_button)

        self.welcome_label = QLabel(self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setFont(QFont("Arial", QFont.Bold))
        self.welcome_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        self.welcome_label.setVisible(False)
        login_layout.addWidget(self.welcome_label)

        self.result_table = QTableWidget(self)
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["Method", "Distance", "Label"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setVisible(False)
        login_layout.addWidget(self.result_table)

        self.register_link = QLabel('<a href="#">Not registered? Click here to register or add an eye</a>')
        self.register_link.setAlignment(Qt.AlignCenter)
        self.register_link.setOpenExternalLinks(False)
        self.register_link.linkActivated.connect(self.go_to_register)
        login_layout.addWidget(self.register_link)

        login_group.setLayout(login_layout)

        layout.addWidget(login_group)

        self.setLayout(layout)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_display.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.image_path = file_name

    def identify_user(self):
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, 'Error', 'Please upload an image.')
            return

        output = process_image(self.image_path, self.switch_input.currentText().lower())

        if "error" in output:
            QMessageBox.warning(self, 'Error', output["error"])
            return

        classic_embedding = json.loads(output["iris_template_output"])
        resnet_embedding = json.loads(output["full_eye_prediction"])
        resnet_normalized_embedding = json.loads(output["normalized_iris_prediction"])

        matches, is_found = find_all_matches_under_threshold(classic_embedding, resnet_embedding, resnet_normalized_embedding)

        if not is_found:
            self.welcome_label.setText(f"User Rejected")
            self.welcome_label.setVisible(True)
            self.result_table.setRowCount(0)
            return

        if matches:
            self.welcome_label.setText(f"Welcome, {matches[0]['label']}!")
            self.welcome_label.setVisible(True)

        self.result_table.setRowCount(0)

        for match in matches:
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)

            self.result_table.setItem(row_position, 0, QTableWidgetItem(match["method"]))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{match['distance']:.4f}"))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(match["label"]))

        self.result_table.setVisible(True)

    def go_to_register(self):
        self.stacked_widget.setCurrentIndex(1)


class RegisterPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        form_group = QGroupBox("Registration Form")
        form_layout = QVBoxLayout()

        username_layout = QHBoxLayout()
        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit(self)
        username_layout.addWidget(self.username_label)
        username_layout.addWidget(self.username_input)
        form_layout.addLayout(username_layout)

        switch_layout = QHBoxLayout()
        self.switch_label = QLabel('Select Eye:')
        self.switch_input = QComboBox(self)
        self.switch_input.addItems(["Left", "Right"])
        switch_layout.addWidget(self.switch_label)
        switch_layout.addWidget(self.switch_input)
        form_layout.addLayout(switch_layout)

        image_layout = QHBoxLayout()
        self.image_label = QLabel('Select an image:')
        self.image_button = QPushButton('Upload Image', self)
        self.image_button.clicked.connect(self.upload_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_button)
        form_layout.addLayout(image_layout)

        self.image_display = QLabel(self)
        self.image_display.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(self.image_display)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_form)
        form_layout.addWidget(self.submit_button)

        self.back_button = QPushButton('Back to Login', self)
        self.back_button.clicked.connect(self.go_to_login)
        form_layout.addWidget(self.back_button)

        form_group.setLayout(form_layout)

        layout.addWidget(form_group)

        self.setLayout(layout)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_display.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.image_path = file_name

    def submit_form(self):
        username = self.username_input.text()
        image_path = getattr(self, 'image_path', None)

        if not username:
            QMessageBox.warning(self, 'Error', 'Please enter a username.')
            return
        if not image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image.')
            return


        output = process_image(image_path, self.switch_input.currentText().lower())

        if "error" in output:
            QMessageBox.warning(self, 'Error', output["error"])
            return

        is_new_user = insert_user(username, output["iris_template_output"], output["full_eye_prediction"], output["normalized_iris_prediction"])
        if is_new_user:
            QMessageBox.information(self, 'Success', 'User registered successfully!')
        else:
            QMessageBox.information(self, 'Success', 'Eye registered successfully!')

        self.username_input.clear()
        self.image_display.clear()
        self.image_path = None

    def go_to_login(self):
        self.stacked_widget.setCurrentIndex(0)


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.stacked_widget = QStackedWidget()

        self.login_page = LoginPage(self.stacked_widget)
        self.register_page = RegisterPage(self.stacked_widget)

        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.register_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

        self.setWindowTitle("Comparison of Iris Recognition Techniques")
        self.resize(1000, 800)

        self.setStyleSheet("""
            QWidget {
                font-size: 16px;
            }
            QPushButton {
                font-size: 16px;
                padding: 10px;
            }
            QLabel {
                font-size: 16px;
            }
            QLineEdit {
                font-size: 16px;
                padding: 5px;
            }
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
            }
            QTableWidget {
                font-size: 14px;
            }
        """)


if __name__ == '__main__':
    initialize_database()

    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
