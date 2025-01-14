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
        # Set up the layout
        layout = QVBoxLayout()

        # Create a group box for the login form
        login_group = QGroupBox("Login Form")
        login_layout = QVBoxLayout()

        # Switch input for Left/Right
        switch_layout = QHBoxLayout()
        self.switch_label = QLabel('Select Eye:')
        self.switch_input = QComboBox(self)
        self.switch_input.addItems(["Left", "Right"])  # Add options
        switch_layout.addWidget(self.switch_label)
        switch_layout.addWidget(self.switch_input)
        login_layout.addLayout(switch_layout)

        # Image upload
        image_layout = QHBoxLayout()
        self.image_label = QLabel('Upload image:')
        self.image_button = QPushButton('Upload Image', self)
        self.image_button.clicked.connect(self.upload_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_button)
        login_layout.addLayout(image_layout)

        # Display selected image
        self.image_display = QLabel(self)
        self.image_display.setAlignment(Qt.AlignCenter)
        login_layout.addWidget(self.image_display)

        # Identify button
        self.identify_button = QPushButton('Identify', self)
        self.identify_button.clicked.connect(self.identify_user)
        login_layout.addWidget(self.identify_button)

        # Welcome message label
        self.welcome_label = QLabel(self)
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setFont(QFont("Arial", QFont.Bold))  # Set large font
        self.welcome_label.setStyleSheet("font-size: 30px; font-weight: bold;")  # Set text color
        self.welcome_label.setVisible(False)  # Hide initially
        login_layout.addWidget(self.welcome_label)

        # Table to display identification results
        self.result_table = QTableWidget(self)
        self.result_table.setColumnCount(3)  # Method, Distance, Label
        self.result_table.setHorizontalHeaderLabels(["Method", "Distance", "Label"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table read-only
        self.result_table.setVisible(False)  # Hide the table initially
        login_layout.addWidget(self.result_table)

        # Link to registration page
        self.register_link = QLabel('<a href="#">Not registered? Click here to register or add an eye</a>')
        self.register_link.setAlignment(Qt.AlignCenter)
        self.register_link.setOpenExternalLinks(False)
        self.register_link.linkActivated.connect(self.go_to_register)
        login_layout.addWidget(self.register_link)

        # Set the layout for the group box
        login_group.setLayout(login_layout)

        # Add the group box to the main layout
        layout.addWidget(login_group)

        # Set the layout to the window
        self.setLayout(layout)

    def upload_image(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            # Load the image and display it
            pixmap = QPixmap(file_name)
            self.image_display.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.image_path = file_name

    def identify_user(self):
        # Check if an image is uploaded
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, 'Error', 'Please upload an image.')
            return

        # Generate random embeddings
        output = process_image(self.image_path, self.switch_input.currentText().lower())

        classic_embedding = json.loads(output["iris_template_output"])
        resnet_embedding = json.loads(output["full_eye_prediction"])
        resnet_normalized_embedding = json.loads(output["normalized_iris_prediction"])

        matches, is_found = find_all_matches_under_threshold(classic_embedding, resnet_embedding, resnet_normalized_embedding)

        if not is_found:
            self.welcome_label.setText(f"User Rejected")
            self.welcome_label.setVisible(True)
            self.result_table.setRowCount(0)
            return

        # Update the welcome message
        if matches:
            self.welcome_label.setText(f"Welcome, {matches[0]['label']}!")
            self.welcome_label.setVisible(True)

        # Clear the table
        self.result_table.setRowCount(0)

        # Populate the table with results
        for match in matches:
            row_position = self.result_table.rowCount()
            self.result_table.insertRow(row_position)

            # Add method, distance, and label to the table
            self.result_table.setItem(row_position, 0, QTableWidgetItem(match["method"]))
            self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{match['distance']:.4f}"))
            self.result_table.setItem(row_position, 2, QTableWidgetItem(match["label"]))

        # Make the table visible
        self.result_table.setVisible(True)

    def go_to_register(self):
        # Switch to the registration page
        self.stacked_widget.setCurrentIndex(1)


class RegisterPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        # Set up the layout
        layout = QVBoxLayout()

        # Create a group box for the registration form
        form_group = QGroupBox("Registration Form")
        form_layout = QVBoxLayout()

        # Username input
        username_layout = QHBoxLayout()
        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit(self)
        username_layout.addWidget(self.username_label)
        username_layout.addWidget(self.username_input)
        form_layout.addLayout(username_layout)

        # Switch input for Left/Right
        switch_layout = QHBoxLayout()
        self.switch_label = QLabel('Select Eye:')
        self.switch_input = QComboBox(self)
        self.switch_input.addItems(["Left", "Right"])  # Add options
        switch_layout.addWidget(self.switch_label)
        switch_layout.addWidget(self.switch_input)
        form_layout.addLayout(switch_layout)

        # Image upload
        image_layout = QHBoxLayout()
        self.image_label = QLabel('Select an image:')
        self.image_button = QPushButton('Upload Image', self)
        self.image_button.clicked.connect(self.upload_image)
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.image_button)
        form_layout.addLayout(image_layout)

        # Display selected image
        self.image_display = QLabel(self)
        self.image_display.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(self.image_display)

        # Submit button
        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_form)
        form_layout.addWidget(self.submit_button)

        # Back button
        self.back_button = QPushButton('Back to Login', self)
        self.back_button.clicked.connect(self.go_to_login)
        form_layout.addWidget(self.back_button)

        # Set the layout for the group box
        form_group.setLayout(form_layout)

        # Add the group box to the main layout
        layout.addWidget(form_group)

        # Set the layout to the window
        self.setLayout(layout)

    def upload_image(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            # Load the image and display it
            pixmap = QPixmap(file_name)
            self.image_display.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.image_path = file_name

    def submit_form(self):
        # Get the username and image path
        username = self.username_input.text()
        image_path = getattr(self, 'image_path', None)

        # Validate the input
        if not username:
            QMessageBox.warning(self, 'Error', 'Please enter a username.')
            return
        if not image_path:
            QMessageBox.warning(self, 'Error', 'Please select an image.')
            return

        # Create embeddings
        output = process_image(image_path, self.switch_input.currentText().lower())

        # Insert into the database
        is_new_user = insert_user(username, output["iris_template_output"], output["full_eye_prediction"], output["normalized_iris_prediction"])
        if is_new_user:
            QMessageBox.information(self, 'Success', 'User registered successfully!')
        else:
            QMessageBox.information(self, 'Success', 'Eye registered successfully!')

        # Clear the form
        self.username_input.clear()
        self.image_display.clear()
        self.image_path = None

    def go_to_login(self):
        # Switch back to the login page
        self.stacked_widget.setCurrentIndex(0)


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the stacked widget
        self.stacked_widget = QStackedWidget()

        # Create the login and register pages
        self.login_page = LoginPage(self.stacked_widget)
        self.register_page = RegisterPage(self.stacked_widget)

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.register_page)

        # Set up the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

        # Set window properties
        self.setWindowTitle("Comparison of Iris Recognition Techniques")
        self.resize(1000, 800)

        # Apply a global stylesheet to increase font size
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
    # Initialize the database
    initialize_database()

    # Start the application
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
