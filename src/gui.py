import sys
import os
import shutil
import uuid
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                            QVBoxLayout, QWidget, QFileDialog, QFrame)
from PyQt5.QtGui import QPixmap, QFont, QColor, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cats vs Dogs Classification")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C3E50;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QLabel {
                color: white;
            }
            QFileDialog {
                background-color: #2C3E50;
                color: #FFFFFF;
            }
            QFileDialog QWidget {
                background-color: #2C3E50;
                color: #FFFFFF;
            }
            QFileDialog QPushButton {
                background-color: #3498DB;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                min-width: 100px;
            }
            QFileDialog QLineEdit {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 3px;
                padding: 5px;
            }
            QFileDialog QTreeView {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
            }
            QFileDialog QTreeView::item:selected {
                background-color: #3498DB;
            }
            QFileDialog QTreeView::item:hover {
                background-color: #2980B9;
            }
            QFileDialog QComboBox {
                background-color: #34495E;
                color: white;
                border: 1px solid #3498DB;
                border-radius: 3px;
                padding: 5px;
            }
            QFileDialog QLabel {
                color: white;
            }
        """)

        # Create temp directory if it doesn't exist
        self.temp_dir = os.path.abspath("../temp")
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        else:
            # Clean temp directory
            self.clean_temp_directory()

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setSpacing(20)

        # Track current image path
        self.current_image_path = None

        # Model logic
        self.model = load_model('../model/cats_vs_dogs_1.keras')
        self.classes = {
            0: 'The image you uploaded is a cat\'s image',
            1: 'The image you uploaded is a dog\'s image'
        }

        # Create widgets
        self.heading = QLabel("Cats vs Dogs Classifier")
        self.heading.setFont(QFont("Arial", 28, QFont.Bold))
        self.heading.setStyleSheet("""
            color: white;
            background-color: #34495E;
            padding: 20px;
            border-radius: 10px;
        """)
        self.heading.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.heading) # Add the heading to the frame

        self.title = QLabel("Select or drop a local image file to classify")
        self.title.setFont(QFont("Arial", 16))
        self.title.setStyleSheet("color: #ECF0F1; margin: 10px;")
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)

        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("""
            QFrame {
                background-color: #34495E;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        self.image_frame.setMinimumSize(400, 400)
        self.image_frame.setMaximumSize(600, 600)

        image_layout = QVBoxLayout(self.image_frame)

        self.sign_image = QLabel()
        self.sign_image.setAlignment(Qt.AlignCenter)
        self.sign_image.setStyleSheet("background-color: transparent;")
        image_layout.addWidget(self.sign_image)

        self.layout.addWidget(self.image_frame)

        self.label = QLabel()
        self.label.setFont(QFont("Arial", 14, QFont.Bold))
        self.label.setStyleSheet("""
            color: #2ECC71;
            background-color: #34495E;
            padding: 10px;
            border-radius: 5px;
        """)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(10)

        self.upload_btn = QPushButton("Upload an Image")
        button_layout.addWidget(self.upload_btn)
        self.upload_btn.clicked.connect(self.upload_image)

        self.classify_btn = QPushButton("Classify Animal")
        self.classify_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #219A52;
            }
        """)
        self.classify_btn.hide()
        button_layout.addWidget(self.classify_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.quit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.quit_btn)

        self.layout.addWidget(button_container)

        # Hide image frame and result label initially
        self.image_frame.hide()
        self.label.hide()

        # Enable drag and drop
        self.setAcceptDrops(True)

    def clean_temp_directory(self):
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            try:
                # Clean temp directory and previous results
                self.clean_temp_directory()
                self.label.setText("")
                self.label.hide()

                # Create new file in temp directory
                new_filename = f"{uuid.uuid4()}.jpg"
                new_file_path = os.path.join(self.temp_dir, new_filename)

                # Copy the file
                shutil.copy2(file_path, new_file_path)

                # Display image
                pixmap = QPixmap(new_file_path)
                scaled_pixmap = pixmap.scaled(
                    400, 400,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.sign_image.setPixmap(scaled_pixmap)

                # Show image frame and classify button
                self.image_frame.show()
                self.classify_btn.show()

                # Disconnect any existing connections
                try:
                    self.classify_btn.clicked.disconnect()
                except:
                    pass

                # Connect with new file path
                self.classify_btn.clicked.connect(lambda: self.classify(new_file_path))

            except Exception as e:
                print(f"Error: {e}")
                # self.label.setText(f"Error processing image: {str(e)}")
                self.label.setText(f"Error processing image")
                self.label.setStyleSheet("color: #E74C3C;")
                self.label.show()
        event.accept()

    def process_dropped_image(self, file_path):
        try:
            # Clean temp directory and previous results
            self.clean_temp_directory()
            self.label.setText("")
            self.label.hide()

            # Create new file in temp directory
            new_filename = f"{uuid.uuid4()}.jpg"
            new_file_path = os.path.join(self.temp_dir, new_filename)

            # Copy file
            shutil.copy2(file_path, new_file_path)

            # Display image
            pixmap = QPixmap(new_file_path)
            scaled_pixmap = pixmap.scaled(
                400, 400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.sign_image.setPixmap(scaled_pixmap)

            # Show image frame and classify button
            self.image_frame.show()
            self.classify_btn.show()

            # Disconnect any existing connections
            try:
                self.classify_btn.clicked.disconnect()
            except:
                pass

            # Connect with new file path
            self.classify_btn.clicked.connect(lambda: self.classify(new_file_path))

        except Exception as e:
            print(f"Error: {e}")
            # self.label.setText(f"Error processing image: {str(e)}")
            self.label.setText(f"Error processing image")
            self.label.setStyleSheet("color: #E74C3C;")
            self.label.show()

    def classify(self, file_path):
        try:
            image = Image.open(file_path)
            image = image.resize((128,128))
            image = np.array(image)
            image = np.expand_dims(image, axis=0)
            image = image/255

            predictions = self.model.predict([image])[0]
            pred = np.argmax(predictions)
            confidence = np.max(predictions)

            CONFIDENCE_THRESHOLD = 0.80

            if confidence >= CONFIDENCE_THRESHOLD:
                sign = self.classes[pred]
            else:
                sign = "This image appears to be neither a cat nor a dog"

            print(f"Confidence: {confidence:.2f}, Prediction: {sign}")
            self.label.setText(sign)
            self.label.show()

            if confidence >= CONFIDENCE_THRESHOLD:
                self.label.setStyleSheet("color: #2ECC71; background-color: #34495E; padding: 10px; border-radius: 5px;")
            else:
                self.label.setStyleSheet("color: #E74C3C; background-color: #34495E; padding: 10px; border-radius: 5px;")

        except Exception as e:
            # self.label.setText(f"Error during classification: {str(e)}")
            self.label.setText("""Error during classification.
                Please check if you uploaded an image and the image extension is jpg.""")
            self.label.setStyleSheet("color: #E74C3C; background-color: #34495E; padding: 10px; border-radius: 5px;")
            self.label.show()

    def upload_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select an Image",
                os.path.expanduser("~"),
                "Image Files (*.jpg *.png *.jpeg *.gif *.bmp)"
            )

            if file_path:
                self.process_dropped_image(file_path)
            else:
                print("No file chosen")

        except Exception as e:
            print(f"Error: {e}")
            # self.label.setText(f"Error uploading image: {str(e)}")
            self.label.setText(f"Error uploading image")
            self.label.setStyleSheet("color: #E74C3C;")
            self.label.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
