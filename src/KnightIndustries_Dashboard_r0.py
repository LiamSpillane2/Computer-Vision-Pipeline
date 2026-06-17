#Kinight Industries Dashboard License plate recognition and search system
#r0 

import sys, os
import pandas as pd
import numpy as np
import cv2
import ast

from PyQt6.QtCore import Qt, QTimer, QRectF, QUrl
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QLinearGradient
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QScrollArea, QTabWidget, QPushButton, QFileDialog,
    QTextEdit, QProgressBar, QSlider, QCheckBox, QFormLayout, QSizePolicy

)

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtTextToSpeech import QTextToSpeech
from sympy import root
from torch import layout

#All this magic to make the Utils folder work.
#----------------------------------
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.dashboard_utils import search4id
#-------------------------------------

BACKGROUND_OPACITY = 0.80

#Temp loading location of license plate data
df=pd.read_csv(r"./data/JackExampleData/aplr_ocr_results.csv")
df = df.dropna(subset = ["confidence"])
def func(df_list):
    df_list = ast.literal_eval(df_list)
    df_list = [float(x) for x in df_list if not pd.isna(x)]
    return round(sum(df_list)/len(df_list),5)
    

df["avg_confidence"] = df["confidence"].apply(func)

#sort by avg confidence
df = df.sort_values(
    by="avg_confidence",
    ascending=False
)

SEARCH_COLUMNS = [

    ("Plate", "text"),
    ("Avg Confidence", "avg_confidence"),
    ("Region", "region"),
    ("File Path", "file_path")
]

DEFAULT_COLUMNS = [
    ("Plate", "text"),
    ("Avg Confidence", "avg_confidence"),
    ("Region", "region")
]


class BackgroundWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.bg = QPixmap("data/assets/background.jpg")

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.bg.isNull():
            scaled = self.bg.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.setOpacity(BACKGROUND_OPACITY)
            painter.drawPixmap(0, 0, scaled)
        painter.setOpacity(1.0)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))


class KITTScanner(QWidget):
    def __init__(self):
        super().__init__()
        self.pos = 0
        self.direction = 1
        self.setMinimumHeight(45)
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(15)

    def animate(self):
        self.pos += self.direction * 8
        if self.pos > self.width() - 120:
            self.direction = -1
        if self.pos < 0:
            self.direction = 1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(10, 10, 10))
        grad = QLinearGradient(self.pos, 0, self.pos + 120, 0)
        grad.setColorAt(0.0, QColor(255,0,0,0))
        grad.setColorAt(0.5, QColor(255,0,0,255))
        grad.setColorAt(1.0, QColor(255,0,0,0))
        p.fillRect(QRectF(self.pos, 8, 120, 25), grad)



class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(factor, factor)

    def load_image(self, path):
        self.scene.clear()

        if not os.path.exists(path):
            self.scene.addText(f"Missing image:\n{path}")
            return

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape
        qimg = QImage(img.data, w, h, c*w, QImage.Format.Format_RGB888)

        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimg)))
        self.fitInView(self.scene.itemsBoundingRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

class LicensePlateTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.setWindowTitle("Knight Industries Intelligence Console")
        #self.resize(1800, 1000)
        self.setAttribute(
        Qt.WidgetAttribute.WA_TranslucentBackground
        )

        self.setStyleSheet("""
            background: transparent;
        """)
        self.setObjectName("searchTab")

        self.setStyleSheet("""
        #searchTab {
            background: transparent;
        }
        """)
        # ---------------------------------
        # Subtitle (keep or remove)
        # ---------------------------------
        subtitle = QLabel("VEHICLE LOCATION SYSTEM")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle.setStyleSheet("""
            color:cyan;
            font-size:18px;
        """)

        layout.addWidget(subtitle)

        layout.addWidget(KITTScanner())

        self.search = QLineEdit()
        self.search.setPlaceholderText("Enter Vehicle Search")
        self.search.textChanged.connect(self.update_dashboard)
        layout.addWidget(self.search)

        # HORIZONTAL CONTENT AREA
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)

        # LEFT PANEL
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Make the whole panel transparent
        left_panel.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(600, 300)

        # Keep border but remove background
        self.viewer.setStyleSheet("""
            background-color: transparent;
            border: 2px solid red;
        """)

        left_layout.addWidget(self.viewer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # IMPORTANT: remove scroll background
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QScrollArea viewport {
                background-color: transparent;
            }
        """)

        self.thumb_widget = QWidget()

        self.thumb_widget.setStyleSheet("""
            background-color: transparent;
            border: 2px solid red;
        """)

        self.thumb_layout = QHBoxLayout(self.thumb_widget)

        scroll.setWidget(self.thumb_widget)
        left_layout.addWidget(scroll)

        # RIGHT PANEL
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        #right_panel.setStyleSheet("border:none;")

        db_label = QLabel("VEHICLE DATABASE")
        db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        db_label.setStyleSheet("color:red;font-size:18px;font-weight:bold;")

        right_layout.addWidget(db_label)

        self.table = QTableWidget()
        right_layout.addWidget(self.table)

        content_layout.addWidget(left_panel, 3)
        content_layout.addWidget(right_panel, 2)

        layout.addWidget(content_widget)

        self.setStyleSheet("""
            QLabel { color: cyan; }

            QLineEdit{
                background:#111;
                color:cyan;
                border:2px solid red;
                padding:8px;
                font-size:16px;
            }

            QTableWidget{
                background:rgba(0,0,0,180);
                color:cyan;
                border:2px solid red;
                gridline-color:red;
            }

            QHeaderView::section{
                background:black;
                color:red;
                border:1px solid red;
                padding:4px;
            }
        """)

        self.update_dashboard()

    def clear_thumbs(self):
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def add_thumb(self, name, path):
        lbl = QLabel()

        if os.path.exists(path):
            pix = QPixmap(path).scaled(
                120, 120,
                Qt.AspectRatioMode.KeepAspectRatio
            )
            lbl.setPixmap(pix)

        lbl.setToolTip(name)
        lbl.mousePressEvent = lambda e, p=path: self.viewer.load_image(p)
        self.thumb_layout.addWidget(lbl)

    def update_dashboard(self):

        txt = self.search.text().strip()

        self.clear_thumbs()

        imagedir = r"./data/license_plate_detection/train/images"

        # Columns to show when searching
        SEARCH_COLUMNS = [
            ("Plate", "text"),
            ("Avg Confidence", "avg_confidence"),
            ("Region", "region"),
            ("File Path", "file_path")
        ]

        # Columns to show by default
        DEFAULT_COLUMNS = [
            ("Plate", "text"),
            ("Avg Confidence", "avg_confidence"),
            ("Region", "region")
        ]

        if txt:

            results, img_name, plate_text, confidence = search4id(
                df,
                txt
            )

            if not img_name:

                self.table.setRowCount(1)
                self.table.setColumnCount(1)
                self.table.setHorizontalHeaderLabels(
                    ["Search Results"]
                )

                self.table.setItem(
                    0,
                    0,
                    QTableWidgetItem("No match found")
                )

                return

            # ----------------------------------
            # Add thumbnails for all matches
            # ----------------------------------
            for name, plate, conf in zip(
                img_name,
                plate_text,
                confidence
            ):

                path = os.path.join(
                    imagedir,
                    name + ".jpg"
                )

                self.add_thumb(
                    f"{plate} ({float(conf):.2f})",
                    path
                )

            # ----------------------------------
            # Load first matching image
            # ----------------------------------
            import cv2
            import ast

            first_img = os.path.join(
                imagedir,
                img_name[0] + ".jpg"
            )

            img = cv2.imread(first_img)

            # Get first matching dataframe row
            row_dict = dict(
                zip(df.columns, results[0])
            )

            bbox = ast.literal_eval(row_dict["bbox"])

            # Assuming format = [x1, x2, y1, y2]
            x1, x2, y1, y2 = bbox

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                3
            )

            temp_path = "_temp_bbox.jpg"
            cv2.imwrite(temp_path, img)

            self.viewer.load_image(temp_path)


            # ----------------------------------
            # Populate search results table
            # ----------------------------------
            self.table.setRowCount(len(results))
            self.table.setColumnCount(len(SEARCH_COLUMNS))

            self.table.setHorizontalHeaderLabels(
                [col[0] for col in SEARCH_COLUMNS]
            )

            for r, row_data in enumerate(results):

                row_dict = dict(
                    zip(df.columns, row_data)
                )

                for c, (_, df_col) in enumerate(
                    SEARCH_COLUMNS
                ):

                    value = row_dict.get(
                        df_col,
                        ""
                    )

                    self.table.setItem(
                        r,
                        c,
                        QTableWidgetItem(
                            str(value)
                        )
                    )

        else:

            # ----------------------------------
            # Default table view
            # ----------------------------------
            self.table.setRowCount(len(df))
            self.table.setColumnCount(
                len(DEFAULT_COLUMNS)
            )

            self.table.setHorizontalHeaderLabels(
                [col[0] for col in DEFAULT_COLUMNS]
            )

            for r in range(len(df)):

                for c, (_, df_col) in enumerate(
                    DEFAULT_COLUMNS
                ):

                    value = df.iloc[r][df_col]

                    self.table.setItem(
                        r,
                        c,
                        QTableWidgetItem(
                            str(value)
                        )
                    )

class UploadTab(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("""
            

            QLabel {
                background-color: transparent;
                color: white;
            }

            QTextEdit {
                background-color: transparent;
                border: 1px solid gray;
                color: white;
            }

            QProgressBar {
                background-color: transparent;
                border: 1px solid gray;
                color: white;
            }

            QProgressBar::chunk {
                background-color: #3daee9;
            }
        """)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("IMAGE UPLOAD SYSTEM"))

        self.btn = QPushButton("SELECT IMAGES")

        self.progress = QProgressBar()

        self.log = QTextEdit()
        self.log.setStyleSheet("background-color: transparent;")

        layout.addWidget(self.btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)

        self.btn.clicked.connect(self.select_images)

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if files:
            self.log.append(f"Loaded {len(files)} image(s)")

class SettingsTab(QWidget):
    def __init__(self, audio_output):
        super().__init__()
        self.audio_output = audio_output
        layout = QFormLayout(self)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(15)
        self.volume.valueChanged.connect(
            lambda v: self.audio_output.setVolume(v/100)
        )
        layout.addRow("Music Volume", self.volume)
        layout.addRow(QCheckBox("Enable Startup Voice"))
        layout.addRow(QCheckBox("Launch Full Screen"))

class Banner(QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.pixmap_orig = pixmap
        self.setFixedHeight(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def resizeEvent(self, event):
        scaled = self.pixmap_orig.scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        bg = BackgroundWidget()
        self.setCentralWidget(bg)

        root = QVBoxLayout(bg)
        root.setContentsMargins(0,0,0,0)

        banner = QLabel()
        pix = QPixmap("data/assets/knight banner.jpg")

        banner.setFixedHeight(100)
        banner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # IMPORTANT: do NOT stretch image
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Scale image down (not up)
        scaled_pix = pix.scaled(
            800, 150,  # <-- control image size here
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        banner.setPixmap(scaled_pix)
        root.addWidget(banner)

        glass = QWidget()
        root.addWidget(glass)

        layout = QVBoxLayout(glass)


         # ---------------------------------
        # Background Music
        # Orginial audio player 
        # ---------------------------------
        self.audio_output = QAudioOutput()
        self.music_player = QMediaPlayer()
        self.music_player.setAudioOutput(self.audio_output)

        music_file = os.path.abspath("data/assets/Knight_Rider.wav")
        self.music_player.setSource(QUrl.fromLocalFile(music_file))

        self.audio_output.setVolume(0.15)

        # ✅ Proper Qt6 looping (NO helper function needed)
        self.music_player.setLoops(QMediaPlayer.Loops.Infinite)

        self.music_player.play()
        #-----------------------------------


        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)

        self.tabs.addTab(LicensePlateTab(), "SEARCH")
        self.tabs.addTab(UploadTab(), "UPLOAD")
        self.tabs.addTab(SettingsTab(self.audio_output), "SETTINGS")

        layout.addWidget(self.tabs)

        self.setStyleSheet("""
        QLabel{color:cyan;}
        QTabWidget::pane{
            border:2px solid red;
            background:rgba(0,0,0,120);
        }
        QTabBar::tab{
            background:black;
            color:cyan;
            border:2px solid red;
            min-width:180px;
            min-height:35px;
        }
        QTabBar::tab:selected{
            background:rgb(60,0,0);
            color:red;
        }
        """)

# #KITT voice orginial code  

        self.tts = QTextToSpeech()
        self.tts.say("Knight Industries Vehicle Location System. Online.")


    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Escape:
            self.close()

        super().keyPressEvent(event)
                 
if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = MainWindow()

    win.showFullScreen()

    sys.exit(app.exec())