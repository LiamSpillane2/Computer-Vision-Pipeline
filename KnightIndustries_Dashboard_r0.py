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
    QScrollArea
)

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtTextToSpeech import QTextToSpeech
from sympy import root
from torch import layout
from utils.dashboard_utils import search4id

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
        painter.fillRect(
            self.rect(),
            QColor(255, 0, 0, 25)
            )

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
        p.fillRect(self.rect(), QColor(10,10,10))

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Knight Industries Intelligence Console")
        #self.resize(1800, 1000)

        bg = BackgroundWidget()
        self.setCentralWidget(bg)
        # ---------------------------------
        # Main Layout
        # ---------------------------------
        root = QVBoxLayout(bg)

        # Remove all margins so banner touches window edges
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------------------------------
        # Full Width Banner
        # ---------------------------------
        self.banner_label = QLabel()
        self.banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_label.setContentsMargins(0, 0, 0, 0)

        self.banner_label.setStyleSheet("""
            border:none;
            margin:0px;
            padding:0px;
        """)

        self.banner_pixmap = QPixmap(
            "data/assets/knight banner.jpg"
        )

        self.banner_label.setFixedHeight(100)

        # Initial display
        self.banner_label.setPixmap(
            self.banner_pixmap.scaled(
                self.width(),
                100,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        # Add banner directly to root layout
        root.addWidget(self.banner_label)

        # ---------------------------------
        # Main Glass Panel
        # ---------------------------------
        glass = QWidget()

        glass.setStyleSheet("""
            background-color: rgba(0,0,0,20);
            border-radius: 15px;
        """)

        root.addWidget(glass)

        layout = QVBoxLayout(glass)

        # Optional: reduce spacing between banner and content
        layout.setContentsMargins(10, 5, 10, 10)

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

        #left_panel.setStyleSheet("border:2px solid red")
        #border: 2px solid red;
        

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(600, 300)
        left_layout.addWidget(self.viewer)
        #this adds border to the image viewer 
        self.viewer.setStyleSheet("border:2px solid red")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
       

        self.thumb_widget = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_widget)
        #this adds a border around the thumbnails pics
        self.thumb_widget.setStyleSheet("border:2px solid red")

        scroll.setWidget(self.thumb_widget)
        left_layout.addWidget(scroll)
        #will this add the single border on the thumbnail
        #this gives a double red border 
        #scroll.setStyleSheet("border:2px solid red")

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

        # ---------------------------------
        # Background Music
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




#KITT voice 
        self.tts = QTextToSpeech()
        self.tts.say(
        "Knight Industries Vehicle Location System. Online."
        )   


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

    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Escape:
            self.close()

        super().keyPressEvent(event)
                 
if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = MainWindow()

    win.showFullScreen()

    sys.exit(app.exec())