# Knight Industries Dashboard License plate recognition and search system
# r0

import os
import sys
import cv2
import ast
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine


from PyQt6.QtCore import Qt

from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor

from PyQt6.QtWidgets import (
    QTextEdit,
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QSizePolicy,
    QScrollArea,
    QTabWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QDoubleSpinBox,
)

# All this magic to make the Utils folder work.
# ----------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from utils.dashboard_utils import search4id

# -------------------------------------

BACKGROUND_OPACITY = 0.80

route = "sql"
# Temp loading location of license plate data
if route == "csv":

    df = pd.read_csv(r"./data/JackExampleData/aplr_ocr_results.csv")

else:
    query = """
    SELECT * 
    FROM cvp_results
    """
    conn_str = Path(__file__).parent / "../data/cvp_database.db"
    engine = create_engine(f"sqlite:///{conn_str}")
    df = pd.read_sql(query, engine)
    # df=pd.read_sql(r"./data/JackExampleData/aplr_ocr_results.db")

df = df.dropna(subset=["ocr_confidence"])


def func(df_list):
    # print(df_list)
    if df_list == "NA":
        return np.nan
    df_list = ast.literal_eval(df_list)
    df_list = [float(x) for x in df_list if not pd.isna(x)]
    return round(sum(df_list) / len(df_list), 5)


# Calculate average confidence
df["avg_confidence"] = df["ocr_confidence"].apply(func)

# sort by avg confidence
df = df.sort_values(by="avg_confidence", ascending=False)

SEARCH_COLUMNS = [
    ("Plate", "ocr_text"),
    ("Avg Confidence", "avg_confidence"),
    ("Region", "ocr_region"),
    ("File Path", "file_name"),
]

DEFAULT_COLUMNS = [
    ("Plate", "ocr_text"),
    ("Avg Confidence", "avg_confidence"),
    ("Region", "ocr_region"),
]


# This creates the background Image for the dashboard
class BackgroundWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.bg = QPixmap("data/assets/Bwxt_backgrnd.jpg")

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.bg.isNull():
            scaled = self.bg.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.setOpacity(BACKGROUND_OPACITY)
            painter.drawPixmap(0, 0, scaled)
        painter.setOpacity(1.0)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))


# This is the image viewer on the main page
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
        qimg = QImage(img.data, w, h, c * w, QImage.Format.Format_RGB888)

        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimg)))
        self.fitInView(
            self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )


# This is the license plate search tab
class LicensePlateTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self.setWindowTitle("BDSC COMPUTER VISION PIPELINE DASHBOARD")
        # self.resize(1800, 1000)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

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

        self.search = QLineEdit()
        self.search.setPlaceholderText("Enter Vehicle Search")
        self.search.textChanged.connect(self.update_dashboard)
        layout.addWidget(self.search)

        # HORIZONTAL CONTENT AREA
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)

        # LEFT PANEL The search results will be displayed here
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

        # RIGHT PANEL This is where the table will appear with metadata
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # right_panel.setStyleSheet("border:none;")

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
            pix = QPixmap(path).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio)
            lbl.setPixmap(pix)

        lbl.setToolTip(name)
        lbl.mousePressEvent = lambda e, p=path: self.on_thumbnail_click(p)
        self.thumb_layout.addWidget(lbl)

    def update_dashboard(self):

        txt = self.search.text().strip()

        self.clear_thumbs()

        imagedir = r"./data/license_plate_detection/test/images"

        # Columns to show when searching
        SEARCH_COLUMNS = [
            ("Plate", "ocr_text"),
            ("Avg Confidence", "avg_confidence"),
            ("Region", "ocr_region"),
            ("File Path", "file_name"),
        ]

        # Columns to show by default
        DEFAULT_COLUMNS = [
            ("Plate", "ocr_text"),
            ("Avg Confidence", "avg_confidence"),
            ("Region", "ocr_region"),
        ]

        if txt:

            results, img_name, plate_text, confidence = search4id(df, txt)

            if not img_name:

                self.table.setRowCount(1)
                self.table.setColumnCount(1)
                self.table.setHorizontalHeaderLabels(["Search Results"])

                self.table.setItem(0, 0, QTableWidgetItem("No match found"))

                return

            # ----------------------------------
            # Add thumbnails for all matches
            # ----------------------------------
            for name, plate, conf in zip(img_name, plate_text, confidence):

                path = os.path.join(imagedir, name + ".jpg")

                self.add_thumb(f"{plate} ({float(conf):.2f})", path)

            # ----------------------------------
            # Load first matching image
            # ----------------------------------
            import cv2
            import ast

            first_img = os.path.join(imagedir, img_name[0] + ".jpg")

            img = cv2.imread(first_img)

            # Get first matching dataframe row
            row_dict = dict(zip(df.columns, results[0]))

            bbox = ast.literal_eval(row_dict["yolo_xy_coords"])

            # Assuming format = [x1, x2, y1, y2]
            x1, y1, x2, y2 = bbox

            bbox = ast.literal_eval(row_dict["yolo_xy_coords"])
            self.load_image_with_bbox(first_img, bbox)

            # ----------------------------------
            # Populate search results table
            # ----------------------------------
            self.table.setRowCount(len(results))
            self.table.setColumnCount(len(SEARCH_COLUMNS))

            self.table.setHorizontalHeaderLabels([col[0] for col in SEARCH_COLUMNS])

            for r, row_data in enumerate(results):

                row_dict = dict(zip(df.columns, row_data))

                for c, (_, df_col) in enumerate(SEARCH_COLUMNS):

                    value = row_dict.get(df_col, "")

                    self.table.setItem(r, c, QTableWidgetItem(str(value)))

        else:

            # ----------------------------------
            # Default table view
            # ----------------------------------
            self.table.setRowCount(len(df))
            self.table.setColumnCount(len(DEFAULT_COLUMNS))

            self.table.setHorizontalHeaderLabels([col[0] for col in DEFAULT_COLUMNS])

            for r in range(len(df)):

                for c, (_, df_col) in enumerate(DEFAULT_COLUMNS):

                    value = df.iloc[r][df_col]

                    self.table.setItem(r, c, QTableWidgetItem(str(value)))

    def load_image_with_bbox(self, image_path, bbox=None):
        import cv2

        img = cv2.imread(image_path)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        temp_path = "_temp_bbox.jpg"
        cv2.imwrite(temp_path, img)

        self.viewer.load_image(temp_path)

    def on_thumbnail_click(self, image_path):
        # You likely need to look up bbox from df using image name
        import ast

        name = os.path.splitext(os.path.basename(image_path))[0]

        row = df[df["file_name"] == name]

        if not row.empty:
            bbox = ast.literal_eval(row.iloc[0]["yolo_xy_coords"])
            self.load_image_with_bbox(image_path, bbox)
        else:
            self.load_image_with_bbox(image_path)


# This is the file upload tab
class UploadTab(QWidget):

    def __init__(self):
        super().__init__()

        self.files = []

        self.setStyleSheet("""
            QWidget{
                background:transparent;
                color:white;
            }

            QGroupBox{
                border:1px solid #3daee9;
                margin-top:12px;
                font-weight:bold;
            }

            QGroupBox::title{
                subcontrol-origin:margin;
                left:10px;
                padding:0 5px;
            }

            QTextEdit,
            QLineEdit,
            QComboBox,
            QTableWidget{

                background-color:rgba(0,0,0,120);
                border:1px solid gray;
                color:white;
            }

            QPushButton{
                background:#2d2d2d;
                border:1px solid #3daee9;
                padding:6px;
            }

            QPushButton:hover{
                background:#3d3d3d;
            }

            QProgressBar{
                border:1px solid gray;
                text-align:center;
            }

            QProgressBar::chunk{
                background:#3daee9;
            }
        """)

        mainLayout = QVBoxLayout(self)

        ##################################################################
        # TITLE
        ##################################################################

        title = QLabel("BDSC COMPUTER VISION PIPELINE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:18pt;font-weight:bold;")

        mainLayout.addWidget(title)

        ##################################################################
        # PARAMETERS
        ##################################################################

        missionBox = QGroupBox("PARAMETERS")

        form = QFormLayout()
        # These boxes need to be linked to API calls.  So it can "dynamically build the API call."
        # Confidence
        self.conf = QDoubleSpinBox()
        self.conf.setValue(0.75)
        self.conf.setRange(0.0, 1.0)
        self.conf.setSingleStep(0.01)
        # Labels
        self.labels = QLineEdit()
        self.labels.setPlaceholderText("Comma-seperated values for Zero-Shot")
        # OCR
        self.do_ocr = QComboBox()
        self.do_ocr.addItems(["true", "false"])
        # Zero-Shot
        self.do_zs = QComboBox()
        self.do_zs.addItems(["true", "false"])

        form.addRow("Bounding Box Confidence", self.conf)
        form.addRow("Zero-Shot Labels", self.labels)
        form.addRow("Do Zero-Shot?", self.do_zs)
        form.addRow("Do OCR?", self.do_ocr)

        missionBox.setLayout(form)

        mainLayout.addWidget(missionBox)

        ##################################################################
        # IMAGE QUEUE
        ##################################################################

        queueBox = QGroupBox("IMAGE QUEUE")
        queueLayout = QVBoxLayout()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Filename", "Status", "Size (KB)"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        queueLayout.addWidget(self.table)
        queueBox.setLayout(queueLayout)
        mainLayout.addWidget(queueBox)

        ##################################################################
        # UPLOAD CONTROL
        ##################################################################

        controlBox = QGroupBox("UPLOAD CONTROL")

        controlLayout = QGridLayout()

        self.status = QLabel("● READY")
        self.status.setStyleSheet("color:#00ff66;font-weight:bold;")

        self.imageCount = QLabel("Images Selected : 0")

        self.selectButton = QPushButton("SELECT IMAGES")
        self.uploadButton = QPushButton("RUN PIPELINE")

        self.uploadButton.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        controlLayout.addWidget(self.status, 0, 0)
        controlLayout.addWidget(self.imageCount, 0, 1)

        controlLayout.addWidget(self.selectButton, 1, 0)
        controlLayout.addWidget(self.uploadButton, 1, 1)

        controlLayout.addWidget(self.progress, 2, 0, 1, 2)

        controlBox.setLayout(controlLayout)

        mainLayout.addWidget(controlBox)

        ##################################################################
        # PROGRESS LOG
        ##################################################################

        logBox = QGroupBox("PROGRESS LOG")

        logLayout = QVBoxLayout()

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        logLayout.addWidget(self.log)

        logBox.setLayout(logLayout)

        mainLayout.addWidget(logBox)

        ##################################################################
        # SIGNALS
        ##################################################################

        self.selectButton.clicked.connect(self.select_images)
        self.uploadButton.clicked.connect(self.start_upload)

    ##################################################################
    # IMAGE SELECTION
    ##################################################################

    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not files:
            return

        self.files = files
        self.imageCount.setText(f"Images Selected : {len(files)}")
        self.table.setRowCount(0)

        for image in files:
            path = Path(image)
            size_kb = path.stat().st_size / 1024
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(path.name))
            self.table.setItem(row, 1, QTableWidgetItem("Queued"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{size_kb:.1f}"))

        self.uploadButton.setEnabled(True)

        self.log.append(f"Loaded {len(files)} image(s).")

        self.status.setText("● READY")
        self.status.setStyleSheet("color:#00ff66;font-weight:bold;")

    def start_upload(self):

        settings = {
            "conf": self.conf.value(),
            "labels": self.labels.text(),
            "do_ocr": self.do_ocr.currentText(),
            "do_zs": self.do_zs.currentText(),
        }

        for i, image in enumerate(self.files):
            self.log.append(f"Evaluating image {i + 1} of {len(self.files)}")
            response = requests.post(
                "http://127.0.0.1:8000/pipeline",
                data=settings,
                files={"file": open(image, "rb")},
            )
            data = response.json()
            fmt_response = json.dumps(data)
            self.log.append(f"Image {i + 1} response:\n{fmt_response}\n")
            self.table.setItem(i, 1, QTableWidgetItem("Evaluated"))
            self.progress.setValue(int((i + 1) / len(self.files) * 100))

        self.log.append("All images evaulated successfully.")


# This sets the Top banner image
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
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


# The Main Window code brings all the pieces together.
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        bg = BackgroundWidget()
        self.setCentralWidget(bg)

        root = QVBoxLayout(bg)
        root.setContentsMargins(0, 0, 0, 0)

        banner = QLabel()
        pix = QPixmap("data/assets/BWXT_Border.png")

        banner.setFixedHeight(100)
        banner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # IMPORTANT: do NOT stretch image
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Scale image down (not up)
        scaled_pix = pix.scaled(
            800,
            150,  # <-- control image size here
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        banner.setPixmap(scaled_pix)
        root.addWidget(banner)

        glass = QWidget()
        root.addWidget(glass)

        layout = QVBoxLayout(glass)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)

        self.tabs.addTab(LicensePlateTab(), "SEARCH")
        self.tabs.addTab(UploadTab(), "UPLOAD")

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

    # This make the esc button exit the dashboard
    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Escape:
            self.close()

        super().keyPressEvent(event)


# Makes is run Full Screen
if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = MainWindow()

    win.showFullScreen()

    sys.exit(app.exec())
