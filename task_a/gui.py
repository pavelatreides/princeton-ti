"""
View/GUI for Task A. Here we define all visible elements, their styles and events

Potential Improvements:
Consider moving some big blocks to other methods, especially in init
Add modules to pkg-extension-whitelist
Make textedits editable only by dialog
"""
import copy
import sys
import logging
import numpy as np
from PySide2 import QtCore as qc, QtWidgets as qw, QtGui as qg
from client import ImageClient
from server import ModelServerProcess
# for overlaying images, might solve artifact problem
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
from IPython import embed
import cv2

class TaskAWidget(qw.QWidget):
    """
    Small GUI/Widget to handle client and server functions, validation logic
    """

    def __init__(self):
        super().__init__()
        self.client = ImageClient()
        self.client.start()
        self.server = None
        # define and format elements
        self.file_name = None
        self.modelloaded = None

        # define video file picker
        self.videopathbox = qw.QLineEdit()
        self.videopathbox.setPlaceholderText('Click Load to select the video')
        self.videodialog = qw.QFileDialog(self)


        # define model file picker
        self.modelpathbox = qw.QLineEdit()
        self.modelpathbox.setPlaceholderText(
            'Click Import to select the model')
        self.modeldialog = qw.QFileDialog(self)

        # define buttons
        self.serverbutton = qw.QPushButton("Start")
        self.predictbutton = qw.QPushButton("Predict")
        self.loadvideobutton = qw.QPushButton("Load")
        self.importmodelbutton = qw.QPushButton("Import")
        self.importmodelbutton.setEnabled(False)
        self.predictbutton.setEnabled(False)

        # define server status busy indicator
        self.busyindicator = qw.QProgressBar()

        # define radios
        self.overlayradio = qw.QRadioButton('Heatmap', self)
        #get heatmaps back by default
        self.overlayradio.setChecked(True)
        #don't touch until model is loaded by the server
        self.overlayradio.setEnabled(False)

        self.peakradio = qw.QRadioButton('Peaks', self)
        #don't touch until model is loaded by the server
        self.peakradio.setEnabled(False)

        #define frame counter
        self.text = qw.QLabel(
            f"Frame Number - {self.client.index}" + f"/{self.client.n_frames} ")
        self.text.setAlignment(qc.Qt.AlignCenter)

        # define frameslider
        self.frameslider = qw.QSlider()
        self.frameslider.setOrientation(qc.Qt.Horizontal)
        self.frameslider.setTickPosition(qw.QSlider.TicksBelow)
        self.frameslider.setEnabled(False)

        #define view of frame
        self.imageframe = qg.QPixmap(400, 400)
        self.label = qw.QLabel()
        self.label.setAlignment(qc.Qt.AlignCenter)
        self.label.setPixmap(self.imageframe)


        # Define 'Setup' panel
        self.setup_layout = qw.QHBoxLayout()
        self.setup_layout.addWidget(self.videopathbox)
        self.setup_layout.addWidget(self.loadvideobutton)
        self.setup_box = qw.QGroupBox("Setup")
        self.setup_box.setLayout(self.setup_layout)

        # Define 'Frame' panel
        self.frame_layout = qw.QVBoxLayout()
        self.frame_layout.addWidget(self.text)
        self.frame_layout.addWidget(self.label)
        self.frame_layout.addWidget(self.frameslider)
        self.frame_layout.addWidget(self.predictbutton)
        #define radios within 'Frame'
        self.radios = qw.QHBoxLayout()
        self.radios.addWidget(self.overlayradio,
                               alignment=qc.Qt.AlignCenter)
        self.radios.addWidget(
            self.peakradio, alignment=qc.Qt.AlignCenter)
        self.frame_layout.addLayout(self.radios)

        self.frame_box = qw.QGroupBox("Frame")
        self.frame_box.setLayout(self.frame_layout)

        #define 'Server' panel
        self.server_layout = qw.QVBoxLayout()
        self.server_layout.addWidget(self.busyindicator)
        self.server_layout.addWidget(self.serverbutton)
        #define model loading widgets in 'Server'
        self.server_sublayout = qw.QHBoxLayout()
        self.server_sub = qw.QWidget(self)
        self.server_sublayout.addWidget(self.modelpathbox)
        self.server_sublayout.addWidget(self.importmodelbutton)
        self.server_sub.setLayout(self.server_sublayout)
        self.server_layout.addWidget(self.server_sub)

        self.server_box = qw.QGroupBox("Server")
        self.server_box.setLayout(self.server_layout)

        # Create main layout
        self.mainlayout = qw.QVBoxLayout()
        self.mainlayout.addWidget(self.setup_box)
        self.mainlayout.addWidget(self.server_box)
        self.mainlayout.addWidget(self.frame_box)
        self.setLayout(self.mainlayout)

        # Connect user interaction signals/events to methods
        self.serverbutton.clicked.connect(self.on_server_start)
        self.predictbutton.clicked.connect(self.on_predict)
        self.loadvideobutton.clicked.connect(self.open_video_window)
        self.importmodelbutton.clicked.connect(self.open_model_window)
        self.frameslider.valueChanged.connect(self.on_slider_change)
        self.peakradio.toggled.connect(self.on_radio_change)

    def on_radio_change(self):
        """When user toggles the radios"""
        self.update_image()
        self.predictbutton.setEnabled(True)

    def on_slider_change(self):
        """When user moves the slider"""
        self.client.seek(self.frameslider.value())
        self.update_image()
        self.predictbutton.setText('Predict')
        # Predict only if model is already loaded
        #Known performance issue for large videos,
        #but only option if you want to prevent predictions before model load
        if self.modelloaded:
            self.predictbutton.setEnabled(True)

    def on_server_start(self):
        """When user starts the server"""
        # disable button until connected
        logging.debug('Server %s' % self.serverbutton.text())
        if self.serverbutton.text() == "Start":
            #don't let use do anything while waiting for request - synchronous..for now
            #we do this elsewhere in the code
            self.setEnabled(False)
            self.serverbutton.setText('Stop')
            self.server = None
            self.server = ModelServerProcess()
            self.server.start()
            self.setEnabled(True)
            self.busyindicator.setRange(0, 0)
            self.importmodelbutton.setEnabled(True)
        else:
            #when you want to disconnect from the server
            self.setEnabled(False)
            self.server.terminate()
            self.setEnabled(True)
            self.serverbutton.setText('Start')
            self.busyindicator.setRange(0, 100)
            self.predictbutton.setEnabled(False)
            self.peakradio.setEnabled(False)
            self.overlayradio.setEnabled(False)

    def on_predict(self):
        """When user asks the server to 'Predict'"""
        # disable button until returned and catch bad state
        if self.predictbutton.isEnabled():
            self.predictbutton.setEnabled(False)
            self.setEnabled(False)
            if self.overlayradio.isChecked():
                self.client.send("predictheatmap")
            else:
                self.client.send("predictpeaks")
            #paint overlay
            self.update_image("overlay")
            self.setEnabled(True)
            # self.client.visualize_output()

    def open_video_window(self):
        """When user imports the video"""
        videopath, _ = self.videodialog.getOpenFileName(self, str("Open File"),
                                                        "../inputdata/",
                                                        str("Videos (*.mp4 *.avi)"))
        # set video path to text input
        self.videopathbox.setText(videopath)
        self.client.imagepath = videopath
        #actually load video
        self.client.load_frames()
        self.update_image()

        #ensure WYSIWYG with video
        self.frameslider.setRange(0, self.client.n_frames - 1)
        self.frameslider.setValue(self.client.index)
        #arbitary, mostly to avoid rendering slowdown
        self.frameslider.setTickInterval(int(self.client.n_frames / 5))
        self.frameslider.setEnabled(True)

    def open_model_window(self):
        """When user picks the model"""
        videopath, _ = self.modeldialog.getOpenFileName(self, str("Open File"),
                                                        "../models/",
                                                        str("Models (*.h5)"))
        # set video path to text input
        self.modelpathbox.setText(videopath)
        self.setEnabled(False)
        self.client.send('importmodel', path=videopath)
        self.setEnabled(True)
        self.predictbutton.setEnabled(True)
        self.peakradio.setEnabled(True)
        self.overlayradio.setEnabled(True)
        self.modelloaded = True

    def update_image(self, include=None):
        """When image in 'Frame' needs to reflect client state"""
        #define background, which is just frame of video
        background_array = copy.deepcopy(self.client.currentframe)
        height, width, __ = background_array.shape
        logging.debug(f"background_array.shape = {background_array.shape}")
        bytes_per_line = 3 * width
        background = qg.QImage(background_array, width,
                               height, bytes_per_line, qg.QImage.Format_RGB888)
        #Paint overlay("foreground") on top of background
        if include == "overlay":
            if self.client.overlay.shape[1] == 2:
                # for coordinates
                #arbitary width, looked nice
                width = 20
                painter = qg.QPainter()
                painter.begin(background)
                painter.setBrush(qg.QBrush(qg.Qt.cyan, qg.Qt.SolidPattern))
                #draw circles where peaks are
                for (x, y) in self.client.overlay:
                    painter.drawEllipse(y * 4, x * 4, width, width)
                    logging.debug('Coordinate drawn')
                painter.end()
            else:
                #for heatmap
                #alpha is higher because greyscale, not matplotlib colormap
                alpha = 0.75
                # define foreground
                logging.debug(
                    f"self.client.overlay.shape = {self.client.overlay.shape}")
                foreground_array = self.client.overlay.squeeze()
                foreground_array = cv2.resize(
                    foreground_array, (background_array.shape[0], background_array.shape[1]))
                logging.debug(
                    f"foreground_array.shape = {foreground_array.shape}")
                logging.debug(f"Max = {np.amax(foreground_array)}")
                foreground_array = foreground_array.astype(np.float64)
                foreground_array = foreground_array * 255
                foreground_array = foreground_array.astype(np.uint8)
                foreground = qg.QImage(
                    foreground_array, width, height, qg.QImage.Format_Grayscale8)
                logging.warning('Conversion of overlay to PySide2 compatible format  has introduced some artifacts (speckling). Does not affect predictions, but should be investigated')
                #where the actual painting occurs (start and stop)
                painter = qg.QPainter()
                painter.begin(background)
                painter.setOpacity(alpha)
                painter.drawImage(0, 0, foreground)
                painter.end()
        self.imageframe = None
        self.imageframe = qg.QPixmap.fromImage(background)
        self.label.setPixmap(self.imageframe.scaledToWidth(400))
        self.label.show()
        #making sure counter is correct after seek or prediction
        self.text.setText(
            f"Frame - {self.client.index+1}" + f"/{self.client.n_frames} ")

def main():
    """ Main """
    app = qw.QApplication([])
    widget = TaskAWidget()
    widget.resize(550, 700)
    widget.show()

    sys.exit(app.exec_())
if __name__ == "__main__":
    main()
