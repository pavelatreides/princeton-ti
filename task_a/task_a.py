"""
File executes application by generating the appropriates objects, GUI and processes

Potential Improvement:
Get logging from server to show up in console.
"""

import logging
import sys
import time
import multiprocessing
from client import ImageClient
from server import ModelServerProcess
from IPython import embed
from gui import TaskAWidget
from PySide2 import QtCore, QtWidgets, QtGui


def main():
    """Main"""
    # Let user know their resources
    logging.basicConfig(
        filename='HISTORYlistener.log',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.debug(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    # Only want to hear client and GUI doing their job
    app = QtWidgets.QApplication([])
    widget = TaskAWidget()

    #size chosen arbitrarily
    widget.resize(550, 600)
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
