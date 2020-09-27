"""
File executes application by generating the appropriates objects, GUI and processes

Potential Improvement:
Improve logging to also output to terminal(getLogger & handler management)
"""

import logging
import sys
import multiprocessing
from gui import TaskAWidget
from PySide2 import QtWidgets


def main():
    """Main"""
    # Let user know their resources and save activity to file
    logging.basicConfig(
        filename='history.log',
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info('Loading....')
    logging.debug(f"Number of CPU cores: {multiprocessing.cpu_count()}")
    # Only want to hear client and GUI doing their job
    app = QtWidgets.QApplication([])

    widget = TaskAWidget()
    # size chosen arbitrarily
    widget.resize(550, 600)
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
