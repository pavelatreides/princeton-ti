"""
This encapsulates the model object and initialization necessary
to run all the functions of the client

Potential Improvements:
Save frames into a data structure, coordinates, heatmap, original image
"""
import logging
import matplotlib.pyplot as plt
import cv2
import zmq
import sys
from utils_a import MessageContainer


class ImageClient():
    """Handles all the functions of the client:  loading image data, sending it to the server,
      receive the model prediction results """

    def __init__(self, port=5556):
        # current frame in view
        self.currentframe = None
        # prepped frame as input to model
        self.x = None
        # last sent overlay
        self.overlay = None
        self.reader = None
        self.n_frames = None
        # 5 second wait time before retry
        self.REQUEST_TIMEOUT = 5000
        self.REQUEST_RETRIES = 3
        self.port = port
        self.context = None
        self.socket = None
        self.index = None
        self.imagepath = None

    def load_frames(self):
        """Load the video and read first frame"""
        # Open video clip for reading
        filetype = 'mp4'
        if filetype in ('mp4', 'avi'):
            self.reader = cv2.VideoCapture(self.imagepath)
            logging.info('Video loaded successfully')

        # Get the number of frames in the video.
        self.n_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.debug(f"Number of frames loaded = {self.n_frames}")

        # start at first frame
        self.seek(0)

    def seek(self, index):
        """Seek to specific frame"""
        self.index = index
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, self.index)
        _, self.currentframe = self.reader.read()
        logging.info(f"Found frame {self.index+1}")

    def decode(self):
        """Decode the image data."""
        decoded = self.currentframe[:, :, :1]  # convert to grayscale
        logging.debug(f"img.shape = {self.currentframe.shape}")
        logging.debug(f"img.dtype = {self.currentframe.dtype}")
        return decoded

    def visualize_frame(self):
        """Visualize image."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.currentframe.squeeze(), cmap="gray")
        plt.show()

    def visualize_output(self):
        """Visualize output."""
        if not isinstance(self.overlay, str):
            plt.figure(figsize=(9, 8))
            # If coordinates came back
            if self.overlay.shape[1] == 2:
                plt.scatter(1024, 1024)
                plt.scatter(self.overlay[:, 1] * 4,
                            self.overlay[:, 0] * 4, alpha=None)
            # If a heatmap came back
            else:
                plt.imshow(self.overlay.squeeze())
            plt.show()

    def start(self):
        """Set up socket"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect('tcp://localhost:%s' % self.port)
        logging.info('Running client on port: %s ' % self.port)

    def send(self, messagetype, path=None):
        """Send a message in a container to the server"""
        # Naive message to load model
        if messagetype == 'importmodel':
            message = MessageContainer(path=path)
            self.socket.send_pyobj(message)
            logging.info('Client sent model path: %s' % path)
        # For heatmaps and peak coordinates
        else:
            decoded = self.decode()
            message = MessageContainer(image=decoded)
            if messagetype == "predictpeaks":
                message.returnpeaks = True
            self.socket.send_pyobj(message)
            logging.info('Client sent frame %s' % self.index)

        # wait graciously for reply and handle delays like lazy pirate
        retries_left = self.REQUEST_RETRIES
        reply_received = False
        while not reply_received:
            if (self.socket.poll(self.REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
                self.overlay = self.socket.recv_pyobj()
                # confirm object is not one of three response types
                if (isinstance(self.overlay, str) or
                        self.overlay.shape[1] in (2, 256)):
                    logging.info("Client received reply")
                    reply_received = True
                    break
                else:
                    logging.error(
                        "Malformed reply from server: %s", self.overlay)
                    continue
            retries_left -= 1
            logging.warning("No response from server")
            # Something is wrong from the server. Close socket and remove it.
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            if retries_left == 0:
                logging.error("Server is corrupt or offline, abandoning..")
                sys.exit()

            logging.info("Reconnecting to server…")
            # Create new connection
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect('tcp://localhost:%s' % self.port)
            # resend request
            logging.info("Resending request")
            self.socket.send_pyobj(message)


def main():
    """ Main """
    testclient = ImageClient()
    testclient.start()


if __name__ == '__main__':
    main()
