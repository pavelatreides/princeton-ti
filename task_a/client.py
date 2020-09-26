"""
This encapsulates the model object and initialization necessary
to run all the functions of the client

Potential Improvements:
Lazy pirate pattern - Client Side
Proof of concept closing and restarting sockets
Save frames into a data structure, coordinates, heatmap, original image
"""
import logging
import matplotlib.pyplot as plt
import cv2
import zmq
from utils import MessageContainer


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
        self.REQUEST_TIMEOUT = 2500
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

        #start at first frame
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
            #If coordinates came back
            if self.overlay.shape[1] == 2:
                plt.scatter(1024, 1024)
                plt.scatter(self.overlay[:, 1] * 4,
                            self.overlay[:, 0] * 4, alpha=None)
            #If a heatmap came back
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
        #Naive message to load model
        if messagetype == 'importmodel':
            message = MessageContainer(path=path)
            self.socket.send_pyobj(message)
            logging.info('Client sent model path: %s' % path)
        #For heatmaps and peak coordinates
        else:
            decoded = self.decode()
            message = MessageContainer(image=decoded)
            if messagetype == "predictpeaks":
                message.returnpeaks = True
            self.socket.send_pyobj(message)
            logging.info('Client sent frame %s' % self.index)
        #Wait for reply
        self.overlay = self.socket.recv_pyobj()
        logging.info("Client received reply")


def main():
    """ Main """
    testclient = ImageClient()
    #FIX THE BELOW
    testclient.run()


if __name__ == '__main__':
    main()
