"""
Responsible for loading the model, listening for requests,receiving the data,
predicting the output heatmaps,and sending it back to the client.

Potential Improvements:
Implement timeout when client is silent or move whole paradigm to ROUTER/DEALER
Implement own serializer for performance gains (see link below)
https://pyzmq.readthedocs.io/en/latest/serialization.html#builtin-serialization
"""
import logging
from multiprocessing import Process
import scipy
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import zmq
from IPython import embed
import utils_a

class ModelServerProcess(Process):
    """ Server that doesn't share memory but is responsible for loading the model,
    listening for requests, receiving the data,
    predicting the output heatmaps,and sending it back to the client. """

    def __init__(self, port="5556"):
        Process.__init__(self)
        #self.ip = ip
        self.port = port
        self.peaks = None
        self.model = None
        # input to model
        self.x = None
        # output to model
        self.y = None
        self.image = None
        self.returnpeaks = None
        self.threshold = None
        self.terminating = None
        self.coordinates = None
        self.message = None

    def load_model(self, path):
        """Load Keras model"""
        self.model = tf.keras.models.load_model(path, compile=False)
        logging.debug(f"Inputs: {self.model.inputs}")
        logging.debug(f"Outputs: {self.model.outputs}")

    def preprocess(self):
        """Preprocess the image for inference"""
        # Input must be float32 in [0, 1] and of rank 4 with shape
        # (batch_size, height, width, 1).
        self.x = np.expand_dims(self.image, axis=0).astype("float32") / 255.
        self.x = tf.image.resize(self.x, size=[512, 512])

    def visualize_output(self):
        """Visualize output from predictions."""
        plt.figure(figsize=(9, 8))
        plt.colorbar()
        if self.returnpeaks:
            plt.scatter(self.coordinates[:, 1] * 4,
                        self.coordinates[:, 0] * 4, alpha=None)
        else:
            plt.imshow(self.y.squeeze())
        plt.show()

    def predict(self):
        """Predict heatmap from image."""
        self.y = self.model.predict(self.x)
        logging.debug(f"Y.shape = {self.y.shape}")
        logging.debug(f"Y.dtype = {self.y.dtype}")

    def find_peaks(self,native = True):
        """Find local peaks from heatmap"""
        #convert tensor to 2d image
        preop = self.y.squeeze()
        ##use from-scratch implimentation based on mask method
        if native:
            postop = utils_a.find_peaks(preop)
            #convert mask to array of indexes
            postop = np.nonzero(postop)
            #convert shape from 2,n_peaks to n_peaks,2
            self.coordinates = np.array(postop).transpose()
        ##use skimage
        else:
            self.coordinates = peak_local_max(
                preop, min_distance=5, threshold_rel=0.5)

        logging.debug(f"coordinates.shape = {self.coordinates.shape}")

    def run(self):
        """Run server workflow, waiting for a request in between"""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        try:
            socket.bind("tcp://*:%s" % self.port)
        except zmq.ZMQError as e:
            #We connect to client first, so we need to acknowledge.
            #If this was a case where it was on the wrong port,
            #further requests would still bubble up the error
            logging.info('Surprise! Client has already connected!')
            return

        # self-terminate if too long without any requests
        while not self.terminating:
            # Wait for next request from client
            message = socket.recv_pyobj()
            #When client wants model to load
            if message.path:
                #message.path = '../models/best_model.h5'
                logging.info("Server received model path: %s" % message.path)
                self.load_model(message.path)
                socket.send_pyobj('Model loaded')
            #When client wants peak values returned
            elif message.returnpeaks:
                #
                logging.info("Server received request for peaks")
                self.image = message.image
                self.preprocess()
                self.predict()
                self.find_peaks()
                socket.send_pyobj(self.coordinates)
                self.visualize_output()
            #When client wants a heatmap returned
            else:
                logging.info("Server received request for heatmap")
                self.image = message.image
                self.preprocess()
                self.predict()
                socket.send_pyobj(self.y)
                self.visualize_output()
            logging.info("Server sent back requested data")
        # self.visualize_output()


def main():
    """ Main """
    testserver = ModelServerProcess()
    testserver.load_model('../models/best_model.h5')
    testserver.start()


if __name__ == '__main__':
    main()
