"""
Run this file to execute partial attempt to Task B.

What I didn't get to:
- Misdetection, duplication and identity switch logic
- Comparison to ground truth
- Performance optimization
- Cleaning up long blocks into methods
- Binding methods and variables to a class for reuse
"""
import logging
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import IPython as ip
import utils_b
from skimage.feature import peak_local_max


def load_model(modelpath):
    """Load Keras model"""
    _model = tf.keras.models.load_model(modelpath, compile=False)
    logging.debug(f"Inputs: {_model.inputs}")
    logging.debug(f"Outputs: {_model.outputs}")
    return _model


def load_frames(imagepath):
    """Load the video and read first frame"""
    # Open video clip for reading
    filetype = 'mp4'
    if filetype in ('mp4', 'avi'):
        _reader = cv2.VideoCapture(imagepath)
        logging.info('Video loaded successfully')
    # Get the number of frames in the video.
    _n_frames = int(_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.debug(f"Number of frames loaded = {_n_frames}")
    return _reader, _n_frames


def seek(_time_index, _reader):
    """Seek to specific frame"""
    _reader.set(cv2.CAP_PROP_POS_FRAMES, _time_index)
    __, _img = _reader.read()
    logging.info(f"Loaded frame {_time_index+1}")
    return _img


def visualize_frame(_img):
    """Visualize image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(_img.squeeze(), cmap="gray")
    plt.show()


def decode(_img):
    """Decode the image data."""
    decoded = _img[:, :, :1]  # convert to grayscale
    logging.debug(f"img.shape = {decoded.shape}")
    logging.debug(f"img.dtype = {decoded.dtype}")
    return decoded


def preprocess(_img):
    """Preprocess the image for inference"""
    # Input must be float32 in [0, 1] and of rank 4 with shape
    # (batch_size, height, width, 1).
    _x = np.expand_dims(_img, axis=0).astype("float32") / 255.
    _x = tf.image.resize(_x, size=[512, 512])
    return _x


def predict(_x, _model):
    """Predict heatmap from image."""
    _y = _model.predict(_x)
    logging.debug(f"Y.shape = {_y.shape}")
    logging.debug(f"Y.dtype = {_y.dtype}")
    return _y


def visualize_output(_img):
    """Visualize output from predictions."""
    plt.figure(figsize=(9, 8))
    # if coordinates
    if 2 in _img.shape:
        # if low-res
        if 256 in _img.shape:
            scaling_factor = 4
            offset = scaling_factor / 2
            for (x, y) in _img * scaling_factor:
                plt.plot(y + offset, x + offset, 'co')
                logging.debug('Coordinate drawn')
        else:
            for (x, y) in _img:
                plt.plot(y, x, 'co')
                logging.debug('Coordinate drawn')
                plt.ylim(ymin=1024, ymax=0)
                plt.xlim(xmin=0, xmax=1024)
        plt.show()
    # if just image
    else:
        plt.imshow(_img.squeeze())
    # plt.colorbar()
    plt.show()


def convert_heatmap_to_peak_array(_y, native=True):
    """Find local peaks from heatmap"""
    # convert tensor to 2d image
    threshold = 0.3
    preop = _y.squeeze()
    threshed = np.where(preop < threshold, 0, preop)
    # use from-scratch implimentation based on mask method
    if native:
        postop = utils_b.find_peaks(threshed)
        # convert mask to array of indexes
        postop = np.nonzero(postop)
        # convert shape from 2,n_peaks to n_peaks,2
        _peak_array = np.array(postop).transpose()
    # use skimage
    else:
        _peak_array = peak_local_max(
            preop, min_distance=5, threshold_rel=0.5)
    logging.debug(f"peak_array.shape = {_peak_array.shape}")
    return _peak_array


def get_gaussian_bias(array, peakx, peaky):
    """Since prediction is low-res, obtaining an approximation of a gaussian bias
    allows us to increase peak detection accuracy. See the following source:
    https://www.researchgate.net/publication/2401027_A_Comparison_of_Algorithms_for_Subpixel_Peak_Detection
    """
    xbias = 0.5 * (np.log(array[peakx - 1, peaky]) - np.log(array[peakx + 1, peaky])) / (np.log(
        array[peakx - 1, peaky]) - 2 * np.log(array[peakx, peaky]) + np.log(array[peakx + 1, peaky]))
    ybias = 0.5 * (np.log(array[peakx, peaky - 1]) - np.log(array[peakx, peaky + 1])) / (np.log(
        array[peakx, peaky - 1]) - 2 * np.log(array[peakx, peaky]) + np.log(array[peakx, peaky + 1]))
    return xbias, ybias


def main():
    """Main"""
    # Load tensorflow model
    model_path = "../models/best_model.h5"
    model = load_model(model_path)

    # Load frames
    image_path = "../inputdata/many_flies.clip.mp4"
    reader, n_frames = load_frames(image_path)
    trackstack = {}
    # For each frame in the video, generate heatmaps
    # for time_index in np.arange(n_frames):
    # since misdetection is inevitable and I did not code the logic yet,
    # I stop at 20 frames to plot what I have so far
    for time_index in np.arange(200):
        img = seek(time_index, reader)

        # visualize_frame(img)
        decoded_img = decode(img)
        x = preprocess(decoded_img)
        y = predict(x, model)
        # visualize_output(y)

        # get peaks from prediction
        peak_array = convert_heatmap_to_peak_array(y)

        # Visualize peak array
        # visualize_output(peak_array)
        # Visualize peak array superimposed on scaled output

        # Get bias for peaks. We calculate from y because it is the unfiltered heatmap
        # Coordinates correspond to indexes in shape of y
        # for each set of coordinates in peak_array
        biases = np.zeros(peak_array.shape)
        i = 0
        for (x_peak, y_peak) in peak_array:
            biases[i, 0], biases[i, 1] = get_gaussian_bias(
                y.squeeze(), x_peak, y_peak)
            i += 1
        # Bias have to be converted to trackspace
        # when going from yspace to trackspace, every pixel becomes 4 pixels * 4 pixels wide.
        # Strategy -  scale your bias by 4
        t_biases = biases * 4

        # Resize peak_array to trackspace ->
        # shape should be (n flies, 2) of float64s
        # get in the center of the 4 pixels by going two pixels in,
        trackframe = (peak_array * 4 + 2).astype(np.float64)

        # Apply bias to increase sub-pixel peak detection accuracy
        trackframewbias = np.add(trackframe, t_biases)
        # visualize_output(trackframe)
        # visualize_output(trackframewbias)
        # This becomes your first trackframe

        vectors = {}
        # Assign points to label
        # Check each new point,
        # If point is in search radius of any point previous index,
        # add to that point's data
        search_radius = 20
        # map of indices
        # If trackstack is not empty, iterate through previous position until close
        # Keys - int 0 to 7 - track name
        # Values - n_frames, ()[x,y] in float64 like ground truth - coordinates
        if bool(trackstack):
            for peak_index in np.arange(8):
                for stack_index in np.arange(8):
                    diff_x = trackframewbias[peak_index, 0] - \
                        trackstack['track' +
                                   str(stack_index)][time_index - 1][0]
                    diff_y = trackframewbias[peak_index, 1] - \
                        trackstack['track' +
                                   str(stack_index)][time_index - 1][1]
                    within_x = abs(diff_x) < search_radius
                    within_y = abs(diff_y) < search_radius
                    if within_x and within_y:
                        trackstack['track' + str(stack_index)] = np.vstack(
                            (trackstack['track' + str(stack_index)],
                             trackframewbias[peak_index, :]))
                        # construct vectors based on previous iteration
                        vectors['track' + str(stack_index)] = [diff_x, diff_y]
        else:
            for track_index in range(8):
                trackstack['track' + str(track_index)
                           ] = np.array([trackframewbias[track_index, :]])
    # output tracks so far
    plt.figure(figsize=(12, 12))
    for track in trackstack.values():
        plt.plot(track[:, 0], track[:, 1], "-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(1024, 0)
    plt.xlim(0, 1024)
    plt.show()
    # Misdetection logic to implement:
    #     if misdetection (number of points is less), predict point of missing one with vector
    #     if duplication, find average of two closest points and reassign
    #     if identity switch, refer back to vector to correct

    # Map your tracks to ground truth tracks

    # Compare to groundtruth using py-motmetrics

    # Report the fraction of frames in which the tracks are correctly assigned correctly.

    # Report the runtime of TensorFlow model prediction time.
    # Report the runtime of tracking time.
    # Confirm tracking time < TensorFlow model prediction time.

    # Additional challenges
    # Implement the peak finding in TensorFlow on the GPU such that the
    # heatmap tensor does not have to be copied back to the CPU when predicting on the images.
    # Implement your tracking algorithm in parallel or on the GPU.


if __name__ == '__main__':
    main()
