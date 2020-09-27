"""
Space to keep utils unrelated to UI but necessary for running Task_A or finding peaks
"""
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter, uniform_filter


def find_peaks(heatmap):
    """ We apply the "mask" method similar to github/erdogant/findpeaks
    directly"""
    # set threshold
    threshold = 0
    # create the footprint
    # a neighborhood of a pixel connected to its 8 neighbors
    neighborhood = generate_binary_structure(2, 2)

    # apply local maximum filter mask, max pixels in neighborhood set to 1
    local_max = maximum_filter(heatmap, footprint=neighborhood) == heatmap

    # Create mask of the background
    background = (heatmap <= threshold)

    # Erode background to prevent artifact when subtracting it from local max
    eroded = binary_erosion(background, structure=neighborhood, border_value=1)

    # XOR to remove background from the local max mask
    detected_peaks = local_max ^ eroded

    # Return obtain final mask of only peaks
    # Shape of heatmap but peaks are 1s & else is 0s
    return detected_peaks
