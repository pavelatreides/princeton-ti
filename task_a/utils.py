"""
Space to keep utils unrelated to UI but necessary for running Task_A or finding peaks
"""

class MessageContainer():
    """A message container that contains what you wish to return.
    This class was created to easily serialize
    and identify data passed through sockets"""

    def __init__(self, returnpeaks=False, image=None, path=None):
        # add super
        # should be 'coordinates' or 'heatmap'
        self.returnpeaks = returnpeaks
        # framenumber in the video
        self.path = path
        self.image = image
