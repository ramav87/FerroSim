import numpy as np

class Lattice:
    """Lattice class containing attributes for polarization, lattice position, and nearest neighbors
        The function calls available are:

        setPosition(): sets the (x,y) location of the object
        getPosition(): returns the position of the object

        setP(): sets the polarization value
        getP(): gets the polarization value

    """

    def __init__(self, pval, xyval, time_steps):
        # Pass the polarization value along with (x,y) tuple for location of atom
        self.P = np.zeros(shape=(2, time_steps)) #(x,y components of P)
        self.P[:,0] = pval
        self.position = xyval
        self.neighbor = []
        self.neighborIndex = []
        self.distance = []

    def getPosition(self):
        return self.position

    def getP(self, time):
        # Returns the polarization at t=time
        return self.P[:,time]

    def getNeighbors(self):
        # Returns a list containing the nearest neighbors, set by using updateNeighbors
        return self.neighbor

    def getNeighborIndex(self):
        # Returns a list containing the nearest neighbors, set by using updateNeighbors
        return self.neighborIndex

    def getDistance(self):
        # Return a list containing the distances of the nearest neighbors. The order is the same as the order of the neighbors
        return self.distance

    def setPosition(self, xyval):
        # Sets the (x,y) position of the object. Pass a two-element tuple.
        self.position = xyval

    def setP(self, time, pval):
        # Sets the (x,y) position of the object. Pass a two-element tuple (time, pval)
        # where pval = (px,py)

        self.P[:,time] = pval

    def updateNeighbors(self, nTuples):
        """Provide a list of tuples of nearest neighbors (nTuples)
        This function will just store them in the object as a list, nothing more"""
        for index, neighbor_pos in enumerate(nTuples):
            self.neighbor.append(neighbor_pos)

    def updateNeighborIndex(self, nIndex):
        """Provide a list of tuples of indieces for the nearest neighbors (nIndex)
        This function will just store them in the object as a list, nothing more"""
        # for index, neighbor_pos in enumerate(nIndex):
        self.neighborIndex.append(nIndex)

    def updateDistance(self, nDist):
        """Provide a an array with the distances to the nearest neighbors (nTuplesDist)
        This function will just store them in a list, nothing more.
        The order is the same as for the NeighborsIndex and Neighbors"""
        # for index, distance_val in enumerate(nDists):
        self.distance.append(nDist)