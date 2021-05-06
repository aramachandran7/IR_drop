"""
Class to build matrices
"""
import numpy as np
import random

class BuildMatrix():
    ''' Builds test cases
    Inputs: N: the size of the NxN array
            voltage_type: either random, uniform, corners. Defines the position of the voltage sources
            spacing: for 'uniform' type, indicates how often there is a voltage source
    Outputs: 4 arguments, N, vltg_src_matrix, R
    '''
    def __init__(self, N):
        self.N = N
        self.NULL_R = np.inf#1.0 # Null resistance
        self.DEV = 1.0 # deviation for gaussian distribution

    def generate_default(self, R=5, current_val=1):
        """Create default voltage, current, and resistance array.
        Inputs: R, resistance in ohms
                current_val, value of each current sink
        """
        v = self.build_voltage_corners()
        i = self.build_current_uniform(v, current_val)
        r = self.build_resistance_uniform(R)
        return v, i, r


    def build_voltage_random(self):
        """Creates a voltage matrix filled with random 1s and 0s"""
        vltg_src_matrix = np.random.randint(2, size=(self.N,self.N), dtype=np.bool)
        return vltg_src_matrix

    def build_voltage_uniform(self, spacing=2):
        """Creates a voltage matrix that is a grid with with voltage nodes spaced apart as vertically and horizontally (not on the diagonal)
        Input: spacing, an int defining the space in between two nodes
        Ex:
        Spacing 1, N = 6
        [[0 0 0 0 0 0]
         [1 0 1 0 1 0]
         [0 0 0 0 0 0]
         [1 0 1 0 1 0]
         [0 0 0 0 0 0]
         [1 0 1 0 1 0]]
        Spacing 3, N = 7
        [[0 0 0 0 0 0 0]
         [0 1 0 0 0 1 0]
         [0 0 1 0 0 0 1]
         [0 0 0 1 0 0 0]
         [0 0 0 0 0 0 0]
         [0 1 0 0 0 1 0]
         [0 0 1 0 0 0 1]]
        """
        spacing += 1 #makes it so that spacing is what you'd expect
        vltg_src_matrix = np.zeros((self.N,self.N), dtype=np.bool)
        counter = 0
        for index in range(self.N**2):
            if counter == 0:
                # compute index, set value to 1
                if (index//self.N) % spacing != 0:
                    vltg_src_matrix[(index//self.N)][index%self.N] = 1

            counter = (counter + 1)% spacing
        return vltg_src_matrix

    def build_voltage_corners(self):
        """Generate voltage source matrix vltg_src_matrix for corner type.
        Each voltage source is located at each of the four corners of the matrix.
        """
        vltg_src_matrix = np.zeros((self.N,self.N), dtype=np.bool)
        # vltg_src_matrix = np.tile([0 for x in range(N)], [N,1])
        vltg_src_matrix[0][0] = 1
        vltg_src_matrix[-1][-1] = 1
        vltg_src_matrix[0][-1] = 1
        vltg_src_matrix[-1][0] = 1

        return vltg_src_matrix

    def build_voltage_center(self):
        """Crete voltage array where there is one central voltage node.
        Ideal for smaller matrices, or those where there is not a lot of current draw."""
        vltg_src_matrix = np.zeros((self.N,self.N), dtype=np.bool)
        # vltg_src_matrix = np.tile([0 for x in range(N)], [N,1])
        vltg_src_matrix[self.N//2][self.N//2] = 1

        return vltg_src_matrix


    def build_resistance_uniform(self, R=5):
        """Create a resistance matrix with all resistances being the same
        Inputs: R, resistance in ohms
        """
        r_arr = []
        for row in range(self.N):
            r_arr.append([])
            for col in range(self.N):
                r_arr[row].append((self.NULL_R if (row == 0) else R, self.NULL_R if ((col+1)%self.N==0) else R, self.NULL_R if ((row+1)%self.N==0) else R, self.NULL_R if (col == 0) else R))

        resistance_matrix = np.array(r_arr)
        return resistance_matrix


    def build_resistance_gaussian(self, R = 5):
        """Build a resistance matrix with all resistances being a gaussian distribution around R
        Inputs: R, resistance in ohms, the average value of each of the edges
        """
        r_arr = self.normal_resistance_distribution(R, DEV_SMALL_BOOLEAN=False)

        return np.array(r_arr)

    def build_resistance_cluster(self, R=5):
        """ Builds a cluster of resistances


        """
        r_arr = self.normal_resistance_distribution(R, DEV_SMALL_BOOLEAN=True)

        # generate random cluster corners & sizes based on normal distribution around optimal cluster size & number

        # general cluster metrics
        cluster_sidelength = .2 #  * N cluster sidelength in terms of N
        total_cluster_area = .33 # *N**2 area of total nodes covered by clusters
        LOW_R = R/3.0

        cluster_number = int(total_cluster_area/(cluster_sidelength**2)) # number of clusters
        print('cluster_number', cluster_number)

        clusters = []
        for i in range(cluster_number):
            side_length = int(np.random.normal(cluster_sidelength*self.N, cluster_sidelength*self.N*.4))
            if side_length == 0:
                side_length += 1

            # generate and check top left corner
            corner = (random.randint(0,self.N-1), random.randint(0,self.N-1))

            while self.overlap(corner, side_length, clusters): # if theres overlap return true
                corner = (random.randint(0,self.N-1), random.randint(0,self.N-1))

            clusters.append((corner,side_length))

        print('clusters:\n', clusters)
        self.visualize_clusters(clusters)
        # walk through all clusters, and alter resistance_matrix
        for cluster in clusters:
            # walk through all nodes, setting appropriate values to low resistances
            corner = cluster[0]
            side_length = cluster[1]

            row_min = corner[0]
            row_max = corner[0]+side_length
            col_min = corner[1]
            col_max = corner[1]+side_length

            for row in range(row_min,row_max+1):
                for col in range(col_min, col_max+1):
                    prev = r_arr[row][col] # original tuple

                    # set new tuple based on bounding box defined by corner and side_length
                    r_arr[row][col] = (
                            prev[0] if (row == row_min) else LOW_R,
                            prev[1] if (col==col_max) else LOW_R,
                            prev[2] if (row==row_max) else LOW_R,
                            prev[3] if (col == col_min) else LOW_R
                            )


            resistance_matrix = np.array(r_arr)

        return resistance_matrix

    def build_current_uniform(self, vltg_src_matrix, current_val=1):
        """Create a current source matrix with equivalent current values"""
        sinks = np.invert(vltg_src_matrix)
        current_sink_matrix = sinks * current_val
        return current_sink_matrix

    def build_current_rand(self, vltg_src_matrix, max):
        """Create a current source matrix with random current values between 0 and the maximum
        Input: vltg_src_matrix, the voltage source matrix for the graph
               max: the maximum current value
        """
        sinks = np.invert(vltg_src_matrix)
        rands = np.random.rand(self.N, self.N)
        current_sink_matrix = sinks * rands * max
        return current_sink_matrix

    def build_current_dist(self, vltg_src_matrix, currents : list, distribution: list):
        """Create current source matrix from current values given in currents with distribution
        Inputs: currents, a list of each current value that should be present in the final matrix, in amps
                distribution, a list containing the percentage value that each current value should be present in the matrix
        """
        sinks = np.invert(vltg_src_matrix)

        list_currents = []

        for i, current in enumerate(currents):
            list_currents += [current for x in range(int(distribution[i]*(self.N**2)))]
        np.random.shuffle(list_currents)
        current_matrix = np.resize(list_currents, (self.N, self.N))

        final =  current_matrix*sinks

        return final

    def normal_resistance_distribution(self, R, DEV_SMALL_BOOLEAN=False):
        if DEV_SMALL_BOOLEAN:
            self.DEV /= 3.0

        r_arr = []
        for row in range(self.N):
            r_arr.append([])
            for col in range(self.N):
                r_arr[row].append((
                    self.NULL_R if (row == 0) else r_arr[row-1][col][2],
                    self.NULL_R if ((col+1)%self.N==0) else np.random.normal(R, self.DEV),
                    self.NULL_R if ((row+1)%self.N==0) else np.random.normal(R, self.DEV),
                    self.NULL_R if (col == 0) else r_arr[row][col-1][1]
                ))

        self.DEV *= 3.0
        return r_arr

    def visualize_clusters(self,clusters):
        arr = np.zeros((self.N,self.N))

        for cluster in clusters:
            # walk through all nodes, setting appropriate values to low resistances
            corner = cluster[0]
            side_length = cluster[1]

            row_min = corner[0]
            row_max = corner[0]+side_length
            col_min = corner[1]
            col_max = corner[1]+side_length

            for row in range(row_min,row_max+1):
                for col in range(col_min, col_max+1):
                    arr[row][col] = 1

        print('CLUSTERS, MAPPED\n', arr)

    def overlap(self, corner, side_length, clusters):
        # return true if there is any of (overlap with another cluster at all, or out of bounds at all), false otherwise
        if (corner[0] + side_length>= self.N) or (corner[1]+side_length >= self.N):
            return True

        for cluster in clusters:
            if corner == cluster[0]:
                return True
            if (corner[0] + side_length) >= (cluster[0][0]-1) and (corner[1]+side_length) >= (cluster[0][1]-1): # checks for one way overlap, not the other way i'm a genius
                return True
            return False


if __name__ == "__main__":
    builder = BuildMatrix(10)
    v = builder.build_voltage_random()
    i = builder.build_current_dist(v, [1,2,3], [.25,.25,.5])
    print('v\n', v)
    print('i\n', i)
    # v,i,r = builder.generate_default()
    # print('v\n', v)
    # print('i\n', i)
    # print('r\n', r)
