"""
DSA final project
adi & isabel

Node Method pseudocode

Todo:
fix r values
np iterate
threshold setting
var names

Voltage_source = [] # voltage source defined at every node, 0 if no source
I_s = [] # current sink defined at every node
Current_sink = []
PV = [] #previous voltage matrix
V = [] # returns voltage matrix
R = [
    (inf, 10, 3, inf),
    (inf, 5, 9, 9),
    ] #connection bt all adjacent nodes (0,1,2,3)

def node_method_algo(R,vltg_src_matrix, I_s):
    PV = starting voltage value at every node (?)
    k = 0 # iteration_number
    threshold = value much less than 1

    # # else:
    #     k = k+1
    #     for i = 1 to len(Voltage_source):
    #         V[k][i] =
    # if PV - V < threshold: #for every node in the matrix
    #     return

    while(boolean_matrix isn't all true):
        # walk thru every node in voltage matrix, compute new value, set boolean matrix value
        for i in V:
            for j in V:
                new_voltage_at_i_j = compute_new_voltage()
                boolean_matrix[i][j] = (V[i][j] -new_voltage_at_i_j)<threshold
    return V
"""

import numpy as np
import random

class NodeMethod():
    def __init__(self, N, vltg_src_matrix, R, V_est = 5, I_sink = 1):
         # setup
         self.vltg_src_matrix = vltg_src_matrix #boolean array to indicate whether a node is a voltage source or not
         # self.I_s = I_s #boolear array to indicate whether a node is a current sink or not

         self.R = R #array containing adjacent resistances at every node
         self.V = np.ones((2,2)) * V_est #need to change once we have voltage sources/actual voltage
         print('self.v', self.V)
         self.N = N

         self.I_sink = I_sink
         print("init")


    def solve(self):
        VBool = self.vltg_src_matrix

        threshold = .0001
        print(np.all(VBool == 1))
        # for trial in range(50):
        while ~(np.all(VBool == 1)):
            print('here', ~(np.all(VBool == 1)))

            print('calc')
            # compute new voltage for every item in array
            for i in range(self.N):
                for j in range(self.N):
                    if self.vltg_src_matrix[i][j] == 0:
                        print('node', i,j)
                        new_voltage_at_i_j = self.compute_new_voltage(i,j)
                        if self.V[i][j] - new_voltage_at_i_j < threshold:
                            VBool[i][j] = 1

                        self.V[i][j] = new_voltage_at_i_j
            print('V', self.V)
            print('VBool',VBool)

        return self.V

    def compute_new_voltage(self, i,j):

        # set adjacent resistances
        r = list(self.R[i][j]) # makes a copy of list
        print('r',r)

        # set adjacent voltages
        v = ((self.V[i-1][j]) if i-1>=0 else 0.0, (self.V[i][j+1]) if j+1<self.N else 0.0, (self.V[i+1][j]) if i+1<self.N else 0.0, (self.V[i][j-1]) if j-1>=0 else 0.0)
        print('adjusted v', v)
        #resistances multipliedself.
        r123 = r[1]*r[2]*r[3]
        r023 = r[0]*r[2]*r[3]
        r013 = r[0]*r[1]*r[3]
        r012 = r[0]*r[1]*r[2]

        print('current at node',(self.I_s[i][j] ^ 1))
        print(r123,r023,r013,r012)
        res = [r123,r023,r013,r012]

        denom = 0
        for val in range(4):
            print(val)
            if v[val] != 0.0: denom+=res[val]
        print(denom)

        final_node_voltage = (v[0]*res[0] + v[1]*res[1] + v[2]*res[2] + v[3]*res[3] - self.I_sink*((self.vltg_src_matrix[i][j])^1)*r[0]*r[1]*r[2]*r[3])/denom

        return final_node_voltage



def build_array(N, voltage_type, resistance_type, spacing=2, R=5):
    ''' Builds test cases
    Inputs: N: the size of the NxN array
            voltage_type: either random, uniform, corners. Defines the position of the voltage sources
            spacing: for 'uniform' type, indicates how often there is a voltage source
    Outputs: 4 arguments, N, vltg_src_matrix, R
    '''

    NULL_R = 1.0 # Null resistance
    DEV = 1.0 # deviation for gaussian distribution
    DEV_SMALL = R/10.0
    # handle voltage_type

    if voltage_type == 'random':
        pass

    elif voltage_type == 'uniform':
        spacing += 1
        vltg_src_matrix = np.zeros((N,N))
        counter = 0
        for index in range(N**2):
            if counter == 0:
                # compute index, set value to 1
                vltg_src_matrix[(index//N)][index%N] = 1

            counter = (counter + 1)% spacing

        # array = [[]]
        # a = np.zeros(spacing-1)
        # a = np.insert(a, 0,1)
        # array = np.tile(a, N//spacing)
        # print('a:',a)
        # print('array:', array)
        # if N % spacing != 0:
        #     array = np.append(array, a[0:N%spacing])
        # prev_row = array
        # for i in range(1,N):
        #     print(i)
        #     next_row = np.roll(prev_row, 1)
        #     print('next row:', next_row, '=====')
        #     array = np.vstack((array, next_row)) # append?
        #     print(array)
        #     prev_row = next_row
        #
        # vltg_src_matrix = array

    elif voltage_type == 'corners':
        # generate Voltage source matrix vltg_src_matrix for corner type
        vltg_src_matrix = np.zeros((N,N))
        # vltg_src_matrix = np.tile([0 for x in range(N)], [N,1])
        vltg_src_matrix[0][0] = 1
        vltg_src_matrix[-1][-1] = 1
        vltg_src_matrix[0][-1] = 1
        vltg_src_matrix[-1][0] = 1

        # generate Resistance matrix

    # handle resistance_type

    if resistance_type == 'uniform':
        # run with all resistances being the same
        r_arr = []
        for row in range(N):
            r_arr.append([])
            for col in range(N):
                r_arr[row].append((NULL_R if (row == 0) else R, NULL_R if ((col+1)%N==0) else R, NULL_R if ((row+1)%N==0) else R, NULL_R if (col == 0) else R))

        resistance_matrix = np.array(r_arr)

    elif resistance_type == 'gaussian':
        # run with all resistances being a gaussian distribution around R
        r_arr = gaussian_resistance(N, R, NULL_R, DEV, DEV_SMALL_BOOLEAN=False)

        resistance_matrix = np.array(r_arr)

    elif resistance_type == 'cluster':

        r_arr = gaussian_resistance(N, R, NULL_R, DEV, DEV_SMALL_BOOLEAN=True)

        # generate random cluster corners & sizes based on normal distribution around optimal cluster size & number

        # general cluster metrics
        cluster_sidelength = .2 #  * N cluster sidelength in terms of N
        total_cluster_area = .33 # *N**2 area of total nodes covered by clusters
        LOW_R = R/3.0

        cluster_number = int(total_cluster_area/(cluster_sidelength**2)) # number of clusters
        print('cluster_number', cluster_number)

        clusters = []
        for i in range(cluster_number):
            side_length = int(np.random.normal(cluster_sidelength*N, cluster_sidelength*N*.4))
            if side_length == 0:
                side_length += 1

            # generate and check top left corner
            corner = (random.randint(0,N-1), random.randint(0,N-1))

            while overlap(corner, side_length, clusters, N): # if theres overlap return true
                corner = (random.randint(0,N-1), random.randint(0,N-1))

            clusters.append((corner,side_length))

        print('clusters:\n', clusters)
        visualize_clusters(N, clusters)
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



    return N, vltg_src_matrix, resistance_matrix

def gaussian_resistance(N, R,NULL_R, DEV, DEV_SMALL_BOOLEAN=False):

    if DEV_SMALL_BOOLEAN:
        DEV /= 3.0

    r_arr = []
    for row in range(N):
        r_arr.append([])
        for col in range(N):
            r_arr[row].append((
                NULL_R if (row == 0) else r_arr[row-1][col][2],
                NULL_R if ((col+1)%N==0) else np.random.normal(R, DEV),
                NULL_R if ((row+1)%N==0) else np.random.normal(R, DEV),
                NULL_R if (col == 0) else r_arr[row][col-1][1]
            ))

    return r_arr

def visualize_clusters(N, clusters):
    arr = np.zeros((N,N))

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

def overlap(corner, side_length, clusters, N):
    # return true if there is any of (overlap with another cluster at all, or out of bounds at all), false otherwise
    if (corner[0] + side_length>= N) or (corner[1]+side_length >= N):
        return True

    for cluster in clusters:
        if corner == cluster[0]:
            return True
        if (corner[0] + side_length) >= (cluster[0][0]-1) and (corner[1]+side_length) >= (cluster[0][1]-1):
            return True
        return False

if __name__ == "__main__":

    N, voltage_source_matrix, resistance_matrix = build_array(N=10, voltage_type='uniform', resistance_type='cluster', spacing=3)
    print('VSOURCEMATRIX:')
    print(voltage_source_matrix)
    print(resistance_matrix)
    # print("-------------------------------------")
    #
    # res = 5 # 5 ohms default resistance
    # arr_size = 2
    # Vdd = 5
    # vltg_src_matrix = np.array([[1, 0], [0, 1]])
    # # I_s = np.array([[0, 1], [1, 0]])
    # R = np.array([[(1.0, res, res, 1.0), (1.0, 1.0, res, res)], [(res, res, 1.0, 1.0), (res, 1.0, 1.0, res)]])
    #
    # t = NodeMethod(arr_size, vltg_src_matrix, R, V_est = Vdd)
    # print(t.solve())
    #
    #
    # print("testing node_method with 3x3 array")
    # print("-------------------------------------")
