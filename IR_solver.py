"""
DSA final project
adi & isabel
"""

import numpy as np
import random
from BuildMatrix import BuildMatrix
from visuals import Visual

class IRDropAnalysis():
    def __init__(self, N, Vdd = 5, threshold = .0001, debug = False): # voltage_src_matrix, current_sink_matrix, resistance_matrix,
         # setup
         self.N = N
         self.Vdd = Vdd
         self.threshold = threshold #value to determine whether a voltage has settled or not, << 1
         self.debug = debug
         # self.init_matrices(voltage_src_matrix, current_sink_matrix, resistance_matrix)


    def init_matrices(self, voltage_src_matrix, current_sink_matrix, resistance_matrix, N, threshold):
        """
        init all matrices
        """
        self.N = N
        self.threshold = threshold
        self.voltage_src_matrix = voltage_src_matrix #boolean array to indicate whether a node is a voltage source or not
        self.resistance_matrix = resistance_matrix #array containing adjacent resistances at every node, form [(u,r,d,l)]
        self.current_sink_matrix = current_sink_matrix # array containing current sink values, structured i,j in an NxN matrix
        self.voltage_matrix = np.ones((self.N, self.N)) * self.Vdd #need to change once we have voltage sources/actual voltage


    def solve_row_based(self, voltage_src_matrix, current_sink_matrix, resistance_matrix, N = -1, threshold=-1):

        self.init_matrices(voltage_src_matrix, current_sink_matrix, resistance_matrix, N = (self.N if N==-1 else N), threshold = (self.threshold if threshold==-1 else threshold))

        threshold_reached = self.voltage_src_matrix
        runs = 0

        while ~(np.all(threshold_reached == 1)):
        # for _ in range(2):
            prev_voltage = self.voltage_matrix.copy()

            for row in range(self.N):
                self.solve_row(row)

                for j in range(self.N): # check threshold reached
                    threshold_reached[row][j] = (abs(self.voltage_matrix[row][j] - prev_voltage[row][j]) < self.threshold)

                    # if abs(self.voltage_matrix[row][j] - prev_voltage[row][j]) < self.threshold:
                    #     threshold_reached[row][j] = 1

            runs += 1
            if (runs % 100 == 0):
                print('completed run ', runs)
                print(self.voltage_matrix)
                temp_inverse = np.invert(threshold_reached)
                print('nodes left to reach threshold: ', np.sum(temp_inverse), '/', self.N*self.N)


        print('total runs row method: ', runs)

        return self.voltage_matrix

    def solve_node_based(self, voltage_src_matrix, current_sink_matrix, resistance_matrix, N = -1, threshold=-1):
        """
        Returns node voltage at every node, by iterating through entire matrix N times until a threshold level of
        precision is reached at all node voltages.
        """

        self.init_matrices(voltage_src_matrix, current_sink_matrix, resistance_matrix, N = (self.N if N==-1 else N),threshold = (self.threshold if threshold==-1 else threshold))


        # set up looping
        threshold_reached = self.voltage_src_matrix
        runs = 0
        # print(voltage_src_matrix)
        while ~(np.all(threshold_reached == 1)):

            # with np.nditer(self.voltage_src_matrix, op_flags=['readwrite'],flags=['multi_index']) as iteratror_obj:
            # compute new voltage for every item in array

            for i in range(self.N):
                for j in range(self.N):
                # for prev_node_voltage in iteratror_obj:
                    # i = iteratror_obj.multi_index[0] # row
                    # j = iteratror_obj.multi_index[1] # col

                    if self.voltage_src_matrix[i][j] == 0: # ensure there is no source at this location

                        new_node_voltage = self.compute_node_voltage(i,j)

                        if abs(self.voltage_matrix[i][j] - new_node_voltage) < self.threshold:
                            threshold_reached[i][j] = 1

                        self.voltage_matrix[i][j] = new_node_voltage

            # Logging to stdout code
            if self.debug:
                runs += 1
                if (runs % 100 == 0):

                    print(self.voltage_matrix)
                    print('completed run ', runs)
                    temp_inverse = np.invert(threshold_reached)
                    print('nodes left to reach threshold: ', np.sum(temp_inverse), '/', self.N*self.N)

        if self.debug:
            print('total runs node method: ', runs)

        return self.voltage_matrix

    def compute_node_voltage(self, i,j):
        """
        Computes and returns a node voltage at a speciifc position (i,j) in matrix, given adjacent resitances & node voltages.
        """
        # set adjacent resistances
        adj_r = list(self.resistance_matrix[i][j]) # makes a copy of list

        # set adjacent voltages
        adj_v = ((self.voltage_matrix[i-1][j]) if i-1>=0 else 0.0,(self.voltage_matrix[i][j+1]) if j+1<self.N else 0.0,(self.voltage_matrix[i+1][j]) if i+1<self.N else 0.0,(self.voltage_matrix[i][j-1]) if j-1>=0 else 0.0)

        #resistances multiplied, helpful for calculations

        r123 = adj_r[1]*adj_r[2]*adj_r[3]
        r023 = adj_r[0]*adj_r[2]*adj_r[3]
        r013 = adj_r[0]*adj_r[1]*adj_r[3]
        r012 = adj_r[0]*adj_r[1]*adj_r[2]

        res = [r123,r023,r013,r012]

        denom = 0
        for val in range(4):
            if adj_v[val] != 0.0: denom+=res[val]

        # run computation based on KVL and KCL
        final_node_voltage = (adj_v[0]*res[0] + adj_v[1]*res[1] + adj_v[2]*res[2] + adj_v[3]*res[3] - self.current_sink_matrix[i][j]*adj_r[0]*adj_r[1]*adj_r[2]*adj_r[3])/denom

        return final_node_voltage

    def solve_row(self, row):
        """ Solve individual row of matrix, for row method """
        # assert(row > 1 or row < self.N - 1); # confirm row isn't first or last row
        u = np.zeros(self.N)
        l = np.zeros(self.N)
        y = np.zeros(self.N)

        d = self.initialize_d(row) #find current from top and bottom, subtract current sink

        #set u[0] to the net conductance at that node
        u[0] = 1/ (self.resistance_matrix[row][0][0]) +  1/ (self.resistance_matrix[row][0][1]) + 1/ (self.resistance_matrix[row][0][2]) + 1/ (self.resistance_matrix[row][0][3])

        for col in range(1,self.N):
            l[col] = -1/self.resistance_matrix[row][col][3] / u[col - 1] # -(left conductance / previuos_node_total_conductance). portion of previous conductance that is connected b/t the current node and the left node
            G = 1/ (self.resistance_matrix[row][col][0]) +  1/ (self.resistance_matrix[row][col][1]) + 1/ (self.resistance_matrix[row][col][2]) + 1/ (self.resistance_matrix[row][col][3]) #total conductance at current node
            u[col] = G + (l[col] * 1/self.resistance_matrix[row][col-1][1]) # conductance at the node + portion of previous conductance times left conductance

        y[0] = d[0]
        for col in range(1,self.N):
            y[col] = d[col] - (l[col] * y[col - 1])

        if not self.voltage_src_matrix[row][self.N-1]:
            self.voltage_matrix[row][self.N-1] = y[self.N - 1] / u[self.N - 1]

        if self.debug:
            print(l)
        # print('l:', l)
        # print('u:', u)
        # print('d:', d)
        # print('y:',y)
        # print('final voltage in row: ', row, ' ', self.voltage_matrix[row][self.N-1])

        for col in reversed(range(self.N-1)): # (int k = N - 2; k >= 0; --k) {
            # node[row * N + k].v = (y[k] + node[row * N + k + 1].v * node[row * N + k].g_r) / u[k];
            if not self.voltage_src_matrix[row][col]:
                self.voltage_matrix[row][col] = (y[col] + self.voltage_matrix[row][col+1]*(1/self.resistance_matrix[row][col][1]))/u[col]

    def initialize_d(self, row):
        """Helper function for row method. Initializes the d matrix by calculating how much current is flowing from the nodes above and below a particular
        node in the row. Returns d, which contains the current yet to be fulfilled by the nodes to the left and right of each node.
            inputs: row, an int
            output: d, a matrix contianing unsatisfied current draw of each node in a row
        """
        d = np.zeros(self.N)

        if row == 0:
            for i in range(self.N):
                # d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) - node[row * N + i].i;
                d[i] = (1/self.resistance_matrix[row][i][2])*self.voltage_matrix[row+1][i] - self.current_sink_matrix[row][i]

        elif row == self.N - 1:
            for i in range(self.N):
                # d[i] = (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].self.voltage_matrix) - node[row * N + i].i;
                d[i] = (1/self.resistance_matrix[row][i][0])*self.voltage_matrix[row-1][i] - self.current_sink_matrix[row][i]

        else:
            for i in range(self.N):
                d[i] = (1/self.resistance_matrix[row][i][2])*self.voltage_matrix[row+1][i] + (1/self.resistance_matrix[row][i][0])*self.voltage_matrix[row-1][i] - self.current_sink_matrix[row][i]
                # d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) +
                #         (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].v) - node[row * N + i].i;

        return d



if __name__ == "__main__":
    pass

    # v,i,r = builder.generate_default(.1, 1)
    #
    # node_method = NodeMethod(size,v,i,r, threshold=.001)
    # voltages = node_method.solve()
    #
    # visualizer = Visual()
    # visualizer.visualize_node_voltage(voltages)
    #
    #
    # print('NodeMethod Output:')
    # print(voltages)
