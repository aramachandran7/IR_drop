"""
DSA final project
adi & isabel

todo:
think about AC
implement row method
think about (colorful!) visualizations
"""

import numpy as np
import random
from BuildMatrix import BuildMatrix

class NodeMethod():
    def __init__(self, N, vltg_src_matrix, current_sink_matrix, resistance_matrix, Vdd = 5, threshold = .0001):
         # setup
         self.vltg_src_matrix = vltg_src_matrix #boolean array to indicate whether a node is a voltage source or not
         self.resistance_matrix = resistance_matrix #array containing adjacent resistances at every node
         self.N = N
         self.voltage_matrix = np.ones((self.N, self.N)) * Vdd #need to change once we have voltage sources/actual voltage
         self.current_sink_matrix = current_sink_matrix # array containing current sink values, structured i,j in an NxN matrix
         self.threshold = threshold

    def solve(self):
        """
        Returns node voltage at every node, by iterating through entire matrix N times until a threshold level of
        precision is reached at all node voltages.
        """

        threshold_reached = self.vltg_src_matrix
        # for trial in range(50):
        runs = 0
        while ~(np.all(threshold_reached == 1)):

            # compute new voltage for every item in array
            for i in range(self.N):
                for j in range(self.N):
                    if self.vltg_src_matrix[i][j] == 0:
                        new_voltage_at_i_j = self.compute_new_voltage(i,j)
                        if self.voltage_matrix[i][j] - new_voltage_at_i_j < self.threshold:
                            threshold_reached[i][j] = 1

                        self.voltage_matrix[i][j] = new_voltage_at_i_j

            # Logging to stdout code
            runs += 1
            if (runs % 100 == 0):
                print('completed run ', runs)
                temp_inverse = np.invert(threshold_reached)
                print('nodes left to reach threshold: ', np.sum(temp_inverse), '/', self.N*self.N)

        print('total runs: ', runs)

        return self.voltage_matrix

    def compute_new_voltage(self, i,j):
        """
        computes and returns a node voltage at a speciifc position (i,j) in matrix, given adjacent resitances & node voltages.
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




if __name__ == "__main__":

    size = 10
    builder = BuildMatrix(size)
    v,i,r = builder.generate_default(.1, 1)


    node_method = NodeMethod(size,v,i,r, threshold=.001)
    voltages = node_method.solve()


    print('NodeMethod Output:')
    print(voltages)
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
