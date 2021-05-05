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
from visuals import Visual

class NodeMethod():
    def __init__(self, N, vltg_src_matrix, current_sink_matrix, resistance_matrix, Vdd = 5, threshold = .0001):
         # setup
         self.vltg_src_matrix = vltg_src_matrix #boolean array to indicate whether a node is a voltage source or not
         self.resistance_matrix = resistance_matrix #array containing adjacent resistances at every node, form [(u,r,d,l)]
         self.N = N
         self.voltage_matrix = np.ones((self.N, self.N)) * Vdd #need to change once we have voltage sources/actual voltage
         self.current_sink_matrix = current_sink_matrix # array containing current sink values, structured i,j in an NxN matrix
         self.threshold = threshold

    def solve(self):
        """
        Returns node voltage at every node, by iterating through entire matrix N times until a threshold level of
        precision is reached at all node voltages.
        """

        # setup looping
        threshold_reached = self.vltg_src_matrix
        runs = 0

        while ~(np.all(threshold_reached == 1)):

            # with np.nditer(self.vltg_src_matrix, op_flags=['readwrite'],flags=['multi_index']) as iteratror_obj:
            # compute new voltage for every item in array

            for i in range(self.N):
                for j in range(self.N):
                # for prev_node_voltage in iteratror_obj:
                    # i = iteratror_obj.multi_index[0] # row
                    # j = iteratror_obj.multi_index[1] # col

                    if self.vltg_src_matrix[i][j] == 0: # ensure there is no source at this location

                        new_node_voltage = self.compute_new_voltage(i,j)

                        if abs(self.voltage_matrix[i][j] - new_node_voltage) < self.threshold:
                            threshold_reached[i][j] = 1

                        self.voltage_matrix[i][j] = new_node_voltage

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

    visualizer = Visual()
    visualizer.visualize_node_voltage(voltages)


    print('NodeMethod Output:')
    print(voltages)
