"""
DSA final project
adi & isabel

todo:
think about AC
implement row method
"""

import numpy as np
import random
from BuildMatrix import BuildMatrix

class NodeMethod():
    def __init__(self, N, vltg_src_matrix, I_sink, R, V_est = 5, threshold = .0001):
         # setup
         self.vltg_src_matrix = vltg_src_matrix #boolean array to indicate whether a node is a voltage source or not
         self.R = R #array containing adjacent resistances at every node
         self.N = N
         self.V = np.ones((self.N, self.N)) * V_est #need to change once we have voltage sources/actual voltage
         self.I_sink = I_sink # array containing current sink values, structured i,j in an NxN matrix
         self.threshold = threshold

    def solve(self):
        VBool = self.vltg_src_matrix
        # for trial in range(50):
        runs = 0
        while ~(np.all(VBool == 1)):
            # print('here', ~(np.all(VBool == 1)))
            runs += 1
            if (runs % 100 == 0):
                print('runs: ', runs)
                temp_inverse = np.invert(VBool)
                print('nodes left to reach threshold: ', np.sum(temp_inverse), '/', self.N*self.N)

            # compute new voltage for every item in array
            for i in range(self.N):
                for j in range(self.N):
                    if self.vltg_src_matrix[i][j] == 0:
                        new_voltage_at_i_j = self.compute_new_voltage(i,j)
                        if self.V[i][j] - new_voltage_at_i_j < self.threshold:
                            VBool[i][j] = 1

                        self.V[i][j] = new_voltage_at_i_j

        print('total runs: ', runs)
        return self.V

    def compute_new_voltage(self, i,j):

        # set adjacent resistances
        r = list(self.R[i][j]) # makes a copy of list

        # set adjacent voltages
        v1 = (self.V[i-1][j]) if i-1>=0 else 0.0
        # print('HERE', self.V[i][j+1])
        v2 = (self.V[i][j+1]) if j+1<self.N else 0.0
        v3 = (self.V[i+1][j]) if i+1<self.N else 0.0
        v4 = (self.V[i][j-1]) if j-1>=0 else 0.0
        v = (v1,v2,v3,v4)
        #resistances multipliedself.
        r123 = r[1]*r[2]*r[3]
        r023 = r[0]*r[2]*r[3]
        r013 = r[0]*r[1]*r[3]
        r012 = r[0]*r[1]*r[2]

        res = [r123,r023,r013,r012]

        denom = 0
        for val in range(4):
            if v[val] != 0.0: denom+=res[val]

        final_node_voltage = (v[0]*res[0] + v[1]*res[1] + v[2]*res[2] + v[3]*res[3] - self.I_sink[i][j]*r[0]*r[1]*r[2]*r[3])/denom

        return final_node_voltage




if __name__ == "__main__":

    size = 50
    builder = BuildMatrix(size)
    v,i,r = builder.generate_default(.1, 1)


    n = NodeMethod(size,v,i,r, threshold=.001)
    voltages = n.solve()


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
