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
specific current sinks can be specified and randomly assigned based off a uniform or custom allocation
    i.e. [3 amps, 2 amps, .5 amps, 6 amps]
         [20, 10, 5, 2]

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
    def __init__(self, N, vltg_src_matrix, R, I_sink, V_est = 5):
         # setup
         self.vltg_src_matrix = vltg_src_matrix #boolean array to indicate whether a node is a voltage source or not
         # self.I_s = I_s #boolear array to indicate whether a node is a current sink or not

         self.R = R #array containing adjacent resistances at every node
         self.V = np.ones((2,2)) * V_est #need to change once we have voltage sources/actual voltage
         print('self.v', self.V)
         self.N = N

         self.I_sink = I_sink # array containing current sink values, structured i,j in an NxN matrix
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

        print(r123,r023,r013,r012)
        res = [r123,r023,r013,r012]

        denom = 0
        for val in range(4):
            print(val)
            if v[val] != 0.0: denom+=res[val]
        print(denom)

        final_node_voltage = (v[0]*res[0] + v[1]*res[1] + v[2]*res[2] + v[3]*res[3] - self.I_sink[i][j]*r[0]*r[1]*r[2]*r[3])/denom

        return final_node_voltage




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
