"""
DSA final project
adi & isabel

Node Method pseudocode


Voltage_source = [] # voltage source defined at every node, 0 if no source
I_s = [] # current sink defined at every node
Current_sink = []
PV = [] #previous voltage matrix
V = [] # returns voltage matrix
R = [
    (inf, 10, 3, inf),
    (inf, 5, 9, 9),
    ] #connection bt all adjacent nodes (0,1,2,3)

def node_method_algo(R,V_s, I_s):
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

class NodeMethod(Object):
    def __init__(self, N, V_s, I_s, R, V_est):
         # setup
         self.V_s = V_s #boolean array to indicate whether a node is a voltage source or not
         self.I_s = I_s #boolear array to indicate whether a node is a current sink or not

         self.R = R #array containing adjacent resistances at every node
         self.V = np.ones(N) * V_est
         self.N = N

         self.I_sink = 1


    def solve(self):
        VBool = np.zeroes(self.N)
        threshold = .1
        while !(np.all(self.VBool == 1)):
            # compute new voltage for every item in array
            for i in self.N:
                for j in self.N:
                    if self.V_s[i][j] == 0:
                        new_voltage_at_i_j = compute_new_voltage(i,j)
                        if self.V[i][j] - new_voltage_at_i_j < threshold:
                            VBool[i][j] = 1

                        self.V[i][j] = new_voltage_at_i_j

        return self.V

    def compute_new_voltage(self, i,j):

        # set adjacent resistances
        r = list(self.R[i][j]) # makes a copy of list
        for val in r:
            if val == float('inf'): val = 1.0

        r = tuple(r)

        # set adjacent voltages
        v = ((self.V[i+1][j]) if i+1<self.N else 0.0, (self.V[i][j+1]) if j+1<self.N else 0.0, (self.V[i-1][j]) if i-1>=0 else 0.0, (self.V[i][j-1]) if j-1>=0 else 0.0)

        #resistances multipliedself.
        r234 = r[1]*r[2]*r[3]
        r134 = r[0]*r[2]*r[3]
        r124 = r[0]*r[1]*r[3]
        r123 = r[0]*r[1]*r[2]

        final_node_voltage = (v[0]*r234 + v[1]*r134 + v[2]*r124 + v[3]*r123 - self.I_sink*self.I_s[i][j]*r[0]*r[1]*r[2]*r[3])/(r234+r134+r124+r123)

        return final_node_voltage

if "__name__" == "__main__":
    print("testing node_method")
