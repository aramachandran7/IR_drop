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

class IRDropAnalysis():
    def __init__(self, N, voltage_src_matrix, current_sink_matrix, resistance_matrix, Vdd = 5, threshold = .0001):
         # setup
         self.voltage_src_matrix = voltage_src_matrix #boolean array to indicate whether a node is a voltage source or not
         self.resistance_matrix = resistance_matrix #array containing adjacent resistances at every node, form [(u,r,d,l)]
         self.N = N
         self.voltage_matrix = np.ones((self.N, self.N)) * Vdd #need to change once we have voltage sources/actual voltage
         self.current_sink_matrix = current_sink_matrix # array containing current sink values, structured i,j in an NxN matrix
         self.threshold = threshold #value to determine whether a voltage has settled or not, << 1

    def solve_node_based(self):
        """
        Returns node voltage at every node, by iterating through entire matrix N times until a threshold level of
        precision is reached at all node voltages.
        """

        # set up looping
        threshold_reached = self.voltage_src_matrix
        runs = 0

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
            runs += 1
            if (runs % 100 == 0):
                print(self.voltage_matrix)
                print('completed run ', runs)
                temp_inverse = np.invert(threshold_reached)
                print('nodes left to reach threshold: ', np.sum(temp_inverse), '/', self.N*self.N)

        print('total runs: ', runs)

        return self.voltage_matrix

    def compute_node_voltage(self, i,j):
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


    def initialize_d(self, row):
        d = np.zeros(self.N)
        if row == 0:
            for i in range(self.N):
                # d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) - node[row * N + i].i;
                d[i] = (1/self.resistance_matrix[row][i][2])*self.voltage_matrix[row][i] - self.current_sink_matrix[row][i]
        elif row == self.N - 1:
            for i in range(self.N):
                # d[i] = (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].self.voltage_matrix) - node[row * N + i].i;
                d[i] = (1/self.resistance_matrix[row][i][0])*self.voltage_matrix[row][i] - self.current_sink_matrix[row][i]
        else:
            for i in range(self.N):
                d[i] = (1/self.resistance_matrix[row][i][2])*self.voltage_matrix[row][i] + (1/self.resistance_matrix[row][i][0])*self.voltage_matrix[row][i] - self.current_sink_matrix[row][i]
                # d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) +
                #         (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].v) - node[row * N + i].i;

        return d

    def solve_row(self, row):
        # assert(row > 1 or row < self.N - 1); # confirm row isn't first or last row
        u = np.zeros(self.N) # list of doubles. conductances
        l = np.zeros(self.N) #
        y = np.zeros(self.N) #

        # TODO:
        d = self.initialize_d(row) #find current from top and bottom, subtract current sink

        u[0] = 1/ (self.resistance_matrix[row][0][0]) +  1/ (self.resistance_matrix[row][0][1]) + 1/ (self.resistance_matrix[row][0][2]) + 1/ (self.resistance_matrix[row][0][3]) #set u[0] to the conductance at that node
        print('u[0] ', u[0])
        for col in range(1,self.N):
            print('numerator',-1/self.resistance_matrix[row][col][3])
            l[i] = -1/self.resistance_matrix[row][col][3] / u[col - 1] # - (left conductance / previuos_node_total_conductance)
            G = 1/ (self.resistance_matrix[row][col][0]) +  1/ (self.resistance_matrix[row][col][1]) + 1/ (self.resistance_matrix[row][col][2]) + 1/ (self.resistance_matrix[row][col][3]) #total conductance at node
            u[i] = G + (l[col] * 1/self.resistance_matrix[row][col][3]) #

            print("l, u: ", l,u)

        y[0] = d[0]
        for j in range(1,self.N):
            y[j] = d[j] - (l[j] * y[j - 1])

        self.voltage_matrix[row][self.N-1] = y[self.N - 1] / u[self.N - 1]

        for k in reversed(range(self.N-1)): # (int k = N - 2; k >= 0; --k) {
            # node[row * N + k].v = (y[k] + node[row * N + k + 1].v * node[row * N + k].g_r) / u[k];
            self.voltage_matrix[row][k] = (y[k] + self.voltage_matrix[row][k+1]*(1/self.resistance_matrix[row][k][1]))/u[k]


    def solve_row_based(self):
        # iterate_cnt = 0;
        # Net net(self.N) ;
        # double e = 0;
        # double *v = new double[N*N];
        # double *d = new double[N];
        # d=0
        threshold_reached = self.voltage_src_matrix
        # while (iterate_cnt < 100):        # for j in range(N**2): #(int j = 0; j < N*N; ++j)
        #     #     v[j] = net.node[j].v;
        #     v_last = self.voltage_matrix.copy()
        #
        #     for row in range(self.N):#(int i = 0; i < N; ++i) {
        #         SolvingOneRow(row)
        #
        #     # e=mse(net.node,v);
        #     # std::cout <<e
        #     #           << "\n";
        #     # iterate_cnt+= 1
        #     # if (e<1e-8) break;
        #     iterate_cnt += 1

        runs = 0

        # while ~(np.all(threshold_reached == 1)):
        for _ in range(3):
            prev_voltage = self.voltage_matrix.copy()

            for row in range(self.N):
                self.solve_row(row)

                for j in range(self.N): # check threshold reached
                    if abs(self.voltage_matrix[row][j] - prev_voltage[row][j]) < self.threshold:
                        threshold_reached[row][j] = 1

            # Logging to stdout code
            print(self.voltage_matrix)
            runs += 1
            if (runs % 100 == 0):
                print('completed run ', runs)
                temp_inverse = np.invert(threshold_reached)
                print('nodes left to reach threshold: ', np.sum(temp_inverse), '/', self.N*self.N)

        print('total runs: ', runs)

        return self.voltage_matrix


if __name__ == "__main__":

    size = 4
    builder = BuildMatrix(size)
    v,i,r = builder.generate_default()

    ir = IRDropAnalysis(size, v,i,r)

    solved_v = ir.solve_row_based()

    print(solved_v)
    print('\n')
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
