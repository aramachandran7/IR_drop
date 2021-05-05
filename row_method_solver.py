"""
row_method_solver.py
"""
#TODO: make a class var with attributes

def InitArrayD(row):
    if row == 0:
        for i in range(N):
            d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) - node[row * N + i].i;
    elif row == N - 1:
        for i in range(N):
            d[i] = (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].v) - node[row * N + i].i;
    else:
        for i in range(N):
            d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) +
                    (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].v) - node[row * N + i].i;
    return d

def SolvingOneRow(v,i,r, N, row):
    assert(row > 1 or row < N - 1); # confirm row isn't first or last row
    u = np.zeros(N) # list of doubles
    l = np.zeros(N)
    d = np.zeros(N)
    y = np.zeros(N)

    # TODO:
    InitArrayD(node, d, row); #find current from top and bottom, subtract current sink


    for i in range(N):
        d[i]

    u[0] = 1.0/r[row][col]; #set u[0] to the conductance at that node

    for col in range(1,N):
        l[i] = -1/r[row][col][3] / u[col - 1]; # - (left conductance / previuos_node_total_conductance)
        G = 1/ (r[row][col][0]) +  1/ (r[row][col][1]) + 1/ (r[row][col][2]) + 1/ (r[row][col][3]) #total conductance at node
        u[i] = G + (l[col] * 1/r[row][col][3]) #

    y[0] = d[0]
    for j in range(1,N):
        y[j] = d[j] - (l[j] * y[j - 1]);

    node[row * N + N - 1].v = y[N - 1] / u[N - 1];
    for (int k = N - 2; k >= 0; --k) {
        node[row * N + k].v = (y[k] + node[row * N + k + 1].v * node[row * N + k].g_r) / u[k];
