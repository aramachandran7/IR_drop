"""
row_method_solver.py
"""
#TODO: make a class var with attributes



def InitArrayD(row):
    d = np.zeros(N)
    if row == 0:
        for i in range(N):
            # d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) - node[row * N + i].i;
            d[i] = (1/r[row][i][2])*v[row][i] - i[row][i]
    elif row == N - 1:
        for i in range(N):
            # d[i] = (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].v) - node[row * N + i].i;
            d[i] = (1/r[row][i][0])*v[row][i] - i[row][i]
    else:
        for i in range(N):
            d[i] = (1/r[row][i][2])*v[row][i] + (1/r[row][i][0])*v[row][i] - i[row][i]
            # d[i] = (node[(row + 1) * N + i].g_d * node[(row + 1) * N + i].v) +
            #         (node[(row - 1) * N + i].g_u * node[(row - 1) * N + i].v) - node[row * N + i].i;

    return d

def SolvingOneRow(v,i,r, N, row):
    assert(row > 1 or row < N - 1); # confirm row isn't first or last row
    u = np.zeros(N) # list of doubles. conductances
    l = np.zeros(N) #
    y = np.zeros(N) #

    # TODO:
    d = InitArrayD(row); #find current from top and bottom, subtract current sink

    u[0] = 1.0/r[row][col]; #set u[0] to the conductance at that node

    for col in range(1,N):
        l[i] = -1/r[row][col][3] / u[col - 1]; # - (left conductance / previuos_node_total_conductance)
        G = 1/ (r[row][col][0]) +  1/ (r[row][col][1]) + 1/ (r[row][col][2]) + 1/ (r[row][col][3]) #total conductance at node
        u[i] = G + (l[col] * 1/r[row][col][3]) #

    y[0] = d[0]
    for j in range(1,N):
        y[j] = d[j] - (l[j] * y[j - 1])

    v[row][N-1] = y[N - 1] / u[N - 1]

    for k in reversed(range(N-1)): # (int k = N - 2; k >= 0; --k) {
        # node[row * N + k].v = (y[k] + node[row * N + k + 1].v * node[row * N + k].g_r) / u[k];
        v[row][k] = (y[k] + v[node][k+1]*(1/r[row][k][1]))/u[k]

    return v


def solve():
    iterate_cnt = 0;
    Net net(N) ;
    # double e = 0;
    # double *v = new double[N*N];
    # double *d = new double[N];
    voltage_array = []
    d=0
    while (iterate_cnt < 100):        # for j in range(N**2): #(int j = 0; j < N*N; ++j)
        #     v[j] = net.node[j].v;
        v_last = voltage_array

        for row in range(N):#(int i = 0; i < N; ++i) {
            SolvingOneRow(row)
        for i in v:

        e=mse(net.node,v);
        std::cout <<e
                  << "\n";
        iterate_cnt+= 1
        if (e<1e-8) break;
