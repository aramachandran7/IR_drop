def build_array(N, voltage_type, resistance_type, current_type, spacing=2, R=5, ):
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
        vltg_src_matrix = np.random.randint(2, size=(N,N))

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

    if current_type == 'uniform':

        current_sink_matrix = (self.vltg_src_matrix[i][j])^1)

    if current_type == 'uniform'



    return N, vltg_src_matrix, resistance_matrix, I_sink_matrix

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
