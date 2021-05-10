"""
test.py
Implement timeit to test node and row method against eachother.
"""

import pytest
import timeit
import matplotlib.pyplot as plt
from numpy import arange


from IR_solver import IRDropAnalysis
from BuildMatrix import BuildMatrix
from visuals import Visual


SMALLEST_MAT = 10


def compare(max_size=25, max_threshold=.01, case_type='default'):
    # determine average time to reach threshold from
    max_size += 1
    # generating test cases
    test_cases_1, test_cases_inf = generate_size_test_cases(max_size=max_size, case_type=case_type)

    t_test_cases_1, t_test_cases_inf, thresholds = generate_threshold_test_cases(max_threshold)


    node_method_results = []
    ir = IRDropAnalysis(3)
    # run timeit
    for test_case in test_cases_1:
        t = timeit.Timer('run = ir.solve_node_based(*test_case)', 'from IR_solver import IRDropAnalysis', globals = locals())
        node_method_results.append(t.timeit(10))

    threshold_results = []
    for test_case in t_test_cases_1:
        t = timeit.Timer('run = ir.solve_node_based(*test_case)', 'from IR_solver import IRDropAnalysis', globals = locals())
        threshold_results.append(t.timeit(10))

    plot_results(node_method_results, thresholds, threshold_results, max_size)



def generate_threshold_test_cases(max_threshold, case_type = "default"):

    # generate spaced thresholds to test
    thresholds = arange(0 + max_threshold/50, max_threshold + max_threshold/50, max_threshold/50)

    MATRIX_SIZE = 20
    test_cases_1 = []
    test_cases_inf = []

    # walk through all thresholds, creating args

    builder = BuildMatrix(MATRIX_SIZE)

    for t in thresholds:
        v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_uniform, builder.build_current_dist, builder.build_resistance_cluster)
        test_cases_1.append((v,i,r_1, MATRIX_SIZE, t))
        test_cases_inf.append((v,i,r_inf, MATRIX_SIZE, t))

    return test_cases_1, test_cases_inf, thresholds

def generate_size_test_cases(max_size, case_type = "default"):
    test_cases_1 = []
    test_cases_inf = []

    if case_type == 'default':
        builder = BuildMatrix(3)
        for size in range(SMALLEST_MAT,max_size):
            builder.N = size
            v,i,r_1, r_inf = builder.generate_default()
            test_cases_1.append((v,i,r_1, size))
            test_cases_inf.append((v,i,r_inf, size))


    elif case_type == 'random': # TODO fix
        builder = BuildMatrix(3)
        for size in range(SMALLEST_MAT,max_size):
            builder.N = size
            v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_random, builder.build_current_random, builder.build_resistance_gaussian)
            test_cases_1.append((v,i,r_1, size))
            test_cases_inf.append((v,i,r_inf, size))

    elif case_type == 'complex':
        builder = BuildMatrix(3)
        for size in range(SMALLEST_MAT,max_size):
            builder.N = size
            v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_uniform, builder.build_current_dist, builder.build_resistance_cluster)
            test_cases_1.append((v,i,r_1, size))
            test_cases_inf.append((v,i,r_inf, size))

    return test_cases_1, test_cases_inf

def plot_results(results, thresholds, threshold_results,max_size):
    indices = [i**2 for i in range(SMALLEST_MAT, max_size)]

    plt.plot(indices, results, 'rs', label="results")
    # plt.plot(x,default_results,'g^', label="default sort results")
    # plt.yscale('log')
    plt.title('Time to solve entire matrix via Node Method')
    plt.xlabel('Total Nodes in Matrix')
    plt.ylabel('Time to solve entire matrix (seconds)')
    plt.legend(loc="upper left")
    plt.show(block=True)

    time_per_node = []
    for i in range(len(results)):
        time_per_node.append(results[i]/indices[i])

    plt.plot(indices, time_per_node, 'rs', label="results")
    plt.title('Computation Time per Node via Node Method')
    plt.xlabel('Total Nodes in Matrix')
    plt.ylabel('Computation Time per Node (seconds)')
    plt.legend(loc="upper left")
    plt.show(block=True)

    plt.plot(thresholds, threshold_results, 'rs', label="results")
    plt.title('Time to solve entire matrix as a function of Error Threshold Setting')
    plt.xlabel('Error Threshold Setting')
    plt.ylabel('Time to solve entire matrix (seconds)')
    plt.legend(loc="upper right")
    plt.show(block=True)

    # plt.plot(x,default_results,'g^', label="default sort results")
    # # plt.yscale('log')
    # plt.title('default sorting algorithm results')
    # plt.xlabel('Test case index')
    # plt.ylabel('Time to sort')
    # plt.legend(loc="upper left")
    # plt.show(block=True)


def general_test():
    size = 20
    builder = BuildMatrix(size)
    ir = IRDropAnalysis(size, debug=True)
    vis = Visual()
    v,i,r_1, r_inf = builder.generate_default()
    # v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_random, builder.build_current_random, builder.build_resistance_gaussian)
    # v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_corners, builder.build_current_dist, builder.build_resistance_cluster)

    vis.visualize_source_voltage(v)
    solved_v_node = ir.solve_node_based(v, i, r_1)

    vis.visualize_node_voltage(solved_v_node)



    # vis.visualize_node_voltage(mat = solved_v_node)
    # vis.visualize_node_voltage(mat = solved_v_row)



    # print('row:\n', solved_v_row)
    # print('node:\n', solved_v_node)
    print('\n')

if __name__ == "__main__":
    # compare(max_size=40,  max_threshold=.01, case_type='complex')
    general_test()
