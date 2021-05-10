"""
test.py
Implement timeit to test node and row method against eachother.
"""

import pytest
import timeit
import matplotlib.pyplot as plt


from IR_solver import IRDropAnalysis
from BuildMatrix import BuildMatrix
from visuals import Visual
test_cases = [


]


def compare(max_size=25):
    # determine average time to reach threshold from

    # generating test cases
    test_cases_1, test_cases_inf = generate_test_cases(max_size=max_size, case_type='default')

    node_method_results = []
    ir = IRDropAnalysis(3   )
    # run timeit
    for test_case in test_cases_1:
        t = timeit.Timer('ir.solve_node_based(*test_case)', 'from IR_solver import IRDropAnalysis', globals = locals())
        node_method_results.append(t.timeit(10))

    plot_results(node_method_results, max_size)




def generate_test_cases(max_size, case_type = "default"):
    test_cases_1 = []
    test_cases_inf = []

    if case_type == 'default':
        builder = BuildMatrix(3)
        for size in range(2,max_size):
            builder.N = size
            v,i,r_1, r_inf = builder.generate_default()
            test_cases_1.append((v,i,r_1, size))
            test_cases_inf.append((v,i,r_inf, size))


    elif case_type == 'random':
        pass

    return test_cases_1, test_cases_inf

def plot_results(results, max_size):
    indices = [i**2 for i in range(2, max_size)]

    plt.plot(indices, results, 'rs', label="results")
    # plt.plot(x,default_results,'g^', label="default sort results")
    # plt.yscale('log')
    plt.title('Node Method Compute times')
    plt.xlabel('Matrix size')
    plt.ylabel('Time to compute')
    plt.legend(loc="upper left")
    plt.show(block=True)

    # TODO: time per node?

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
    ir = IRDropAnalysis(size)
    vis = Visual()
    # v,i,r_1, r_inf = builder.generate_default()
    # v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_random, builder.build_current_random, builder.build_resistance_gaussian)
    v,i,r_1, r_inf = builder.generate_custom(builder.build_voltage_uniform, builder.build_current_dist, builder.build_resistance_cluster)

    vis.visualize_source_voltage(v)


    solved_v_row = ir.solve_row_based(v, i ,r_inf)

    solved_v_node = ir.solve_node_based(v, i, r_1)

    vis.visualize_node_voltage(mat = solved_v_node)


    print('row:\n', solved_v_row)
    print('node:\n', solved_v_node)
    print('\n')

if __name__ == "__main__":
    compare(max_size=10)

    # general_test()
