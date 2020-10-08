import numpy as np


def print_test_train_params(test, train, params=None, dims=None):
    n = len(test)
    if dims == None:
        dims = range(1, n + 1)
    best_index = np.argmax(test)
    if params == None:
        print('Dimension Testscore  Trainingsscore')
        for i, dim in enumerate(dims):
            print(dim, '       ', np.round(test[i], 4), '     ', np.round(train[i], 4))
        print(f'Bester Testscore: {test[best_index]} erreicht mit Dimension {dims[best_index]}')
    else:
        print('Dimension Testscore  Trainingsscore beste Parameter')
        for i, dim in enumerate(dims):
            print(dim, '       ', np.round(test[i], 4), '     ',
                  np.round(train[i], 4), '   ', params[i])
        print(
            f'Bester Testscore: {test[best_index]} erreicht mit Dimension {dims[best_index]} und Parametern {params[best_index]}')
