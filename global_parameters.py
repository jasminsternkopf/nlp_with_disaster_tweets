DATA_DIR = "data"
MAX_DIM = 15
LSI_DIMENSION_MULTIPLIER = 50
PARAMETERS_SVC = {
    'Classifier__C': (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,
                      10, 50, 100),
    'Classifier__kernel': ('linear', 'rbf', 'poly', 'sigmoid')
}
