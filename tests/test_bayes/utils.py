from conditional_inference.bayes import Improper


def run_common_methods(results):
    # test that you can run all the common methods without error
    results.conf_int()
    results.expected_wasserstein_distance()
    results.likelihood()
    if not isinstance(results.model, Improper):
        results.line_plot(0)
    results.point_plot()
    results.rank_matrix_plot()
    results.reconstruction_point_plot()
    results.summary()
