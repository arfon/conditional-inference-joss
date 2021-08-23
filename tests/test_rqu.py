import numpy as np
import pytest

from conditional_inference.rqu import RQU


n_policies = 3
mean = np.arange(n_policies)
cov = np.identity(n_policies)
rqu = RQU(mean, cov)


class TestRQU:
    # tests rarely invoked pieces of code in RQU

    def test_projection_quantile(self):
        # test when alpha == 0
        # see simulation tests for more rigorous tests when alpha != 0
        rqu = RQU(np.arange(3), np.identity(3))
        assert rqu.compute_projection_quantile(alpha=0) == np.inf

    def test_s_V_condition(self):
        # this condition applies with cov(x_i,x_j) == var(x_i)
        # when it fails, the truncation set is empty
        # see paper for mathematical detail
        with pytest.raises(ValueError):
            RQU(np.arange(2), np.ones((2, 2))).get_distribution(rank=1)

    @pytest.mark.parametrize("rank", ["invalid_rank", "floor", "ceil"])
    def test_rank_arguments(self, rank):
        def get_truncation_interval(dist):
            truncation_set = dist.truncnorm_kwargs["truncation_set"]
            a, b = list(zip(*truncation_set))
            return min(a), max(b)

        rqu = RQU(np.arange(2), np.identity(2))

        if rank not in ("floor", "ceil"):
            with pytest.raises(ValueError):
                rqu.get_distributions(rank=rank)
            return

        with pytest.raises(ValueError):
            rqu.get_distribution(rank=rank)

        dists = rqu.get_distributions(rank=rank)
        truncation_sets = [get_truncation_interval(dist) for dist in dists]
        if rank == "floor":
            assert truncation_sets == [(-np.inf, np.inf), (0, np.inf)]
        else:  # rank == "ceil"
            assert truncation_sets == [(-np.inf, 1), (-np.inf, np.inf)]

    def test_get_distribution_default(self):
        assert rqu.get_distribution().y == n_policies - 1


@pytest.fixture(scope="module", params=[{}, dict(beta=.005), dict(projection=True)])
def results(request):
    return rqu.fit(**request.param)


class TestResults:
    def test_conf_int(self, results):
        results.conf_int()

    def test_summary(self, results):
        results.summary()

    def test_point_plots(self, results):
        results.point_plot()
