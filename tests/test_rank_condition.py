import numpy as np
import pytest

from conditional_inference.rank_condition import RankCondition


def test_common_methods():
    model = RankCondition(np.arange(3), np.identity(3))
    results = model.fit()
    results.conf_int()
    results.summary()
    results.point_plot()


class TestRankCondition:
    model = RankCondition(np.arange(3), np.diag([1, 2, 3]) ** 2)

    @pytest.mark.parametrize("column", ("x0", "x1", "x2"))
    @pytest.mark.parametrize("beta", (0, 0.005))
    def test_get_marginal_distribution(self, column, beta):
        dist = self.model.get_marginal_distribution(column, beta=beta)

        if column == "x0":
            expected_truncation_set = [-np.inf, 1]
            expected_scale = 1
            expected_y = 0
        elif column == "x1":
            expected_truncation_set = [0, 2]
            expected_scale = 2
            expected_y = 1
        else:
            expected_truncation_set = [1, np.inf]
            expected_scale = 3
            expected_y = 2

        np.testing.assert_almost_equal(
            dist.truncnorm_kwargs["truncation_set"][0], expected_truncation_set
        )
        assert dist.truncnorm_kwargs["scale"] == expected_scale
        assert dist.y == expected_y

        assert (dist.projection_interval == (-np.inf, np.inf)) == (beta == 0)

    @pytest.mark.parametrize("rank", ([0], [0, 1], [0, 1, 2]))
    def test_truncation_set(self, rank):
        dist = self.model.get_marginal_distribution("x2", rank)

        if rank == [0]:
            expected_value = [[1, np.inf]]
        elif rank == [0, 1]:
            expected_value = [[0, 1], [1, np.inf]]
        else:
            expected_value = [[-np.inf, 0], [0, 1], [1, np.inf]]

        np.testing.assert_almost_equal(
            dist.truncnorm_kwargs["truncation_set"], expected_value
        )
