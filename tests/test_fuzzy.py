import pytest
from src.context_aware.fuzzy import (
    fuzzy_triangular,
    fuzzy_gaussian,
    fuzzy_trapezoidal,
    apply_fuzzy,
    add_with_diminishing,
    subtract_with_diminishing,
)


def test_fuzzy_triangular():
    # Center = 0.5, width = 0.2
    assert fuzzy_triangular(0.5, 0.5, 0.2) == 1.0
    assert fuzzy_triangular(0.3, 0.5, 0.2) == pytest.approx(0.0)
    assert fuzzy_triangular(0.7, 0.5, 0.2) == pytest.approx(0.0)
    assert fuzzy_triangular(0.4, 0.5, 0.2) == pytest.approx(0.5)
    assert fuzzy_triangular(0.1, 0.5, 0.2) == 0.0


def test_fuzzy_gaussian():
    # Peak at center
    assert fuzzy_gaussian(10.0, 10.0, 2.0) == 1.0
    # Symmetric decay
    val_left = fuzzy_gaussian(8.0, 10.0, 2.0)
    val_right = fuzzy_gaussian(12.0, 10.0, 2.0)
    assert val_left == pytest.approx(val_right)
    assert 0.0 < val_left < 1.0


def test_fuzzy_trapezoidal():
    # a=10, b=20, c=30, d=40
    assert fuzzy_trapezoidal(5, 10, 20, 30, 40) == 0.0
    assert fuzzy_trapezoidal(15, 10, 20, 30, 40) == pytest.approx(0.5)
    assert fuzzy_trapezoidal(25, 10, 20, 30, 40) == 1.0
    assert fuzzy_trapezoidal(35, 10, 20, 30, 40) == pytest.approx(0.5)
    assert fuzzy_trapezoidal(45, 10, 20, 30, 40) == 0.0


def test_apply_fuzzy_configured_features():
    # Cold temperature around 0.2
    score_cold = apply_fuzzy("temp_cold", 0.20)
    assert score_cold == pytest.approx(1.0)

    # Hot temperature around 0.9
    score_hot = apply_fuzzy("temp_hot", 0.90)
    assert score_hot == pytest.approx(1.0)

    # Unknown feature
    assert apply_fuzzy("non_existent_feature", 0.5) == 0.0


def test_diminishing_returns():
    # Starting from 0.5, adding 0.4 should not exceed 1.0
    res = add_with_diminishing(0.5, 0.4, factor=0.7)
    assert 0.5 < res <= 1.0

    # Subtracting should not drop below 0.0
    sub = subtract_with_diminishing(0.2, 0.5, factor=0.7)
    assert 0.0 <= sub < 0.2
