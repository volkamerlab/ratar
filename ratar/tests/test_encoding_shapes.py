"""
Unit and regression test for the Shapes class in the ratar.encoding module of the ratar package.
"""

import sys

import pandas as pd

from flatten_dict import flatten
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Shapes


@pytest.mark.parametrize('distances, moment1, moment2, moment3', [
    (
        pd.DataFrame([
            [1.0, 2.0],
            [1.0, 2.0],
            [4.0, 2.0],
            [4.0, 2.0],
            [5.0, 2.0]
        ], columns='dist_c1 dist_c2'.split()),
        pd.Series([3.0, 2.0], index='dist_c1 dist_c2'.split()),
        pd.Series([1.6733, 0.0000], index='dist_c1 dist_c2'.split()),
        pd.Series([-1.0627, 0.0000], index='dist_c1 dist_c2'.split())
    )
])
def test_calc_moments(distances, moment1, moment2, moment3):

    shapes = Shapes()
    moments = shapes._calc_moments(distances)

    print(moments)

    assert all((moments['m1'] - moment1) < 0.0001)
    assert all((moments['m2'] - moment2) < 0.0001)
    assert all((moments['m3'] - moment3) < 0.0001)


@pytest.mark.parametrize('nested_dict, key_order, flat_keys_before, flat_keys_after', [
    (
        {'H': {'O': {'L': {'A': 'halo', 'E': 'helo'}}}},
        [0, 3, 2, 1],
        ['H/O/L/A', 'H/O/L/E'],
        ['H/A/L/O', 'H/E/L/O']
    )
])
def test_reorder_nested_dict_keys(nested_dict, key_order, flat_keys_before, flat_keys_after):

    shapes = Shapes()
    reordered_dict = shapes._reorder_nested_dict_keys(nested_dict, key_order)

    assert sorted(list(flatten(nested_dict, reducer='path').keys())) == flat_keys_before
    assert sorted(list(flatten(reordered_dict, reducer='path').keys())) == flat_keys_after


@pytest.mark.parametrize('point, points, nearest_point_ref', [
    (
        pd.Series([1, 2, 3]),
        pd.DataFrame([[1, 2, 4], [1, 2, 5]]),
        pd.Series([1, 2, 4])
    )
])
def test_calc_nearest_point(point, points, nearest_point_ref):

    shapes = Shapes()
    nearest_point = shapes._calc_nearest_point(point, points, 1)

    assert all(abs(nearest_point - nearest_point_ref) < 0.0001)


@pytest.mark.parametrize('points, ref_point, distances_to_point_ref', [
    (
        pd.DataFrame([[1, 2, 4], [1, 2, 5]]),
        pd.Series([1, 2, 3]),
        pd.Series([1.0, 2.0])
    )
])
def test_calc_distances_to_point(points, ref_point, distances_to_point_ref):

    shapes = Shapes()
    distances_to_point = shapes._calc_distances_to_point(points, ref_point)

    assert all(distances_to_point.index == distances_to_point_ref.index)
    assert all(abs(distances_to_point - distances_to_point_ref) < 0.0001)


@pytest.mark.parametrize('ref_points, dist, moments_ref', [
    (
        [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])],
        [pd.Series([3, 4, 4]), pd.Series([1, 1, 1])],
        pd.DataFrame(
            [
                [3.6667, 0.4714, -0.4200],
                [1.0, 0.0, 0.0]
            ],
            index='dist_c1 dist_c2'.split(),
            columns='m1 m2 m3'.split()
        )
    )
])
def test_get_shape_dict(ref_points, dist, moments_ref):

    shapes = Shapes()
    shape_dict = shapes._get_shape_dict(ref_points, dist)

    assert all(shape_dict['ref_points'].index == [f'c{i+1}' for i, j in enumerate(ref_points)])
    assert all(shape_dict['dist'].columns == [f'dist_c{i+1}' for i, j in enumerate(dist)])
    assert all(shape_dict['moments'].index == [f'dist_c{i+1}' for i, j in enumerate(dist)])
    assert all(shape_dict['moments'].columns == 'm1 m2 m3'.split())

    assert (abs(shape_dict['moments'] - moments_ref) < 0.0001).all(axis=None)


@pytest.mark.parametrize('coord_origin, coord_point_a, coord_point_b', [
    (
        pd.Series([0, 0, 0, 3]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 1, 0])
    ),
    (
        pd.Series([0, 0]),
        pd.Series([1, 0]),
        pd.Series([0, 1])
    )
])
def test_calc_adjusted_3d_cross_product_exceptions(coord_origin, coord_point_a, coord_point_b):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_adjusted_3d_cross_product(coord_origin, coord_point_a, coord_point_b)


@pytest.mark.parametrize('coord_origin, coord_point_a, coord_point_b, adjusted_3d_cross_product_ref', [
    (
        pd.Series([0, 0, 0]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 1, 0]),
        pd.Series([0, 0, 1])
    ),
    (
        pd.Series([0, 0, 0, 3]),
        pd.Series([1, 0, 0, 3]),
        pd.Series([0, 1, 0, 3]),
        pd.Series([0, 0, 1])
    ),
    (
        pd.Series([0, 0, 0]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 2, 0]),
        pd.Series([0, 0, 1.5])
    )
])
def test_calc_adjusted_3d_cross_product(coord_origin, coord_point_a, coord_point_b, adjusted_3d_cross_product_ref):

    shapes = Shapes()

    adjusted_3d_cross_product = shapes._calc_adjusted_3d_cross_product(coord_origin, coord_point_a, coord_point_b)

    assert (abs(adjusted_3d_cross_product - adjusted_3d_cross_product_ref) < 0.0001).all()


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # Must be at least 4 points
    ),
    (
        pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # Points must be in 3D
    )
])
def test_calc_shape_3dim_usr_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_3dim_usr(points)


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # Must be at least 4 points
    ),
    (
        pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # Points must be in 3D
    )
])
def test_calc_shape_3dim_csr_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_3dim_csr(points)


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # Must be at least 5 points
    ),
    (
        pd.DataFrame([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0]])  # Points must be in 4D
    ),
    (
        pd.DataFrame([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1]])  # Points must be in 4D
    )
])
def test_calc_shape_4dim_electroshape_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_4dim_electroshape(points)


@pytest.mark.parametrize('points, ref_points', [
    (
        pd.DataFrame([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [0, 0, 10],
            [0, 0, 5],
        ], columns='x y z'.split()
        ),
        pd.DataFrame([
            [0.1429, 0.4286, 2.2857],
            [0, 0, 1],
            [0, 0, 10],
            [0, 2, 0]
        ], index='c1 c2 c3 c4'.split(), columns='x y z'.split()
        )
    )
])
def test_calc_shape_3dim_usr(points, ref_points):

    shapes = Shapes()

    shape_3dim_usr = shapes._calc_shape_3dim_usr(points)

    assert shape_3dim_usr['ref_points'].shape == ref_points.shape
    assert (abs(shape_3dim_usr['ref_points'] - ref_points) < 0.0001).all(axis=None)


@pytest.mark.parametrize('points, ref_points', [
    (
        pd.DataFrame([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 2, 0],
            [0, 0, 10],
            [0, 0, 5],
        ], columns='x y z'.split()
        ),
        pd.DataFrame([
            [0.1429, 0.4286, 2.2857],
            [0, 0, 10],
            [0, 2, 0],
            [-3.6883, -0.0626, 2.1875]
        ], index='c1 c2 c3 c4'.split(), columns='x y z'.split()
        )
    )
])
def test_calc_shape_3dim_csr(points, ref_points):

    shapes = Shapes()

    shape_3dim_usr = shapes._calc_shape_3dim_csr(points)

    assert shape_3dim_usr['ref_points'].shape == ref_points.shape
    assert (abs(shape_3dim_usr['ref_points'] - ref_points) < 0.0001).all(axis=None)


@pytest.mark.parametrize('points, ref_points', [
    (
        pd.DataFrame([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 10, 10],
            [0, 0, 5, -1],
        ], columns='x y z z1'.split()
        ),
        pd.DataFrame([
            [0.1429, 0.1429, 2.2857, 1.4286],
            [0, 0, 10, 10],
            [1, 0, 0, 0],
            [-3.6883, -0.0626, 2.1875],
            [0, 0, 0]
        ], index='c1 c2 c3 c4 c5'.split(), columns='x y z z1'.split()
        )
    )
])
def test_calc_shape_4dim_electroshape(points, ref_points):

    shapes = Shapes()

    shape_3dim_usr = shapes._calc_shape_4dim_electroshape(points)

    assert shape_3dim_usr['ref_points'].shape == ref_points.shape
    assert (abs(shape_3dim_usr['ref_points'] - ref_points) < 0.0002).all(axis=None)

