"""
Unit and regression test for the Shapes class in the ratar.encoding module of the ratar package.
"""

from flatten_dict import flatten
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from ratar.auxiliary import MoleculeLoader
from ratar.encoding import Shapes


@pytest.mark.parametrize('mol_file1, mol_file2', [
    ('AAK1_4wsq_altA_chainA.mol2', 'AAK1_4wsq_altA_chainB.mol2')
])
def test_shapes_eq(mol_file1, mol_file2):
    """
    Test __eq__ function for Shapes class.

    Parameters
    ----------
    mol_file1 : str
        Name of file containing the structure for molecule A.
    mol_file2 : str
        Name of file containing the structure for molecule B.

    """

    molecule_path1 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file1
    molecule_path2 = Path(__name__).parent / 'ratar' / 'tests' / 'data' / mol_file2

    molecule_loader1 = MoleculeLoader(molecule_path1)
    molecule_loader2 = MoleculeLoader(molecule_path2)

    shapes1 = Shapes()
    shapes2 = Shapes()
    shapes3 = Shapes()

    shapes1.from_molecule(molecule_loader1.get_first_molecule())
    shapes2.from_molecule(molecule_loader1.get_first_molecule())
    shapes3.from_molecule(molecule_loader2.get_first_molecule())

    assert (shapes1 == shapes2)
    assert not (shapes1 == shapes3)


@pytest.mark.parametrize('distances, moment1, moment2, moment3', [
    (
        pd.DataFrame([
            [1.0, 2.0],
            [1.0, 2.0],
            [4.0, 2.0],
            [4.0, 2.0],
            [5.0, 2.0]
        ], columns='dist_ref1 dist_ref2'.split()),
        pd.Series([3.0, 2.0], index='dist_ref1 dist_ref2'.split()),
        pd.Series([1.6733, 0.0000], index='dist_ref1 dist_ref2'.split()),
        pd.Series([-1.0627, 0.0000], index='dist_ref1 dist_ref2'.split())
    )
])
def test_calc_moments(distances, moment1, moment2, moment3):

    shapes = Shapes()
    moments = shapes._calc_moments(distances)

    assert np.isclose(moments['m1'], moment1, rtol=1e-04).all()
    assert np.isclose(moments['m2'], moment2, rtol=1e-04).all()
    assert np.isclose(moments['m3'], moment3, rtol=1e-04).all()


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
    ),
    (
        pd.Series([1, 2, 3]),
        pd.DataFrame([[1, 2, 4, 9], [1, 2, 5, 9]]),
        pd.Series([1, 2, 4, 9])
    )
])
def test_calc_nearest_point(point, points, nearest_point_ref):

    shapes = Shapes()
    nearest_point = shapes._calc_nearest_point(point, points, 1)

    assert all(np.isclose(nearest_point, nearest_point_ref, rtol=1e-04))


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
    assert all(np.isclose(distances_to_point, distances_to_point_ref, rtol=1e-04))


@pytest.mark.parametrize('ref_points, dist, moments_ref', [
    (
        [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])],
        [pd.Series([3, 4, 4]), pd.Series([1, 1, 1])],
        pd.DataFrame(
            [
                [3.6667, 0.4714, -0.4200],
                [1.0, 0.0, 0.0]
            ],
            index='dist_ref1 dist_ref2'.split(),
            columns='m1 m2 m3'.split()
        )
    )
])
def test_get_shape_dict(ref_points, dist, moments_ref):

    shapes = Shapes()
    shape_dict = shapes._get_shape_dict(ref_points, dist)

    assert all(shape_dict['ref_points'].index == [f'ref{i+1}' for i, j in enumerate(ref_points)])
    assert all(shape_dict['dist'].columns == [f'dist_ref{i+1}' for i, j in enumerate(dist)])
    assert all(shape_dict['moments'].index == [f'dist_ref{i+1}' for i, j in enumerate(dist)])
    assert all(shape_dict['moments'].columns == 'm1 m2 m3'.split())

    assert np.isclose(shape_dict['moments'], moments_ref, rtol=1e-04).all(axis=None)


@pytest.mark.parametrize('coord_origin, coord_point_a, coord_point_b, scaled_by', [
    (
        pd.Series([0, 0, 0, 3]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 1, 0]),
        'mean_norm'
    ),
    (
        pd.Series([0, 0]),
        pd.Series([1, 0]),
        pd.Series([0, 1]),
        'mean_norm'
    ),
    (
        pd.Series([0, 0, 0]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 1, 0]),
        'xxx'
    ),
])
def test_calc_scaled_3d_cross_product_exceptions(coord_origin, coord_point_a, coord_point_b, scaled_by):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_scaled_3d_cross_product(coord_origin, coord_point_a, coord_point_b, scaled_by)


@pytest.mark.parametrize('coord_origin, coord_point_a, coord_point_b, scaled_by, scaled_3d_cross_product_ref', [
    (
        pd.Series([0, 0, 0]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 1, 0]),
        'mean_norm',
        pd.Series([0, 0, 1])
    ),
    (
        pd.Series([0, 0, 0, 3]),
        pd.Series([1, 0, 0, 3]),
        pd.Series([0, 1, 0, 3]),
        'mean_norm',
        pd.Series([0, 0, 1])
    ),
    (
        pd.Series([0, 0, 0]),
        pd.Series([1, 0, 0]),
        pd.Series([0, 2, 0]),
        'mean_norm',
        pd.Series([0, 0, 1.5])
    ),
    (
        pd.Series([0, 0, 0, 0]),
        pd.Series([1, 0, 0, 1]),
        pd.Series([0, 2, 0, 1]),
        'half_norm_a',
        pd.Series([0, 0, 0.7071])
    ),
    (
        pd.Series([5.8840, 12.5871, 43.3804]),
        pd.Series([2.0888, 16.4158, 54.4161]),
        pd.Series([9.5988, 10.512, 35.0883]),
        'half_norm_a',
        pd.Series([2.1283, 16.6303, 40.6861])
    )
])
def test_calc_scaled_3d_cross_product(coord_origin, coord_point_a, coord_point_b, scaled_by, scaled_3d_cross_product_ref):

    shapes = Shapes()

    scaled_3d_cross_product = shapes._calc_scaled_3d_cross_product(coord_origin, coord_point_a, coord_point_b, scaled_by)

    assert np.isclose(scaled_3d_cross_product, scaled_3d_cross_product_ref, rtol=1e-04).all()


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])  # Must be at least 4 points
    ),
    (
        pd.DataFrame([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])  # Points must be in 3D
    )
])
def test_calc_shape_3dim_usr_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_3dim_usr(points)


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])  # Must be at least 4 points
    ),
    (
        pd.DataFrame([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])  # Points must be in 3D
    )
])
def test_calc_shape_3dim_csr_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_3dim_csr(points)


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])  # Must be at least 5 points
    ),
    (
        pd.DataFrame([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 0]
        ])  # Points must be in 4D
    ),
    (
        pd.DataFrame([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]
        ])  # Points must be in 4D
    )
])
def test_calc_shape_4dim_electroshape_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_4dim_electroshape(points)


@pytest.mark.parametrize('points', [
    (
        pd.DataFrame([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])  # Must be at least 7 points
    ),
    (
        pd.DataFrame([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0]
        ])  # Points must be in 4D
    ),
    (
        pd.DataFrame([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [5, 0, 0, 0, 0],
            [6, 0, 0, 0, 0]
        ])  # Vectors are linear dependent
    )
])
def test_calc_shape_6dim_ratar1_exceptions(points):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._calc_shape_6dim_ratar1(points)


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
            [0.14286, 0.42857, 2.28571],
            [0, 0, 1],
            [0, 0, 10],
            [0, 2, 0]
        ], index='ref1 ref2 ref3 ref4'.split(), columns='x y z'.split()
        )
    )
])
def test_calc_shape_3dim_usr(points, ref_points):

    shapes = Shapes()

    shape_3dim_usr = shapes._calc_shape_3dim_usr(points)

    assert shape_3dim_usr['ref_points'].shape == ref_points.shape
    assert np.isclose(shape_3dim_usr['ref_points'], ref_points, rtol=1e-04).all(axis=None)


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
            [0.14286, 0.42857, 2.28571],
            [0, 0, 10],
            [0, 2, 0],
            [-3.6883, -0.0626, 2.1875]
        ], index='ref1 ref2 ref3 ref4'.split(), columns='x y z'.split()
        )
    )
])
def test_calc_shape_3dim_csr(points, ref_points):

    shapes = Shapes()

    shape_3dim_usr = shapes._calc_shape_3dim_csr(points)

    assert shape_3dim_usr['ref_points'].shape == ref_points.shape
    assert np.isclose(shape_3dim_usr['ref_points'], ref_points, rtol=1e-04).all(axis=None)


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
            [0.14286, 0.14286, 2.2857, 1.4286],
            [0, 0, 10, 10],
            [1, 0, 0, 0],
            [1.4206, 5.7647, 2.4135, 10],
            [1.4206, 5.7647, 2.4135, -1]
        ], index='ref1 ref2 ref3 ref4 ref5'.split(), columns='x y z z1'.split()
        )
    )
])
def test_calc_shape_4dim_electroshape(points, ref_points):

    shapes = Shapes()

    shape_3dim_usr = shapes._calc_shape_4dim_electroshape(points)

    assert shape_3dim_usr['ref_points'].shape == ref_points.shape
    assert np.isclose(shape_3dim_usr['ref_points'], ref_points, rtol=1e-04).all(axis=None)


@pytest.mark.parametrize('points, ref_points', [
    (
        pd.DataFrame([
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 1, 0, 2, 0, 0],
            [1, 0, 1, 0, 2, 0],
            [1, 1, 1, 0, 0, 2]
        ], columns='x y z z1 z2 z3'.split()
        ),
        pd.DataFrame([
            [0.5714, 0.4286, 0.4286, 0.4286, 0.4286, 0.4286],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 2],
            [1, 1, 0, 2, 0, 0],
            [1, 1, 0, 2, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0]  # TODO linear dependent reference points! Bad!
        ], index='ref1 ref2 ref3 ref4 ref5 ref6 ref7'.split(), columns='x y z z1 z2 z3'.split()
        )
    )
])
def test_calc_shape_6dim_ratar1(points, ref_points):

    shapes = Shapes()

    shape_6dim_ratar1 = shapes._calc_shape_6dim_ratar1(points)

    assert shape_6dim_ratar1['ref_points'].shape == ref_points.shape

    for index, row in shape_6dim_ratar1['ref_points'].iterrows():
        assert np.isclose(row, ref_points.loc[index], rtol=1e-04).all(axis=None)


@pytest.mark.parametrize('points_df', [
    (
        pd.DataFrame([
            [0, 0]
        ])
    ),  # Too few dimensions.
    (
        pd.DataFrame([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    ),  # Too few points for 3D.
    (
            pd.DataFrame([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
    ),  # Too few points for 4D.
    (
            pd.DataFrame([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
    ),  # Too few points for 6D.
    (
            pd.DataFrame([
                [0, 0, 0, 0, 0, 0, 0]
            ])
    )  # Too many dimensions
])
def ttest_get_shape_by_method_exceptions(points_df):

    shapes = Shapes()

    with pytest.raises(ValueError):
        shapes._get_shape_by_method(points_df)


@pytest.mark.parametrize('points_df, shape_keys', [
    (
        pd.DataFrame([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        '3Dusr 3Dcsr'.split()
    ),
    (
        pd.DataFrame([
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        '4Delectroshape'.split()
    ),
    (
        pd.DataFrame([
            [1, 1, 1, 1, 1, 1],
            [2, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ]),
        '6Dratar1'.split()
    )
])
def test_get_shape_by_method(points_df, shape_keys):

    shapes = Shapes()
    shape = shapes._get_shape_by_method(points_df)

    assert list(shape.keys()) == shape_keys


@pytest.mark.parametrize('filename, keys_ref', [
    (
            'AAK1_4wsq_altA_chainA_reduced.mol2',
            f'ca/no/3Dusr ca/no/3Dcsr ca/z1/4Delectroshape ca/z123/6Dratar1 '
            f'pca/no/3Dusr pca/no/3Dcsr pca/z1/4Delectroshape pca/z123/6Dratar1 '
            f'pc/no/3Dusr pc/no/3Dcsr pc/z1/4Delectroshape pc/z123/6Dratar1'.split()
    )
])
def test_from_molecule(filename, keys_ref):
    """
    Test if points are correctly extracted from representatives of a molecule.

    Parameters
    ----------
    filename : str
        Name of molecule file.
    keys_ref : list of str
        Flattened keys for different types of representatives and physicochemical properties.
    """

    # Load molecule
    molecule_path = Path(__name__).parent / 'ratar' / 'tests' / 'data' / filename
    molecule_loader = MoleculeLoader(molecule_path)
    molecule = molecule_loader.get_first_molecule()

    # Set points
    shapes = Shapes()
    shapes.from_molecule(molecule)

    shapes_flat = flatten(shapes.data, reducer='path')

    keys_ref = sorted(keys_ref)
    keys_calc = sorted(set(['/'.join(i.split('/')[:-1]) for i in shapes_flat.keys()]))

    assert keys_calc == keys_ref
