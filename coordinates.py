
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def convert_cart_to_spherical(coords):
    '''
    Vectorized converter from cartesian coordinates to spherical coordinates
    :param coords:
    :return: radius, thetas
    '''

    assert coords.shape[1] >= 2

    def _single_angle(current, rest):
        radii = np.linalg.norm(rest, axis=1)
        theta = np.arctan2(radii, current)
        return theta

    radii = np.linalg.norm(coords, axis=1)
    num_thetas = coords.shape[1] - 1 # one fewer parameters since radius is a parameter
    thetas = np.zeros(shape=(coords.shape[0], num_thetas))

    for i in range(num_thetas):
        current = coords[:,i]
        rest = coords[:,i+1:]
        theta = _single_angle(current, rest)
        if i == (num_thetas - 1):
            ids = np.argwhere(coords[:,-1] < 0)
            if len(ids) > 0:
                theta[ids] = 2*np.pi - theta[ids]

        thetas[:,i] = theta

    return radii, thetas

def convert_spherical_to_cart(radii, thetas):
    '''
    Vectorized converter from spherical coordinates to cartesian coordinates
    :param radii:
    :param thetas:
    :return: coords
    '''

    num_coords = thetas.shape[1] + 1
    coords = np.zeros(shape=(thetas.shape[0], num_coords))

    for i in range(num_coords):
        if i == 0:
            coords[:,i] = radii * np.cos(thetas[:,i])
        elif i == (num_coords - 1): # last one
            coords[:, i] = radii * np.prod(np.sin(thetas[:, 0:i-1]), axis=1) * np.sin(thetas[:,i-1])
        else: # all others
            coords[:, i] = radii * np.prod(np.sin(thetas[:, 0:i]), axis=1) * np.cos(thetas[:,i])

    return coords

def generate_points_on_hypersphere(num_points, dimensions=2):
    '''
    Generates points on a hypersphere according to the method of Marsaglia, 1972.
    :param num_points:
    :param dimensions:
    :return:
    '''
    p = np.random.randn(num_points, dimensions)
    radii = np.tile(np.linalg.norm(p, axis=1).reshape(-1,1), [1, dimensions])
    return np.divide(p, radii)

def calculate_pairwise_angles(coords1, coords2):
    '''
    Returns the angle between all points in coords1 vs coords2.
    If these points happen to fall on a hypersphere, multiply the angle by the
    radius to calculate the arc distance on the hypersphere between points.
    :param coords1: [n x m] cartesian coordinates
    :param coords2: [n x m] cartesian coordinates
    :return: [n x n] pairwise angles
    '''

    cos_theta = cosine_similarity(coords1, coords2)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    return theta

def test_calculate_pairwise_angles():

    p = generate_points_on_hypersphere(100, 5)
    thetas = calculate_pairwise_angles(p, p)
    np.testing.assert_almost_equal(np.diag(thetas), np.zeros(shape=(p.shape[0],)))



def test_generate_points_on_hypersphere():

    p = generate_points_on_hypersphere(100, 5)
    r, t = convert_cart_to_spherical(p)
    np.testing.assert_almost_equal(r, np.ones(shape=(p.shape[0],)))

def test_spherical_to_cart():

    #2D test
    p = np.array([1, 0]).reshape(1, -1)
    r, t = convert_cart_to_spherical(p)
    p2 = convert_spherical_to_cart(r,t)
    np.testing.assert_equal(p, p2)

    # 2D test with multiple vectors
    p = np.array([[1, 0], [0, 1]])
    r, t = convert_cart_to_spherical(p)
    p2 = convert_spherical_to_cart(r, t)
    np.testing.assert_array_almost_equal(p, p2)

    #3D test
    p = np.array([1, 0, 0]).reshape(1, -1)
    r, t = convert_cart_to_spherical(p)
    p2 = convert_spherical_to_cart(r, t)
    np.testing.assert_array_almost_equal(p, p2)

    #3D test with multiple vectors
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    r, t = convert_cart_to_spherical(p)
    p2 = convert_spherical_to_cart(r, t)
    np.testing.assert_array_almost_equal(p, p2)

    p = np.random.randn(1000, 3)
    r, t = convert_cart_to_spherical(p)
    p2 = convert_spherical_to_cart(r, t)
    np.testing.assert_array_almost_equal(p, p2)

def test_cart_to_spherical():

    # 2D test
    r, t = convert_cart_to_spherical(np.array([1,0]).reshape(1,-1))
    assert r == 1 and t == 0

    r, t = convert_cart_to_spherical(np.array([0,1]).reshape(1,-1))
    assert r == 1 and t == np.pi/2

    r, t = convert_cart_to_spherical(np.array([1/np.sqrt(2),1/np.sqrt(2)]).reshape(1,-1))
    np.testing.assert_almost_equal(r, np.array([1.]))
    np.testing.assert_almost_equal(t, np.array([1*np.pi/4]).reshape(1,-1))

    r, t = convert_cart_to_spherical(np.array([-1/np.sqrt(2),1/np.sqrt(2)]).reshape(1,-1))
    np.testing.assert_almost_equal(r, np.array([1.]))
    np.testing.assert_almost_equal(t, np.array([3*np.pi/4]).reshape(1,-1))

    r, t = convert_cart_to_spherical(np.array([-1/np.sqrt(2),-1/np.sqrt(2)]).reshape(1,-1))
    np.testing.assert_almost_equal(r, np.array([1.]))
    np.testing.assert_almost_equal(t, np.array([2*np.pi-3*np.pi/4]).reshape(1,-1))

    r, t = convert_cart_to_spherical(np.array([1/np.sqrt(2),-1/np.sqrt(2)]).reshape(1,-1))
    np.testing.assert_almost_equal(r, np.array([1.]))
    np.testing.assert_almost_equal(t, np.array([2*np.pi-1*np.pi/4]).reshape(1,-1))

    # 2D test with multiple vectors
    p = np.array([[1,0], [0,1]])
    r, t = convert_cart_to_spherical(p)
    np.testing.assert_equal(r, np.array([1, 1]))
    np.testing.assert_equal(t, np.array([0, np.pi/2]).reshape(2,1))

    # 3D test
    r, t = convert_cart_to_spherical(np.array([1, 0, 0]).reshape(1, -1))
    np.testing.assert_equal(r, np.array([1]))
    np.testing.assert_equal(t, np.array([0, 0]).reshape(1, 2))

    r, t = convert_cart_to_spherical(np.array([0, 1, 0]).reshape(1, -1))
    np.testing.assert_equal(r, np.array([1]))
    np.testing.assert_equal(t, np.array([np.pi/2, 0]).reshape(1, 2))

    r, t = convert_cart_to_spherical(np.array([0, 0, 1]).reshape(1, -1))
    np.testing.assert_equal(r, np.array([1]))
    np.testing.assert_equal(t, np.array([np.pi/2, np.pi/2]).reshape(1, 2))

    # 3D test with multiple vectors
    p = np.array([[1,0,0], [0,1,0], [0,0,1]])
    r, t = convert_cart_to_spherical(p)
    np.testing.assert_equal(r, np.array([1,1,1]))
    np.testing.assert_equal(t, np.array([[0,0],[np.pi/2,0],[np.pi / 2, np.pi / 2]]))
