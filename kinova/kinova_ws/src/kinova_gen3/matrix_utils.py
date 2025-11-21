import numpy as np

def compute_transformation(pixel_pts, world_pts):
    """
    Computes the affine transform matrix that maps pixel coordinates -> world coordinates.
    
    pixel_pts: Nx2 array of (x_pixel, y_pixel)
    world_pts: Nx2 array of (x_world, y_world)

    Returns: 2x3 affine transform matrix A such that:
        [xw, yw]^T = A @ [xp, yp, 1]^T
    """

    pixel_pts = np.array(pixel_pts)
    world_pts = np.array(world_pts)

    N = pixel_pts.shape[0]
    A = []

    # Build least-squares system
    for i in range(N):
        xp, yp = pixel_pts[i]
        xw, yw = world_pts[i]

        A.append([xp, yp, 1, 0,  0,  0])
        A.append([0,  0,  0, xp, yp, 1])

    A = np.array(A)
    b = world_pts.flatten()

    # Solve for transformation parameters
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # params = [a11, a12, tx, a21, a22, ty]
    return params.reshape(2, 3)


def apply_transformation(A, pixel_point):
    """
    Applies a 2x3 affine transform A to a pixel coordinate.
    """
    xp, yp = pixel_point
    vec = np.array([xp, yp, 1])
    return A @ vec  # returns (x_world, y_world)
