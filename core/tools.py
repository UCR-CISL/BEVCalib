import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_single_perturbation_from_T(T, angle_range_deg=20, trans_range=1.5):
    """
    Given a homogeneous transformation matrix T, generate a single perturbed transformation matrix
    with random rotation and translation perturbations.

    Parameters:
        T: np.ndarray, shape (B, 4, 4)
           Original homogeneous transformation matrices. The upper left (3x3) sub-matrix is the rotation matrix, 
           the first 3 elements of the last column are the translation vector, and the last row should be [0, 0, 0, 1].
        angle_range_deg: float, the range of the rotation perturbation in degrees (absolute value)
        trans_range: float, the range of translation perturbation (each coordinate will be perturbed within [-trans_range, trans_range])
    """
    B = T.shape[0]
    T_new = []

    for i in range(B):
        # Extract the rotation matrix and translation vector from T
        orig_rot_matrix = T[i, :3, :3]
        orig_trans = T[i, :3, 3]

        # Create a Rotation object from the rotation matrix
        orig_rot = R.from_matrix(orig_rot_matrix)

        # Generate random perturbation:
        # Create a random axis and normalize it
        rand_axis = np.random.randn(3)
        rand_axis /= np.linalg.norm(rand_axis)
        # Generate a random rotation angle (in radians)
        rand_angle = np.deg2rad(np.random.uniform(-angle_range_deg, angle_range_deg))
        # Create the rotation perturbation
        delta_rot = R.from_rotvec(rand_angle * rand_axis)
        # Apply the perturbation (right multiply) to the original rotation
        new_rot = delta_rot * orig_rot

        # Generate translation perturbation with exact magnitude of trans_range
        # First generate a random direction vector
        rand_direction = np.random.randn(3)
        # Normalize it
        rand_direction = rand_direction / np.linalg.norm(rand_direction)
        # Scale it to have magnitude of trans_range
        delta_trans_magnitude = np.random.uniform(0, trans_range)
        delta_trans = rand_direction * delta_trans_magnitude
        # Apply the perturbation
        new_trans = orig_trans + delta_trans

        # Construct a new 4x4 homogeneous transformation matrix
        T_temp = np.eye(4)
        T_temp[:3, :3] = new_rot.as_matrix()
        T_temp[:3, 3] = new_trans
        T_new.append(T_temp)

    return np.array(T_new), np.rad2deg(rand_angle), delta_trans_magnitude

def generate_intrinsic_matrix(fx, fy, cx, cy):
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic_matrix
