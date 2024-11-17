import math
import numpy as np

def eulerAnglesToRotationMatrix(theta):
    """
    Converts Euler angles to a rotation matrix.
    :param theta: Array of Euler angles [roll, pitch, yaw] in radians.
    :return: 3x3 rotation matrix.
    """
    assert len(theta) == 3, "Input must be a 3-element array [roll, pitch, yaw]."
    
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])

    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])

    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    return R_z @ R_y @ R_x  # Using @ operator for readability


def isRotationMatrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    :param R: 3x3 matrix.
    :return: True if valid, False otherwise.
    """
    Rt = R.T
    identity_check = Rt @ R
    I = np.identity(3, dtype=R.dtype)
    error = np.linalg.norm(I - identity_check)
    return error < 1e-6


def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix to Euler angles.
    :param R: 3x3 rotation matrix.
    :return: Array of Euler angles [roll, pitch, yaw] in radians.
    """
    assert isRotationMatrix(R), "Input matrix is not a valid rotation matrix."
    
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])  # Roll
        y = math.atan2(-R[2, 0], sy)      # Pitch
        z = math.atan2(R[1, 0], R[0, 0])  # Yaw
    else:
        x = math.atan2(-R[1, 2], R[1, 1])  # Roll (special case)
        y = math.atan2(-R[2, 0], sy)       # Pitch (special case)
        z = 0                              # Yaw is indeterminate

    return np.array([x, y, z])


if __name__ == '__main__':
    # Input Euler angles in degrees
    theta_degrees = [-80, 80, 45]
    theta_radians = np.radians(theta_degrees)  # Convert to radians

    # Compute rotation matrix
    rot_mat = eulerAnglesToRotationMatrix(theta_radians)

    # Convert back to Euler angles
    angles = rotationMatrixToEulerAngles(rot_mat)

    # Print results
    print(f"Input Euler angles (degrees): {theta_degrees}")
    print("Rotation matrix:")
    print(rot_mat)
    print("Recovered Euler angles (degrees):")
    print(np.degrees(angles))
