import numpy as np
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian

AZ_MAX = 360


def revert_azimuth_angle(features: np.ndarray, angle_index: int):
    az = features[:, angle_index]
    az[az > AZ_MAX] -= AZ_MAX

    assert max(features[:, angle_index]) < AZ_MAX, "Revert error"


def make_gamma_label_equal_to_1(features: np.ndarray) -> np.ndarray:
    features[:, 0] = abs(features[:, 0] - 1)
    return features


def get_target(features: np.ndarray) -> np.ndarray:
    return features[:, 0]


def to_R_astropy(x, y, z=None):
    if z is None:
        z = np.sqrt(1 - x * x - y * y)
    _, ZeR, AzR = cartesian_to_spherical(x, y, z)
    Ze = 90 - np.degrees(ZeR.value)
    Az = np.degrees(AzR.value)
    return Ze, Az


def to_XY_astropy(Ze, Az):
    r = 1
    ZeR = np.radians(90 - Ze)
    AzR = np.radians(Az)
    x, y, z = spherical_to_cartesian(r, ZeR, AzR)
    return x.value, y.value, z.value
