from math import floor, fmod

import numpy as np
import torch


def get_station_locs():
    dstcl = 5200.0
    dstqa = 2600.0
    dstas = 1300.0
    sddy = [-1.0, -1.0, 1.0, 1.0]
    sddx = [-1.0, 1.0, 1.0, -1.0]
    iasn = 0
    real_iasn = 0
    stations = {}

    for icl in range(1, 17):
        ycl = (float(floor((icl - 1) / 4)) - 1.5) * dstcl
        xcl = (float(fmod(icl - 1, 4)) - 1.5) * dstcl

        for j in range(0, 4):
            yqa = ycl + 0.5 * dstqa * sddy[j]
            xqa = xcl + 0.5 * dstqa * sddx[j]

            for k in range(0, 4):
                yas = yqa + 0.5 * dstas * sddy[k]
                xas = xqa + 0.5 * dstas * sddx[k]
                iasn += 1

                if yas < -dstas or yas > dstas or xas < -dstas or xas > dstas:
                    real_iasn += 1
                    iccdd = icl * 100 + fmod(iasn - 1, 16) + 1

                    if iccdd == 610:
                        xas += 23.0
                        yas += 100.0
                    elif iccdd == 713:
                        xas += 100.0
                    elif iccdd == 1007:
                        xas += 20.0
                        yas += 495.9
                    elif iccdd == 1104:
                        xas += 366.5
                        yas += 461.0
                    stations[real_iasn] = [xas / 100.0, yas / 100.0]
    return stations


def update_keys(stations):
    d = {}
    for k, v in stations.items():
        if k < 91:
            d[k] = v
        elif 91 <= k < 111:
            d[k + 1] = v
        elif 111 <= k < 148:
            d[k + 2] = v
        elif 148 <= k < 158:
            d[k + 3] = v
        elif 158 <= k:
            d[k + 4] = v
    return d


# узнаем в какой квадрат попали
def get_big_square_num(order_num):
    x = (order_num % 16) // 4
    y = 3 - (order_num // 16) // 4

    return x, y


# узнаем начальный номер в квадрате
def get_start_num(x, y):
    return y * 64 + x * 16 + 1


# узнаем номер из словаря
def get_pos(order_num, mat):
    x, y = get_big_square_num(order_num)

    start_num = get_start_num(x, y)

    x = order_num % 4
    y = (order_num // 16) % 4

    dict_key = start_num + mat[y, x]

    return dict_key


def get_coordinates(order_num, mat, our_dict):
    dict_key = get_pos(order_num, mat)
    if dict_key in our_dict:
        return our_dict[dict_key]
    return None


def get_all_edges_real_dist(n: int):
    # 119, 120, 135, 136
    a = [i for i in range(n * n - 4) for _ in range(n * n - 1 - 4)]
    b = [j for i in range(n * n - 4) for j in range(n * n - 4) if j != i]

    a_all = [i for i in range(n * n) for _ in range(n * n - 1)]
    b_all = [j for i in range(n * n) for j in range(n * n) if j != i]

    stations = update_keys(get_station_locs())
    mat = np.array([[15, 14, 11, 10], [12, 13, 8, 9], [3, 2, 7, 6], [0, 1, 4, 5]])

    dist = []

    for i, j in zip(a_all, b_all):
        a_coord, b_coord = get_coordinates(i, mat, stations), get_coordinates(
            j, mat, stations
        )

        if a_coord is None or b_coord is None:
            continue

        dist.append(
            ((a_coord[0] - b_coord[0]) ** 2 + (a_coord[1] - b_coord[1]) ** 2) ** 0.5
        )

    edge_index = np.array([a, b])
    return torch.tensor(edge_index, dtype=torch.long).contiguous(), torch.tensor(
        np.array(dist).reshape(-1, 1)
    )
