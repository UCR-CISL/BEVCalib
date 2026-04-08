# BEV shape : (xbound[1] - xbound[0]) / xbound[2], (ybound[1] - ybound[0]) / ybound[2] = (90, 90)
xbound = (-90.0, 90.0, 2.0)
ybound = (-90.0, 90.0, 2.0)
zbound = (-10.0, 10.0, 20.0)
d_conf = (1.0, 90.0, 1.0)
down_ratio = 8
sparse_shape = (720, 720, 41) # bev_shape * down_ratio = sparse_shape
vsize_xyz = [
    (xbound[1] - xbound[0]) / sparse_shape[0],
    (ybound[1] - ybound[0]) / sparse_shape[1],
    (zbound[1] - zbound[0]) / sparse_shape[2],
]