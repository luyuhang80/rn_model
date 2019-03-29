import numpy as np

def unpack_onr_npz_fc(npz_path, N=25):

    npzd = np.load(npz_path)
    v_objs = npzd['v_objs'][:N, :]
    v_rels = npzd['v_rels'][:N, :N, :]

    n_obj, F_o = v_objs.shape
    _, _, F_r = v_rels.shape
    x = np.zeros([N, F_o])
    A = np.zeros([N, N], dtype='int32')
    R = np.zeros([N, N, F_r])

    x[:n_obj, :] = v_objs

    A[1:n_obj, 1:n_obj] = np.ones([n_obj-1, n_obj-1], dtype='int32')

    R[:n_obj, :n_obj, :] = v_rels

    return [x, R, A]

def unpack_onr_npz(npz_path, N=25):

    npzd = np.load(npz_path)
    v_objs = npzd['v_objs'][:N, :]
    v_rels = npzd['v_rels'][:N, :N, :]
    adj_mat = npzd['adj_mat'][:N, :N]

    n_obj, F_o = v_objs.shape
    _, _, F_r = v_rels.shape
    x = np.zeros([N, F_o])
    A = np.zeros([N, N], dtype='int32')
    R = np.zeros([N, N, F_r])

    x[:n_obj, :] = v_objs

    A[1:n_obj, 1:n_obj] = adj_mat[1:, 1:]

    R[:n_obj, :n_obj, :] = v_rels

    return [x, R, A]

def unpack_onu_npz(npz_path, N=25):

    npzd = np.load(npz_path)
    objs = npzd['objs'][:N, :]
    unions = npzd['unions'][:N, :N, :]

    n_obj, F = objs.shape
    x = np.zeros([N, F])
    A = np.zeros([N, N], dtype='int32')
    R = np.zeros([N, N, F])

    x[:n_obj, :] = objs

    A[1:n_obj, 1:n_obj] = np.ones([n_obj-1, n_obj-1], dtype='int32')

    R[:n_obj, :n_obj, :] = unions

    return [x, R, A]

def unpack_o_npz(npz_path, N=25):

    npzd = np.load(npz_path)
    objs = npzd['objs'][:N, :]

    n_obj, F = objs.shape
    x = np.zeros([N, F])
    A = np.zeros([N, N], dtype='int32')

    x[:n_obj, :] = objs

    A[1:n_obj, 1:n_obj] = np.ones([n_obj-1, n_obj-1], dtype='int32')


    return [x, A]