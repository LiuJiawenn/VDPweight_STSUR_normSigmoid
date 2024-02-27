import mat73


def read_mat(path, name):
    data = mat73.loadmat(path)
    x = data[name]
    return x