def list_of_ints(arg):
    return list(map(int, arg.replace(" ", "").split(",")))


def list_of_floats(arg):
    return list(map(float, arg.replace(" ", "").split(",")))


def list_of_str(arg):
    return list(map(str, arg.replace(" ", "").split(",")))
