from functools import reduce


def compose2(f, g):
    """
    :param f: first function
    :param g: second function
    :return: a lambda that is of the form f(g(x))
    """
    return lambda x: f(g(x))


def compose_n(functions):
    """
    :param functions: a list of functions f1, f2 .... fn
    :return: a lambda that is of the form f1(f2(.....fn(x))...))
    """
    # initial value for reduce is a Identity function
    # [::-1] is used to reverse the operations as we expect the functions to be provided in the first to last order.
    # i.e. for f(g(h(x))) the input will be [h, g, f]
    return reduce(compose2, functions[::-1], lambda x: x)
