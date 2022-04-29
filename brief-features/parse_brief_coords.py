import re

import numpy as np
from os import path, getcwd


def convert_offset_to_numpy(coords_offsets_list):
    img_dims = 24
    return np.array(
        [[int(offsets[0]), int(offsets[1]), int(offsets[2]), int(offsets[3])] for offsets in
         coords_offsets_list]) + img_dims


def load_coords(file_path=None):
    file_path = path.join(getcwd(), 'brief-coords-16exp.txt') if not file_path else file_path
    with open(file_path, 'r') as file:
        expressions = file.read()

        ineq_expression_matcher = re.compile(
            r'(SMOOTHED\(-*[\d]+,\s*-*[\d]+\)\s*[<>]\s*SMOOTHED\(-*[\d]+,\s*-*[\d]+\))')
        ineq_expressions = ineq_expression_matcher.findall(expressions)

        coords_offset_matcher = re.compile(r'(-*\d+)')
        coords_offsets_list = [coords_offset_matcher.findall(expr) for expr in ineq_expressions]

    return convert_offset_to_numpy(coords_offsets_list)


if __name__ == '__main__':
    offset_array = load_coords()
    print(offset_array)
