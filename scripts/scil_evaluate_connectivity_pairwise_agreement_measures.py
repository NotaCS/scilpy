#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pair-wise similarity measures of connectivity matrix.

The computed similarity measures are:
sum of square difference and pearson correlation coefficent
"""

import argparse
import itertools
import json
import logging
import os
import shutil

import numpy as np

from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_matrices', nargs='+',
                   help='Path of the input matricies.')
    p.add_argument('out_json',
                   help='Path of the output json file.')
    p.add_argument('--single_compare',
                   help='Compare inputs to this single file.')

    add_json_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_matrices)
    assert_outputs_exist(parser, args, args.out_json)

    all_matrices = []
    for filename in args.in_matrices:
        tmp_mat = load_matrix_in_any_format(filename)
        all_matrices.append(tmp_mat / np.max(tmp_mat))

    output_measures_dict = {'SSD': [], 'correlation': []}
    pairs = list(itertools.combinations(all_matrices, r=2))
    for i in pairs:
        ssd = np.sum((i[0] - i[1]) ** 2)
        output_measures_dict['SSD'].append(ssd)
        corrcoef = np.corrcoef(i[0].ravel(), i[1].ravel())
        output_measures_dict['correlation'].append(corrcoef[0][1])

    if args.single_compare:
        # Move the single_compare only once, at the end.
        if args.single_compare in args.in_matrices:
            args.in_matrices.remove(args.single_compare)
        matrices_list = args.in_matrices + [args.single_compare]
        last_matrix = matrices_list[matrices_list.len-1]
        single_compare_pairs = list(itertools.product(matrices_list, last_matrix))
        for i in single_compare_pairs:
            ssd = np.sum((i[0] - i[1]) ** 2)
            output_measures_dict['SSD'].append(ssd)
            corrcoef = np.corrcoef(i[0].ravel(), i[1].ravel())
            output_measures_dict['correlation'].append(corrcoef[0][1])
    else :
        matrices_list = args.in_matrices

    with open(args.out_json, 'w') as outfile:
        json.dump(output_measures_dict, outfile,
                  indent=args.indent, sort_keys=args.sort_keys)


if __name__ == "__main__":
    main()
