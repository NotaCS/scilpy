#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import os
import time

from dipy.io.util import
from nibabel.streamlines import load, save
import numpy as np

from scilpy.tracking.tools import filter_streamlines_by_length
from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exists,
                             add_overwrite_arg)

def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Subsample a set of streamlines.\n'
                    'WARNING: data_per_point is carried')
    p.add_argument(
        'input', action='store',  metavar='input',
        type=str,  help='Streamlines input file name.')
    p.add_argument(
        'output', action='store',  metavar='output',
        type=str,  help='Streamlines output file name.')
    p.add_argument(
        '--minL', default=0., type=float,
        help='Minimum length of streamlines. [%(default)s]')
    p.add_argument(
        '--maxL', default=0., type=float,
        help='Maximum length of streamlines. [%(default)s]')
    p.add_argument('-v', action='store_true', dest='isVerbose',
                   help='Produce verbose output. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.input])
    assert_outputs_exists(parser, args, args.output)

    tractogramFile = load(args.input)
    streamlines = list(tractogramFile.streamlines)

    new_streamlines, new_data = filter_streamlines_by_length(streamlines,
                                             data_per_point,
                                             data_per_streamline,
                                             args.minL,
                                             args.maxL)

    new_tractogram = Tractogram(new_streamlines,
                                data_per_streamline=new_data['per_streamline'],
                                data_per_point=new_data['per_point'],
                                affine_to_rasmm=np.eye(4))

    save(new_tractogram, args.output, header=tractogramFile.header)
    streamlines.save()

if __name__ == "__main__":
    main()
