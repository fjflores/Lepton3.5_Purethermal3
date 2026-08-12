# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Defines scripts.
"""

import argparse
import os
import sys
from lepton import Stream

def _parse_args():
    parser = argparse.ArgumentParser()
    default_dev_idx = int(os.getenv('LEPTON_DEVICE_INDEX', '0'))
    parser.add_argument(
        '-id',
        '--dev_index',
        help = "Lepton camera device index. Can also be set via LEPTON_DEVICE_INDEX env var. Default is 0.",
        type = int,
        default = default_dev_idx,
    )
    parser.add_argument(
        '-r',
        "--record",
        help = "Record data stream. Default is False.",
        action = argparse.BooleanOptionalAction,
        default = False,
    )
    parser.add_argument(
        '-sp',
        "--save_path",
        help = "Path to the save dir. Default is Lepton_Recordings.",
        type = str,
        default = "Lepton_Recordings",
    )
    parser.add_argument(
        '-c',
        "--cmap",
        help = "Colormap used in viewer. Default is magma.",
        default = 'magma',
    )
    parser.add_argument(
        '-vs',
        "--viewer_scale",
        help = "Scale of viewer. Default is 1.",
        type = float,
        default = 1.0,
    )
    parser.add_argument(
        '-d',
        "--detect",
        help = "Moving fronts are detected. Default is False.",
        action = argparse.BooleanOptionalAction,
        default = False,
    )
    args = parser.parse_args()
    return args


def leprun(args = None):
    """
    Starts a Lepton stream.

    Parameters
    ----------
    args : argparse.Namespace, optional
        The arguments to leprun passed through command line. The default is None. Type
        leprun -h in command line to see arguments.

    Returns
    -------
    None
    """
    if args is None:
        args = sys.argv[1:]
    args = _parse_args()

    stream = Stream(args.dev_index)
    stream.start(
        blocking = True,
        record = args.record,
        detect = args.detect,
        cmap = args.cmap,
        scale = args.viewer_scale,
        save_path = args.save_path,
    )

if __name__ == "__main__":
    leprun()
