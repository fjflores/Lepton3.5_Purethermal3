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

def _positive_float(val):
    try:
        fval = float(val)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid float value: '{val}'") from e
    if fval <= 0.0:
        raise argparse.ArgumentTypeError(f"interval must be positive, got {val}")
    return fval


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
        '-sr',
        "--save_raw",
        help = "Interval in minutes between raw uint16 TIFF snapshots (pre-denoise, "
               "pre-homography), saved under the save path. Default is off.",
        type = _positive_float,
        default = None,
    )
    parser.add_argument(
        '-tr',
        "--temp_range",
        help = "Fixed color-scale temperature range in C as MIN MAX. Applies to the viewer "
               "and recorded video only; raw TIFF snapshots and the stats CSV are unaffected. "
               "Default is per-frame autoscale.",
        type = float,
        nargs = 2,
        metavar = ("MIN", "MAX"),
        default = None,
    )
    parser.add_argument(
        '-rot',
        "--rotate",
        help = "Clockwise rotation of the camera image in degrees. Applies to the viewer, "
               "recorded data and video, and raw TIFF snapshots. Default is 0.",
        type = int,
        choices = [0, 90, 180, 270],
        default = 0,
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
    if args.temp_range is not None and not args.temp_range[0] < args.temp_range[1]:
        parser.error(
            f"temp_range MIN must be less than MAX, got {args.temp_range[0]} {args.temp_range[1]}"
        )
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
        save_raw = args.save_raw,
        temp_range = args.temp_range,
        rotation = args.rotate,
    )

if __name__ == "__main__":
    leprun()
