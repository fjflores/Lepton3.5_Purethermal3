# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Classes used to record camera stream and render video.
"""

from threading import Thread
from collections import deque
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from pickle import PicklingError
import zipfile
import struct
import os
import json
from compression import gzip
import cv2
import numpy as np
import lepton
from . import ViewerImage

class FrameWriter:
    """
    Writes frame information to zip archive during camera stream and then extracts and deletes that
    archive when streaming is done

    Parameters
    ----------
    dirpath : string
        The path to the directory in which the frame archive is made.
    cmap : matplotlib.colors.ListedColormap
        The colormap used to colorize the image.

    Attributes
    ----------
    src_verts : list of tuples
        The coordinates of the corners of the ROI defined in viewer window coordinates.

    """
    def __init__(self, dirpath, cmap):
        parentpath = Path(dirpath)
        parentpath.mkdir(parents=True, exist_ok=True)
        fname = Path(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        self._dirpath = parentpath / fname
        self._archive = None
        self._cmap = cmap
        self._archive_fnames = deque([])

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _extract(self):
        frames = {
            "frame_number": [],
            "frame_time": [],
            "temperature": [],
            "mask": [],
            "telemetry": [],
        }
        for f in self._archive.namelist():
            with self._archive.open(f) as file:
                dat = file.read()
                fmt_str = "II" + "H"*lepton.RES + "B"*lepton.RES + "B"*(len(dat)-(3*lepton.RES+8))
                dat = struct.unpack(fmt_str, dat)
                temp = dat[2:(lepton.RES + 2)]
                temp = np.array(temp).reshape(lepton.SHAPE[::-1]) * .01 - 273.15
                temp[temp > 250.0] = float('nan')
                temp[temp < -50.0] = float('nan')
                mask = dat[(lepton.RES + 2):(2*lepton.RES + 2)]
                mask = np.array(mask, dtype=bool).reshape(lepton.SHAPE[::-1])
                telem = dat[(2*lepton.RES + 2):]
                telem = json.loads(bytes(telem).decode("utf-8"))
                frames["frame_number"].append(dat[0])
                frames["frame_time"].append(dat[1])
                frames["temperature"].append(temp)
                frames["mask"].append(mask)
                frames["telemetry"].append(telem)

        self._dirpath.mkdir()
        makevideo(
            frames["temperature"],
            frames["telemetry"],
            frames["mask"],
            self._cmap,
            self._dirpath / ("video.mp4")
        )
        for k, v in frames.items():
            if k == "telemetry":
                continue
            with gzip.open(self._dirpath / (k + ".npy.gz"), "wb") as f:
                np.save(f, np.array(v))
        with gzip.open(self._dirpath / ("telemetry.json.gz"), "wt", encoding="utf-8") as file:
            json.dump(frames["telemetry"], file, indent = 4)

    def add(self, data):
        """
        Adds a frame to the frame archive.

        Parameters
        ----------
        data: dict
            A dictionary containing the frame information. Must include the keys
            "num": int: The frame number.
            "time": int: The frame time in ms.
            "temperature": ndarray: A float ndarray of the frame temperature in C.
            "telemetry": dict: The frame telemetry
            "mask": ndarray: A bool ndarray of the frame detection mask.

        Returns
        -------
        None.

        """
        if data is None:
            return
        n = np.array([data["num"], ], dtype=np.uint32)
        t = np.array([data["time"], ], dtype=np.uint32)
        temp = np.round(100.0 * (data["temperature"].flatten() + 273.15))
        temp[temp > 52315] = 52315 # 250 C
        temp[temp < 22315] = 22315 # -50 C
        temp[np.isnan(temp)] = 0   # Handle nans
        temp = temp.astype(np.uint16)
        try:
            mask = data["mask"].flatten().astype(np.uint8)
        except AttributeError:
            mask = np.zeros(lepton.RES, dtype=np.uint8)
        telem = json.dumps(data["telemetry"])
        msg = n.tobytes() + t.tobytes() + temp.tobytes() + mask.tobytes() + telem.encode("utf-8")
        fname = f"fr{data["num"]:07d}.dat"
        if not fname in self._archive_fnames:
            self._archive.writestr(fname, msg)
            self._archive_fnames.append(fname)

    def open(self):
        """
        Opens the frame writer's zip archive.

        Returns
        -------
        None.

        """
        self._archive = zipfile.ZipFile(
            self._dirpath.with_suffix(".zip"),
            "w",
            zipfile.ZIP_DEFLATED
        )

    def close(self):
        """
        Closes the frame writer's zip archive. Must be called when done.

        Returns
        -------
        None.

        """
        self._extract()
        self._archive.close()
        os.remove(self._dirpath.with_suffix(".zip"))
        self._archive = None

class RawFrameWriter:
    """
    Periodically saves the true raw uint16 sensor frame (centikelvin, before denoising and
    homography) as a 16-bit TIFF during camera stream.

    Parameters
    ----------
    dirpath : string
        The path to the directory in which the snapshot directory is made.
    interval : float > 0
        The time between saved snapshots in minutes.

    """
    def __init__(self, dirpath, interval):
        if not interval > 0.0:
            raise ValueError(f"interval must be a positive number of minutes, got {interval}")
        parentpath = Path(dirpath)
        fname = Path(datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_Thermal")
        self._dirpath = parentpath / fname
        self._interval = 60000.0 * float(interval)
        self._next_time = 0.0
        self._count = 0

    def add(self, raw_data, time):
        """
        Saves a raw frame snapshot if the frame time has reached the next scheduled snapshot
        time. The first valid frame is always saved.

        Parameters
        ----------
        raw_data: ndarray
            The uint16 ndarray of raw frame data. The last 2 telemetry rows are excluded from
            the saved image.
        time: int
            The frame time in ms, offset from the first frame of the stream.

        Returns
        -------
        None.

        """
        if time < self._next_time:
            return
        while self._next_time <= time:
            self._next_time += self._interval
        self._dirpath.mkdir(parents=True, exist_ok=True)
        fname = f"Lepton_Capture_{self._count:04d}.tiff"
        cv2.imwrite(str(self._dirpath / fname), raw_data[:-2])
        self._count += 1

def _render_frame(frame):
    return cv2.cvtColor(ViewerImage(*frame).asuint8(), cv2.COLOR_BGR2RGB)

def _write_rendered_frames(out, frames, rendered_frames):
    cap_time = [f[1]["Uptime (ms)"] - f[3] for f in frames]
    frame_time = np.round(np.arange(0.0, round(cap_time[-1] + 100/3, 8), round(100/3, 8)), 4)
    cap_idx = frame_time[:, np.newaxis] - cap_time
    cap_idx = np.where(cap_idx > 0, cap_idx, np.inf).argmin(axis = 1)
    for i in cap_idx:
        out.write(rendered_frames[i])

def _write_loop(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    i0 = ViewerImage(*frames[0]).asuint8()
    out = cv2.VideoWriter(path, fourcc, 30, (i0.shape[1], i0.shape[0]))

    try:
        with Pool() as pool:
            rendered_frames = pool.map(_render_frame, frames)
        _write_rendered_frames(out, frames, rendered_frames)

    except (AttributeError, PicklingError):
        rendered_frames = [_render_frame(frame) for frame in frames]
        _write_rendered_frames(out, frames, rendered_frames)

    finally:
        out.release()

def makevideo(temperature, telemetry, mask, cmap, path):
    """
    Renders and saves a recording.

    Parameters
    ----------
    temperature: list[ndarray]
        A list of float ndarrays of the frames' temperatures in C.
    telemetry: list[dict]
        A list of the frames' telemetries.
    mask: list[ndarray]
        A list of bool ndarrays of the frames' detection masks.
    cmap : matplotlib.colors.ListedColormap
        The colormap used to colorize the frames.
    path: pathlib.Path
        The path of the video.

    Returns
    -------
    None.

    """
    if len(temperature) < 1:
        return
    t0 = telemetry[0]["Uptime (ms)"]
    opts = {
        "scale" : 4.0,
        "cmap" : cmap,
        "record" : False,
    }
    frames = [(temp, telem, m, t0, opts) for (temp, telem, m) in zip(temperature, telemetry, mask)]
    thread = Thread(target=_write_loop, args=(frames, path, ))
    thread.start()
