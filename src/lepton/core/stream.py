# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Classes for starting camera stream.
"""

from copy import copy
import time
from threading import Thread, Lock
from collections import deque
from dataclasses import dataclass
import traceback
import sys
import lepton
from lepton.misc import colormaps, detect_fp_fronts
from lepton.exceptions import CaptureException, CaptureTimeout, ShapeException, UnknownCmapException
from . import Capture, Viewer, ViewerImage, FrameWriter, RawFrameWriter

@dataclass()
class StreamBuffer:
    """
    Stores stream data in rolling buffer.

    Parameters
    ----------
    size: int
        The maximum length of the buffer before oldest data is removed.

    """
    size: int

    def __post_init__(self):
        self._data = deque()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __contains__(self, key):
        return key in self._data

    def _trim(self):
        while len(self._data) > self.size:
            self._data.popleft()

    def append(self, val):
        """
        Adds an element to the right end of the buffer.

        """
        self._data.append(val)
        self._trim()

    def popleft(self):
        """
        Removes and returns the element from the left end of the buffer.

        """
        return self._data.popleft()

    def isempty(self):
        """
        Return True if the buffer is empty.

        """
        return len(self._data) == 0

class Stream:
    """
    Provides functionality for starting and stopping a camera stream. The stream is displayed
    to an onscreen viewer window and is optionally recorded.

    Parameters
    ----------
    dev_idx : int
        Integer that specifies which camera device is the Lepton.

    """
    def __init__(self, dev_idx):
        self._lock = Lock()
        self._params = {
            "dev_idx": dev_idx,
            "window": "LepViewer",
            "n0": None,
            "t0": None,
        }
        self._max_bufsize = 64
        self._bufs = {
            'num' : StreamBuffer(self._max_bufsize),
            'time' : StreamBuffer(self._max_bufsize),
            'temperature' : StreamBuffer(self._max_bufsize),
            'telemetry' : StreamBuffer(self._max_bufsize),
            'mask' : StreamBuffer(self._max_bufsize),
        }
        self._flags = {
            "streaming": False,
            "complete": False,
        }

    def __call__(self, **kwargs):
        """
        Starts the camera stream. Functionality is identical to Stream.start but only runs in
        non-blocking mode. Is useful for context management.

        Parameters
        ----------
        **kwargs

        Keyword Args
        ------------
        record: bool
             Whether to record the stream. The default is False.
        detect: bool
            Whether to detect frontal polymerization fronts. The default is False
        cmap: string
            The colormap used to color the frame data in the viewer window. The default is 'magma'
        scale: float > 1
            The scale of the viewer window compared to the camera temperature data. The default is 1
        dirpath: string
            The path to the directory in which the recording data is saved. The default is
            'Lepton_Recordings'
        save_raw: float > 0
            When given, the true raw uint16 sensor frame (centikelvin, before denoising and
            homography) is saved as a 16-bit TIFF every save_raw minutes, starting with the
            first valid frame. The frame's minimum, median, and maximum temperature are also
            appended to a Temperature_Stats.csv file, one row per snapshot. Snapshots are
            saved to a '<timestamp>_Thermal' directory inside the save path. Works with or
            without recording. The default is None (no snapshots).
        temp_range: tuple (min_C, max_C)
            When given, the viewer (and any recorded video) maps this fixed temperature range
            across the colormap instead of autoscaling each frame to its own min/max. Only
            affects visualization; recorded temperature data and raw snapshots are unchanged.
            The default is None (autoscale).
        rotation: int
            Clockwise rotation in degrees applied to the camera image. Must be a multiple of
            90 (negative values allowed). Applies everywhere: the viewer, the recorded
            temperature/mask arrays and video, and raw TIFF snapshots. The default is 0.

        Returns
        -------
        None.

        Examples
        --------
        >>> import time
        ... stream = Stream(0)
        ... with stream() as s:
        ...     for _ in range(10):
        ...         frame_dat = s.get_frame()
        ...         time.sleep(0.5)

        """
        kwargs["blocking"] = False
        self.start(**kwargs)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def _reset(self):
        self._params["n0"] = None
        self._params["t0"] = None
        self._bufs = {
            'num' : StreamBuffer(self._max_bufsize),
            'time' : StreamBuffer(self._max_bufsize),
            'temperature' : StreamBuffer(self._max_bufsize),
            'telemetry' : StreamBuffer(self._max_bufsize),
            'mask' : StreamBuffer(self._max_bufsize),
        }
        self._flags = {
            "streaming" : False,
            "complete": False,
        }

    def _done(self):
        with self._lock:
            return not self._flags["streaming"]

    def _bufs_empty(self):
        with self._lock:
            return self._bufs['num'].isempty()

    def _bufs_popleft(self, min_len):
        with self._lock:
            if len(self._bufs['num']) >= min_len:
                bufdat = {
                    'num': self._bufs['num'].popleft(),
                    'time': self._bufs['time'].popleft(),
                    'temperature': self._bufs['temperature'].popleft(),
                    'telemetry': self._bufs['telemetry'].popleft(),
                    'mask': self._bufs['mask'].popleft()
                }
                return bufdat
        time.sleep(0.05)
        return None

    def _record_loop(self, dirpath, cmap, temp_range, shape):
        with FrameWriter(dirpath, cmap, temp_range, shape) as writer:
            while not self._done() or not self._bufs_empty():
                data = self._bufs_popleft(1 if self._done() else 8)
                writer.add(data)

    def _process_frame(self, frame, viewer, opts):
        curr_num = frame.telemetry["Frame count since reboot"]
        curr_time = frame.telemetry["Uptime (ms)"]
        with self._lock:
            # Check for FFC in progress by unupdated number, unupdated time, or invalid number/time
            if (curr_num in self._bufs['num'] or
                curr_time in self._bufs['time'] or
                not 0 <= curr_num <= 1555200 or
                not 0 <= curr_time <= 172800000):
                return

            if self._params["n0"] is None and self._params["t0"] is None:
                self._params["n0"] = copy(curr_num)
                self._params["t0"] = copy(curr_time)

            self._bufs['num'].append(curr_num - self._params["n0"])
            self._bufs['time'].append(curr_time - self._params["t0"])
            self._bufs['temperature'].append(frame.temperature)
            self._bufs['telemetry'].append(frame.telemetry)

            if opts["raw_writer"] is not None:
                opts["raw_writer"].add(frame.raw_data, frame.telemetry, self._bufs['time'][-1])

            if opts["detect"]:
                i_range = range(max(-len(self._bufs['temperature']), -3), 0)
                mask = detect_fp_fronts([self._bufs['temperature'][i] for i in i_range])
                self._bufs['mask'].append(mask)
            else:
                self._bufs['mask'].append(None)

            image = ViewerImage(
                self._bufs['temperature'][-1],
                self._bufs['telemetry'][-1],
                self._bufs['mask'][-1],
                self._params["t0"],
                opts
            )
            if viewer.imshow(image.asuint8()) == "esc":
                self._flags["streaming"] = False

    def _print_exception(self, e):
        msg = ""
        msg += "-"*75 + "\n"
        msg += "An error has occured and will be handled now.\n\n"
        n_spaces = 42 - len(type(e).__name__)
        msg += f"{type(e).__name__}" + " "*n_spaces + "Traceback (most recent call last)\n"
        _, _, tb = sys.exc_info()
        stack_summary = traceback.extract_tb(tb)
        for s in traceback.format_list(stack_summary):
            msg += s
        msg += f"\n{type(e).__name__}: {e}\n"
        msg += "-"*75 + "\n"
        print(msg, flush = True)

    def _stream_loop(self, cap, viewer, opts):
        while not self._done():
            try:
                frame = cap.read(homography = viewer.homography)
            except (CaptureException, CaptureTimeout, ShapeException) as e:
                self._print_exception(e)
                return
            self._process_frame(frame, viewer, opts)

    def _start(self, **kwargs):
        self._reset()

        try:
            opts = {
                "record": kwargs.get("record", False),
                "detect": kwargs.get("detect", False),
                "cmap": colormaps[kwargs.get("cmap", "magma")],
                "scale": 4.0*max(min(kwargs.get("scale", 1.0), 2.0), 0.25),
                "dirpath": kwargs.get("save_path", "Lepton_Recordings"),
                "save_raw": kwargs.get("save_raw", None),
                "temp_range": kwargs.get("temp_range", None),
                "rotation": int(kwargs.get("rotation", 0)) % 360,
            }
            if opts["rotation"] not in (0, 90, 180, 270):
                raise ValueError(
                    f"rotation must be a multiple of 90 degrees, got {kwargs.get('rotation')}"
                )
            frame_shape = lepton.SHAPE if opts["rotation"] % 180 == 0 else lepton.SHAPE[::-1]
            if opts["temp_range"] is not None:
                opts["temp_range"] = (float(opts["temp_range"][0]), float(opts["temp_range"][1]))
                if not opts["temp_range"][0] < opts["temp_range"][1]:
                    raise ValueError(
                        f"temp_range min must be less than max, got {opts['temp_range']}"
                    )
            if opts["save_raw"] is not None:
                opts["raw_writer"] = RawFrameWriter(
                    opts["dirpath"], opts["save_raw"], opts["rotation"]
                )
            else:
                opts["raw_writer"] = None
            with(
                Capture(self._params["dev_idx"], opts["rotation"]) as cap,
                Viewer(self._params["window"], opts["scale"], frame_shape) as viewer,
            ):
                with self._lock:
                    self._flags["streaming"] = True
                if opts["record"]:
                    _record_thread = Thread(
                        target=self._record_loop,
                        args=(opts["dirpath"], opts["cmap"], opts["temp_range"], frame_shape, )
                    )
                    _record_thread.start()
                self._stream_loop(cap, viewer, opts)

        except UnknownCmapException as e:
            self._print_exception(e)
            print("The stream never started, so no data was generated.", flush = True)

        finally:
            with self._lock:
                self._flags["streaming"] = False
                self._flags["complete"] = True

    def start(self, **kwargs):
        """
        Starts the camera stream.

        Parameters
        ----------
        **kwargs

        Keyword Args
        ------------
        blocking: bool
            Whether to run the stream as a blocking thread (True) or a non-blocking thread (False).
            The default is False.
        record: bool
             Whether to record the stream. The default is False.
        detect: bool
            Whether to detect frontal polymerization fronts. The default is False
        cmap: string
            The colormap used to color the frame data in the viewer window. The default is 'magma'
        scale: float > 1
            The scale of the viewer window compared to the camera temperature data. The default is 1
        dirpath: string
            The path to the directory in which the recording data is saved. The default is
            'Lepton_Recordings'
        save_raw: float > 0
            When given, the true raw uint16 sensor frame (centikelvin, before denoising and
            homography) is saved as a 16-bit TIFF every save_raw minutes, starting with the
            first valid frame. The frame's minimum, median, and maximum temperature are also
            appended to a Temperature_Stats.csv file, one row per snapshot. Snapshots are
            saved to a '<timestamp>_Thermal' directory inside the save path. Works with or
            without recording. The default is None (no snapshots).
        temp_range: tuple (min_C, max_C)
            When given, the viewer (and any recorded video) maps this fixed temperature range
            across the colormap instead of autoscaling each frame to its own min/max. Only
            affects visualization; recorded temperature data and raw snapshots are unchanged.
            The default is None (autoscale).
        rotation: int
            Clockwise rotation in degrees applied to the camera image. Must be a multiple of
            90 (negative values allowed). Applies everywhere: the viewer, the recorded
            temperature/mask arrays and video, and raw TIFF snapshots. The default is 0.

        Returns
        -------
        None.

        """
        if kwargs.get("blocking", True):
            self._start(**kwargs)
            return
        Thread(target=self._start, kwargs=kwargs).start()

        # Wait until stream is running to return
        while not self.is_running() and not self.is_complete():
            time.sleep(0.1)

    def is_running(self):
        """
        True if the stream is currently running (camera frames are being collected and displayed)

        """
        with self._lock:
            return self._flags["streaming"] and len(self._bufs['num']) > 0

    def is_complete(self):
        """
        True if the stream has started before but is no longer running.

        """
        with self._lock:
            return self._flags["complete"]

    def get_frame(self):
        """
        Returns the most recent captured frame.

        Returns
        -------
        frame_data: dict
            A dictionary containing the frame information. Includes the keys
            "num": int: The frame number.
            "time": int: The frame time in ms.
            "temperature": ndarray: A float ndarray of the frame temperature in C.
            "telemetry": dict: The frame telemetry
            "mask": ndarray: A bool ndarray of the frame detection mask.

        """
        with self._lock:
            if not self._flags["streaming"] or len(self._bufs['num']) < 1:
                return None
            return {
                'num': self._bufs['num'][-1],
                'time': self._bufs['time'][-1],
                'temperature': self._bufs['temperature'][-1],
                'telemetry': self._bufs['telemetry'][-1],
                'mask': self._bufs['mask'][-1],
            }

    def terminate(self):
        """
        Terminates the stream if it is running. Must be called when done.

        """
        with self._lock:
            self._flags["streaming"] = False
