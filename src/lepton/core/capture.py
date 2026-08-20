# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Classes for directly interfacing with the camera.
"""

import struct
import time
from copy import copy
from dataclasses import dataclass, field
from collections import deque
import cv2
import numpy as np
from skimage import filters
import lepton
from lepton.exceptions import CaptureException, ShapeException, CaptureTimeout

_ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

def rotate_frame(arr, rotation):
    """
    Rotates a frame clockwise in 90 degree steps.

    Parameters
    ----------
    arr: ndarray
        The frame data to rotate.
    rotation: int
        The clockwise rotation in degrees. Must be one of 0, 90, 180, or 270.

    Returns
    -------
    rotated: ndarray
        The rotated frame data.

    Raises
    ------
    ValueError
        Raised when rotation is not one of 0, 90, 180, or 270.

    """
    try:
        code = _ROTATE_CODES[rotation]
    except KeyError:
        raise ValueError(
            f"rotation must be one of {sorted(_ROTATE_CODES)}, got {rotation}"
        ) from None
    if code is None:
        return arr
    return cv2.rotate(arr, code)

@dataclass
class CapFrame:
    """
    Stores raw and decoded captured Lepton frame data.

    Parameters
    ----------
    raw_data: ndarray
        An integer ndarray containing the raw captured Lepton frame data.
    frame_times: deque
        An integer deque giving the previous frame times in ns.
    homography: ndarray, optional
        A float ndarray that defines a homography transform to apply to the captured temperature
        data after decoding. When None, no homography is applied. The default is None.
    rotation: int, optional
        The clockwise rotation in degrees applied to the temperature data after denoising and
        before the homography. Must be one of 0, 90, 180, or 270. The default is 0.

    Attributes
    ----------
    temperature: ndarray
        An float ndarray containing the decoded temperature data in celcius.
    telemetry: dict
        A dict containing the decoded frame telemetry.

    """
    raw_data: np.ndarray
    frame_times: deque
    homography: tuple(np.ndarray, np.ndarray) = (None, None)
    rotation: int = 0
    temperature: np.ndarray = field(default_factory = lambda: np.ndarray(0))
    telemetry: dict = field(default_factory = dict)

    def __post_init__(self):
        self._get_temperature()
        self._get_telemetry()

    def _denoise(self, temperature):
        for _ in range(3):
            median = cv2.medianBlur(temperature, 3)
            diff = abs(temperature - median)
            if np.all(diff == diff[0]):
                break
            dead = diff > filters.threshold_yen(diff)
            temperature[dead] = median[dead]
        return temperature

    def _get_temperature(self):
        temperature = self.raw_data[:-2].astype(np.float32) * 0.01 - 273.15
        self.temperature = self._denoise(temperature)
        self.temperature = rotate_frame(self.temperature, self.rotation)
        if not any(h is None for h in self.homography):
            shape = self.temperature.shape[::-1]
            self.temperature = cv2.warpPerspective(
                self.temperature,
                self.homography[0],
                shape,
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpPerspective(
                np.ones(shape[::-1]),
                self.homography[0],
                shape,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            corners = np.round(self.homography[1]).astype(np.int32)
            mask[0:corners[0, 1], :] = 0
            mask[corners[1, 1] + 1:, :] = 0
            mask[:, 0:corners[0, 0]] = 0
            mask[:, corners[2, 0] + 1:] = 0
            self.temperature[mask == 0] = float('nan')

    def _unpack(self):
        dat_a = struct.unpack("<bbII16c8B6xI5H4xHIH2x6H64xIH10x", self.raw_data[-2,:80])
        dat_b = struct.unpack("<38x8H106x", self.raw_data[-2,80:])
        dat_c = struct.unpack("<10x5H8xHH12x4H44x?x9H44x", self.raw_data[-1,:80])
        return dat_a, dat_b, dat_c

    def _read_a(self, dat):
        conditionals = [None, ]*6
        conditionals[0] = (
            "not desired" if dat[3] & 8 == 0 else
            "desired" if dat[3] & 8 == 8 else
            ""
        )
        conditionals[1] = (
            "never commanded" if dat[3] & 48 == 0 else
            "imminent" if dat[3] & 48 == 16 else
            "in progress" if dat[3] & 48 == 32 else
            "complete" if dat[3] & 48 == 48 else
            ""
        )
        conditionals[2] = (
            "disabled" if dat[3] & 4096 == 0 else
            "enabled" if dat[3] & 4096 == 4096 else
            ""
        )
        conditionals[3] = (
            "not locked out" if dat[3] & 32768 == 0 else
            "locked out" if dat[3] & 32768 == 32768 else
            ""
        )
        conditionals[4] = (
            "not imminent" if dat[3] & 1048576 == 0 else
            "within 10s" if dat[3] & 1048576 == 1048576 else
            ""
        )
        conditionals[5] = (
            "RGB888" if dat[43] == 3 else
            "RAW14" if dat[43] == 7 else
            ""
        )
        self.telemetry["Telemetry version"] = f"{dat[0]}.{dat[1]}"
        self.telemetry["Uptime (ms)"] = dat[2]
        self.telemetry["FFC desired"] = conditionals[0]
        self.telemetry["FFC state"] = conditionals[1]
        self.telemetry["AGC state"] = conditionals[2]
        self.telemetry["Shutter lockout"] = conditionals[3]
        self.telemetry["Overtemp shutdown"] = conditionals[4]
        self.telemetry["Serial number (hex)"] = b"".join(dat[4:20]).hex()
        self.telemetry["gpp version"] = f"{dat[20]}.{dat[21]}.{dat[22]}"
        self.telemetry["dsp version"] = f"{dat[24]}.{dat[25]}.{dat[26]}"
        self.telemetry["Frame count since reboot"] = dat[28]
        self.telemetry["Frame mean"] = dat[29]
        self.telemetry["FPA temperature (counts)"] = dat[30]
        self.telemetry["FPA temperature (C)"] = round(dat[31] * 0.01 - 273.15, 2)
        self.telemetry["Housing temperature (counts)"] = dat[32]
        self.telemetry["Housing temperature (C)"] = round(dat[33] * 0.01 - 273.15, 2)
        self.telemetry["FPA temperature at last FFC (C)"] = round(dat[34] * 0.01 - 273.15, 2)
        self.telemetry["Uptime at last FFC (ms)"] = dat[35]
        self.telemetry["Housing temperature at last FFC (C)"] = round(dat[36] * 0.01 - 273.15, 2)
        self.telemetry["AGC ROI (top left bottom right)"] = dat[37:41]
        self.telemetry["AGC clip high"] = dat[41]
        self.telemetry["AGC clip low"] = dat[42]
        self.telemetry["Video format"] = conditionals[5]
        self.telemetry["Number of frames used for FFC"] = 2**dat[44]

    def _read_b(self, dat):
        self.telemetry["Assumed emissivity"] = round(dat[0] / 8192, 2)
        self.telemetry["Assumed background temperature (C)"] = round(dat[1] * 0.01 - 273.15, 2)
        self.telemetry["Assumed atmospheric transmission"] = round(dat[2] / 8192, 2)
        self.telemetry["Assumed atmospheric temperature (C)"] = round(dat[3] * 0.01 - 273.15, 2)
        self.telemetry["Assumed window transmission"] = round(dat[4] / 8192, 2)
        self.telemetry["Assumed window reflection"] = round(dat[5] / 8192, 2)
        self.telemetry["Assumed window temperature (C)"] = round(dat[6] * 0.01 - 273.15, 2)
        self.telemetry["Assumed reflected temperature (C)"] = round(dat[7] * 0.01 - 273.15, 2)

    def _read_c(self, dat):
        conditionals = [None, ] * 3
        conditionals[0] = (
            "high" if dat[0] == 0 else
            "low" if dat[0] == 1 else
            "auto" if dat[0] == 2 else
            ""
        )
        conditionals[1] = (
            "high" if dat[1] == 0 else
            "low" if dat[1] == 1 else
            "auto" if dat[1] == 2 else
            "not in auto mode"
        )
        conditionals[2] = (
            copy(conditionals[0]) if dat[2] == 0 else
            "low" if dat[2] == 1 and dat[0] == 0 else
            "high" if dat[2] == 1 and dat[0] == 1 else
            ""
        )
        self.telemetry["Gain mode"] = conditionals[0]
        self.telemetry["Effective gain mode"] = conditionals[1]
        self.telemetry["Desired gain mode"] = conditionals[2]
        self.telemetry["Temperature switch high gain to low gain (C)"] = dat[3]
        self.telemetry["Temperature switch low gain to high gain (C)"] = dat[4]
        self.telemetry["Population switch high gain to low gain (%)"] = dat[5]
        self.telemetry["Population switch low gain to high gain (%)"] = dat[6]
        self.telemetry["Gain mode ROI (top left bottom right)"] = dat[7:11]
        self.telemetry["TLinear enabled"] = str(dat[11])
        self.telemetry["TLinear resolution"] = round(dat[12] * -0.09 + 0.1, 2)
        self.telemetry["Spotmeter max temperature (C)"] = round(dat[13] * 0.01 - 273.15, 2)
        self.telemetry["Spotmeter mean temperature (C)"] = round(dat[14] * 0.01 - 273.15, 2)
        self.telemetry["Spotmeter min temperature (C)"] = round(dat[15] * 0.01 - 273.15, 2)
        self.telemetry["Spotmeter population (px)"] = dat[16]
        self.telemetry["Spotmeter ROI (top left bottom right)"] = dat[17:]

    def _get_fps(self):
        rates = 1e9 / np.diff(self.frame_times)
        if all(np.isnan(rates)):
            self.telemetry["Frame Rate (fps)"] = float('nan')
            return
        self.telemetry["Frame Rate (fps)"] = round(float(np.nanmean(rates)), 2)

    def _get_temp_range(self):
        self.telemetry["Minimum Temperature (C)"] = round(float(np.nanmin(self.temperature)), 2)
        self.telemetry["Median Temperature (C)"] = round(float(np.nanmedian(self.temperature)), 2)
        self.telemetry["Maximum Temperature (C)"] = round(float(np.nanmax(self.temperature)), 2)

    def _get_telemetry(self):
        a, b, c = self._unpack()
        self._read_a(a)
        self._read_b(b)
        self._read_c(c)
        self._get_fps()
        self._get_temp_range()

class Capture():
    """
    Directly interfaces with the Lepton to capture and decode frame data.

    Parameters
    ----------
    dev_idx : int
        Integer that specifies which camera device is the Lepton.
    rotation : int, optional
        The clockwise rotation in degrees applied to the captured temperature data. Must be one
        of 0, 90, 180, or 270. The default is 0.

    """
    def __init__(self, dev_idx, rotation = 0):
        if rotation not in _ROTATE_CODES:
            raise ValueError(
                f"rotation must be one of {sorted(_ROTATE_CODES)}, got {rotation}"
            )
        self._dev_idx = dev_idx
        self._rotation = rotation
        self._prev_time = deque([float('nan'), ] * 10)
        self._cap = self._aquire()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def terminate(self):
        """
        Safely releases the Lepton. Must be called at end.

        Returns
        -------
        None

        """
        self._cap.release()

    def read(self, homography = (None, None)):
        """
        Captures and decodes the current Lepton frame.

        Parameters
        ----------
        homography: tuple of (ndarray, list of tuples), optional
            First element: a float ndarray that defines a homography transform to apply to the
            captured temperature data after decoding. When None, no homography is applied.
            The default is None.
            Second element: the destination coordinates of the ROI's corners defined in raw
            temperature coordinates. The default is None.

        Returns
        -------
        frame_data : CapFrame
            The captured and decoded frame data.

        Raises
        ------
        CaptureException
            Raised when the capture fails.
        ShapeException
            Raised when the capture frame shape does not match the expected frame shape.

        """
        # If the frame is captured during camera boot or FFC frame, ignore it and try again
        for _ in range(4):
            frame = CapFrame(
                self._get_frame_data(),
                self._prev_time,
                homography = homography,
                rotation = self._rotation,
            )
            if (not frame.telemetry["FFC state"] == "never commanded" and
                not frame.telemetry["FFC state"] == "in progress"):
                self._prev_time.append(time.monotonic_ns())
                self._prev_time.popleft()
                return frame
            time.sleep(0.5)

        # Capture failure
        msg = "Capture timeout. Reboot the camera and try again."
        raise CaptureTimeout(msg, payload = frame)

    def _aquire(self):
        cap = cv2.VideoCapture(self._dev_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, lepton.SHAPE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, lepton.SHAPE[1] + 2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return cap

    def _get_frame_data(self):
        # Get the current frame data
        res, dat = self._cap.read()

        # In the event of cv2.VideoCapture.read() failure
        if not res:
            msg = "Capture failure. Reboot the camera and try again."
            raise CaptureException(msg, payload = dat)

        # In the event the captured data is of the incorrect type
        cap_shape = (dat.shape[1], dat.shape[0] - 2)
        if cap_shape != lepton.SHAPE:
            msg = (f"Captured shape: {cap_shape} does not equal expected shape: {lepton.SHAPE}.\n"
                   "Are you sure the correct port is selected?\n"
                   "If captured shape is (80, 61), the Lepton may be seated incorrectly.\n"
                   "Reseat it in its socket and try again.")
            raise ShapeException(msg, payload = (cap_shape, lepton.SHAPE))

        return dat
