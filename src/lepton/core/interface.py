# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Classes used to render camera stream.
"""

from math import floor, ceil, isnan
from dataclasses import dataclass
from copy import copy
import time
import cv2
import numpy as np
import lepton

class Homography:
    """
    Stores information for a homography transform applied to camera temperature data.

    Parameters
    ----------
    scale: float > 1
        The scale of the viewer window compared to the camera temperature data.
    shape: tuple (width, height), optional
        The shape of the temperature data displayed in the viewer. The default is lepton.SHAPE.

    Attributes
    ----------
    src_verts : list of tuples
        The source coordinates of the ROI's corners defined in viewer window coordinates.
    dst_points : list of tuples
        The destination coordinates of the ROI's corners defined in raw temperature coordinates.
    value : 3 x 3 ndarray
        A 3x3 float ndarray that defines a homography that takes a user defined region of
        interest quadrilateral to a rectangle. The rectangle has user defined aspect ratio and
        is aligned with the axes of the viewer. Its first vertex is in the top left of the
        viewer and all other vertices follow CCW. When no valid roi is locked, has a value of
        None. Scaled to be applied directly to temperature data (width, height), not to
        image data.

    """
    def __init__(self, scale, shape = None):
        self._scale = scale
        self._width, self._height = lepton.SHAPE if shape is None else shape
        self._value = None
        self._time_set = None
        self._src_verts = [None, None, None, None]
        self._dst_pts = None

    def _get_src_pts(self):
        width_factor = (self._width - 1) / (self._scale * self._width - 1)
        height_factor = (self._height - 1) / (self._scale * self._height - 1)
        pts = [(min(max(v[0] * width_factor, 0), self._width - 1),
                min(max(v[1] * height_factor, 0), self._height - 1)) for v in self._src_verts]
        return np.array(pts, dtype=np.float32)

    def _get_dst_pts(self, dst_ar):
        if self._width / dst_ar <= self._height:
            w_dst = self._width
            h_dst = w_dst / dst_ar
        else:
            h_dst = self._height
            w_dst = dst_ar * h_dst
        lft = 0.5 * (self._width - w_dst)
        top = 0.5 * (self._height - h_dst)
        rgt = lft + w_dst - 1
        bot = top + h_dst - 1
        return np.array([(lft, top), (lft, bot), (rgt, bot), (rgt, top)], dtype=np.float32)

    @property
    def src_verts(self):
        """
        The source coordinates of the ROI's corners defined in viewer window coordinates.
        """
        return self._src_verts

    @property
    def dst_points(self):
        """
        The destination coordinates of the ROI's corners defined in raw temperature coordinates.
        """
        return self._dst_pts

    @property
    def value(self):
        """
        A 3x3 float ndarray that defines a homography that takes a user defined region of
        interest quadrilateral to a rectangle. The rectangle has user defined aspect ratio and
        is aligned with the axes of the viewer. Its first vertex is in the top left of the
        viewer and all other vertices follow CCW. When no valid roi is locked, has a value of
        None. Scaled to be applied directly to temperature data (lepton.WIDTH, lepton.HEIGHT),
        not to image data.
        """
        try:
            p = min((time.monotonic() - self._time_set) / 2.0, 1.0)
            if p >= 1.0:
                return self._value
            return (1.0 - p) * np.eye(3) + p * self._value
        except TypeError:
            return None

    def add_vert(self, x, y):
        """
        Adds a new vert to the end of the ROI vertex list.

        Parameters
        ----------
        x : int
            The x position of the vertex in viewer window coordinates.
        y : int
            The y position of the vertex in viewer window coordinates.

        Returns
        -------
        None.

        """
        try:
            self._src_verts[self._src_verts.index(None)] = (x, y)
        except ValueError:
            pass

    def set(self, dst_ar):
        """
        If possible, locks the ROI and calculates the homography transform.

        Parameters
        ----------
        dst_ar : float
            The aspect ratio of the dst coordinates.

        Returns
        -------
        None.

        """
        if dst_ar <= 0.0 or None in self._src_verts:
            return
        self._dst_pts = self._get_dst_pts(dst_ar)
        self._value = cv2.findHomography(self._get_src_pts(), self._dst_pts)[0]
        self._time_set = time.monotonic()

    def reset(self):
        """
        Resets the homography transform and clears the ROI vertices.

        Returns
        -------
        None.

        """
        self._value = None
        self._time_set = None
        self._src_verts = [None, None, None, None]
        self._dst_pts = None

@dataclass
class Viewer:
    """
    The interface that both displays the lepton stream and promotes user interaction through
    keystrokes and mousepresses. The interactions include

        Keys
        ----
        "esc":
            Return the string "ESC" when imshow is called.
        "f":
            Enter or exit set roi mode.
        "r":
            Reset the roi.
        numeric:
            When in set roi mode, defines the aspect ratio of the viewer-aligned rectangle
            to which the roi is taken via the homography transform matrix.
        "enter":
            When in set roi mode, and a valid roi is defined, locks that roi and calculates the
            homography transform matrix.

        Mouse
        -----
        LMB:
            When in set roi mode, defines the next roi vertex as the pixel under the mouse pointer.

    Parameters
    ----------
    name: string
        The name of the viewer window.
    scale: float > 1
        The scale of the viewer window. The exact size (width, height) in pixels is
        (self.shape[0]*self.scale, self.shape[1]*self.scale + lepton.TELEM_HEIGHT)
    shape: tuple (width, height), optional
        The shape of the temperature data displayed in the viewer. The default is lepton.SHAPE.

    Attributes
    ----------
    homography : ndarray or None
        A 3x3 float ndarray that defines a homography.

    """
    name: str
    scale: float
    shape: tuple = lepton.SHAPE

    def __post_init__(self):
        self._homography = Homography(self.scale, self.shape)
        self._show_src = False
        self._nxt_src_vert = None
        self._dst_ar = [None, ] * 3
        shape = (
            round(self.shape[0]*self.scale),
            round(self.shape[1]*self.scale + lepton.TELEM_HEIGHT)
        )
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.name, self._mouse_event)
        self.imshow(np.zeros(shape[::-1] + (4, ), dtype=np.uint8))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def _get_ar(self):
        ar = 0.0
        i = 2
        for digit in reversed(self._dst_ar):
            try:
                ar += (0.1**i) * digit
                i -= 1
            except TypeError:
                pass
        return round(ar, 2)

    def _callback(self):
        # Refresh viewer window and wait 1 ms for user input
        key = cv2.waitKeyEx(1)

        # No keypress return fast
        if key == -1:
            return None

        # Toggle set ROI mode
        if key == ord("f"):
            self._show_src = not self._show_src

        # Reset ROI
        elif key == ord("r"):
            self._dst_ar = [None, ] * 3
            self._homography.reset()

        # Add numeric keys to homography destination aspect ratio string
        elif 48 <= key <= 57 and self._show_src:
            try:
                self._dst_ar[self._dst_ar.index(None)] = key - 48
            except ValueError:
                pass

        # Backspace removes previous number from aspect ratio string
        elif key == 8:
            try:
                self._dst_ar[self._dst_ar.index(None) - 1] = None
            except ValueError:
                self._dst_ar[-1] = None

        # When enter is pressed in set ROI mode, calculate the homography
        elif key == 13 and self._show_src:
            self._homography.set(self._get_ar())
            if not self._homography.value is None:
                self._show_src = False

        # When ESC is pressed, inform user via callback return, other return nothing
        if key == 27:
            return "esc"
        return None

    def _mouse_event(self, event, x, y, *_):
        # Track the mouse movement for setting the next ROI source vert
        if event == cv2.EVENT_MOUSEMOVE and self._show_src:
            self._nxt_src_vert = (x, y)

        # When click, add current mouse position to ROI source verts
        if event == cv2.EVENT_LBUTTONDOWN and self._show_src:
            self._homography.add_vert(x, y)

    def _draw_src(self, image):
        if not self._show_src or not self._homography.value is None:
            return
        lines = [self._homography.src_verts[:-1], self._homography.src_verts[1:]]
        try:
            lines[1][lines[1].index(None)] = self._nxt_src_vert
        except ValueError:
            pass
        lines = list(map(list, zip(*lines)))
        lines.append([lines[-1][-1], lines[0][0]])
        for l in lines:
            if None in l:
                continue
            cv2.line(image, l[0], l[1], (0, 255, 0), 2)
        if self._homography.src_verts[0] is None:
            return
        for l in lines:
            if l[1] is None:
                continue
            cv2.circle(image, l[1], 5, (0, 255, 0), 2)

    def _draw_dst(self, image):
        if not self._show_src or not self._homography.value is None:
            return
        txt = f"ROI AR: {self._get_ar():.02f}"
        (width, height), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        cv2.rectangle(image, (5, 5), (7 + width, 7 + height), (0, 0, 0), -1)
        cv2.putText(
            image,
            txt,
            (6, 6 + height - baseline),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    @property
    def homography(self):
        """
        First element:
        A 3x3 float ndarray that defines a homography that takes a user defined region of interest
        quadrilateral to a rectangle. The rectangle has user defined aspect ratio and is aligned
        with the axes of the viewer. Its first vertex is in the top left of the viewer and all
        other vertices follow CCW. When no valid roi is locked, has a value of None. Scaled to be
        applied directly to temperature data (lepton.WIDTH, lepton.HEIGHT), not to image data
        (lepton.WIDTH*self.scale, lepton.HEIGHT*self.scale + lepton.TELEM_HEIGHT).
        Second element:
        The destination coordinates of the ROI's corners defined in raw temperature coordinates.
        """
        return (self._homography.value, self._homography.dst_points)

    def imshow(self, image):
        """
        Displays an image to the viewer and returns a keypress callback.

        Parameters
        ----------
        image: ndarray
            A uint8 ndarray with shape (m, n, 3).

        Returns
        -------
        keypress_callback: string or None
            The key that was pressed during the viewer refresh. Has possible values "esc" or None.

        """
        if not image is None:
            self._draw_src(image)
            self._draw_dst(image)
            cv2.imshow(self.name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self._callback()

    def terminate(self):
        """
        Terminates the viewer. Must be called when done.

        Returns
        -------
        None

        """
        cv2.destroyWindow(self.name)

class ViewerImage:
    """
    Converts camera frame data to an image that can be displayed by the Viewer.

    Parameters
    ----------
    temperature: ndarray
        A float ndarray that defines the captured temperature. Collected from object of
        type CapFrame.
    telemetry: dict
        A dict that gives the telemetry associated with the captured temperature. Collected from
        object of type CapFrame.
    mask: ndarray
        A Boolean ndarray that provides a feature mask which will be visualized on the image.
        If None, not visualized.
    t0: int
        The camera uptime in ms that was observed at the beginning of the stream. Used to show the
        time since start of stream.
    opts: dict
        A dictionary of options defined at the start of a stream. Must include the keys
        "scale": float > 1: The scale of the viewer window. Used to properly size the image.
        "cmap": matplotlib.colors.ListedColormap: The colormap used to colorize the image.
        "record": bool: When True, adds a recording circle to the top right of the image.
        May optionally include the key
        "temp_range": tuple (min_C, max_C): Fixed temperature range mapped across the colormap.
        When absent or None, each frame is autoscaled to its own min/max.

    """
    def __init__(self, temperature, telemetry, mask, t0, opts):
        self._data = copy(temperature)
        self._data, mask = self._resize(opts["scale"], self._data, mask)
        self._data = self._normalize(self._data, temp_range = opts.get("temp_range"))
        self._data = self._colorize(self._data, opts["cmap"])
        self._data = self._add_mask(self._data, mask)
        self._data = self._touint8(self._data)
        self._data = self._telemetrize(self._data, telemetry, t0, opts["record"])

    def _resize(self, scale, *args):
        resized = []
        for arg in args:
            try:
                resized.append(cv2.resize(
                    arg.astype(np.float32),
                    tuple(round(x*scale) for x in arg.shape[::-1]),
                    interpolation = cv2.INTER_CUBIC
                ))
            except (AttributeError, cv2.error):
                resized.append(None)
        return resized[0] if len(resized) == 1 else tuple(resized)

    def _normalize(self, *args, temp_range = None):
        normed = []
        for arg in args:
            if np.all(np.isnan(arg)):
                normed.append(arg)
                continue
            if temp_range is None:
                mn, mx = np.nanmin(arg), np.nanmax(arg)
                if mn == mx:
                    mx = mn + 1
            else:
                mn, mx = temp_range
            normed.append((arg - mn) / (mx - mn))
        return normed[0] if len(normed) == 1 else tuple(normed)

    def _colorize(self, arr, cmap):
        return cmap(arr)[:, :, :-1]

    def _add_mask(self, arr, mask):
        if not mask is None and not np.all(mask == 0.0):
            mask[mask < 0] = 0
            mask /= ((1.0 / lepton.MASK_ALPHA) * np.nanmax(mask))
            return (1 - mask[:, :, None]) * arr + mask[:, :, None] * lepton.MASK_COLOR
        return arr

    def _touint8(self, arr):
        arr *= 255.0
        arr[arr < 0] = 0
        arr[arr > 255] = 255
        return arr.astype(np.uint8)

    def _ms_2_hmsms(self, t):
        hr = t / 3600000.0
        mn = 60.0 * (hr - floor(hr))
        sc = 60.0 * (mn - floor(mn))
        ms = 1000.0 * (sc - floor(sc))
        return floor(hr), floor(mn), floor(sc), floor(ms)

    def _uptime_str(self, telem, t0):
        hr, mn, sc, ms =  self._ms_2_hmsms(telem["Uptime (ms)"] - t0)
        return f"{hr:01d}:{mn:02d}:{sc:02d}.{ms:03d}"

    def _ffc_uptime_str(self, telem):
        t = telem["Uptime (ms)"] - telem["Uptime at last FFC (ms)"]
        _, mn, sc, _ =  self._ms_2_hmsms(t)
        return f"FFC: {mn:01d}:{sc:02d}"

    def _time_str(self, telem, t0):
        return f"{self._uptime_str(telem, t0)} | {self._ffc_uptime_str(telem)}"

    def _temp_str(self, telem):
        return (
            f"MN: {telem['Minimum Temperature (C)']:>6.2f} | "
            f"MD: {telem['Median Temperature (C)']:>6.2f} | "
            f"MX: {telem['Maximum Temperature (C)']:>6.2f} C"
        )

    def _fps_str(self, telem):
        if isnan(telem['Frame Rate (fps)']):
            return "---"
        return f"FR: {telem['Frame Rate (fps)']:.2f}"

    def _get_telem_pos(self, arr, time_str, fps_str, temp_str):
        (time_width, txt_height), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        (fps_width, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        (temp_width, _), _ = cv2.getTextSize(temp_str, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        img_width = arr.shape[1]
        space = int(round(0.5 * (img_width - 6 - time_width - fps_width - temp_width)))
        space = 0 if space < 0 else space
        bottom_pad = ceil(0.5*(lepton.TELEM_HEIGHT - txt_height)) + 1
        time_pos = (3, arr.shape[0] - bottom_pad)
        fps_pos = (3 + time_width + space, arr.shape[0] - bottom_pad)
        temp_pos = (3 + time_width + fps_width + 2*space, arr.shape[0] - bottom_pad)
        return time_pos, fps_pos, temp_pos

    def _put_text(self, arr, string, position):
        return cv2.putText(
            arr,
            string,
            position,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255,255,255),
            1,
            cv2.LINE_AA,
        )

    def _telemetrize(self, arr, telem, t0, record):
        extra_rows = np.zeros((lepton.TELEM_HEIGHT, ) + arr.shape[1:], dtype=np.uint8)
        arr = np.vstack([arr, extra_rows])
        strings = self._time_str(telem, t0), self._fps_str(telem), self._temp_str(telem)
        positions = self._get_telem_pos(arr, *strings)
        for s, p in zip(strings, positions):
            self._put_text(arr, s, p)
        if telem["FFC state"] == "imminent":
            (ffc_width, ffc_height), _ = cv2.getTextSize("FFC", cv2.FONT_HERSHEY_PLAIN, 1, 1)
            arr = cv2.rectangle(
                arr,
                (5, 5),
                (7 + ffc_width, 7 + ffc_height),
                [0, 0, 0],
                -1
            )
            self._put_text(arr, "FFC", (6, 6 + ffc_height))
        if record:
            arr = cv2.circle(arr, (arr.shape[1] - 10, 10), 5, [238,75,43], -1)
        return arr

    def asuint8(self):
        """
        Returns the generated image as a ndarray of uint8.

        Returns
        -------
        image: ndarray
            A uint8 ndarray of shape
            (lepton.WIDTH*self.scale, lepton.HEIGHT*self.scale + lepton.TELEM_HEIGHT, 3).

        """
        return self._data

    def asfloat32(self):
        """
        Returns the generated image as a ndarray of float32.

        Returns
        -------
        image: ndarray
            A float32 ndarray of shape
            (lepton.WIDTH*self.scale, lepton.HEIGHT*self.scale + lepton.TELEM_HEIGHT, 3).

        """
        return np.array(self._data, dtype=np.float32) / 255.0
