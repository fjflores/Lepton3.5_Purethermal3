Usage
=====

Streaming
---------

After the Lepton is seated in the Purethermal board and connected to a device via a USB-C, start streaming the camera using the ``leprun`` command.

.. code-block:: console

   (.venv) C:\Users\username> leprun


When you are finshed streaming, press the ``esc`` while the viewer window is active to terminate.

Recording
---------

To record a stream, use the ``-r`` flag.

.. code-block:: console

   (.venv) C:\Users\username> leprun -r


All generated data is saved to the directory *Lepton_Recordings* which itself is generated in the active directory. After the recording is terminated, data is rendered into a .mp4 video.

When you are finshed recording, press the ``esc`` while the viewing window is active to terminate. After termination, a background process will render the video. This may take several minutes depending on the length of the recording.

Raw Snapshots
-------------

To periodically save the raw uint16 radiometric frame as a 16-bit TIFF, use the ``-sr`` flag with an interval in minutes. It works with or without ``-r``.

.. code-block:: console

   (.venv) C:\Users\username> leprun -sr 5


The first valid frame is saved immediately and then one frame every interval. Images are named *Lepton_Capture_<idx>.tiff* and saved to a *YYYY-MM-DD_HH-MM-SS_Thermal* directory inside the save path. Pixel values are the camera's native centikelvin counts, before dead-pixel repair and before any homography warp. The frame's minimum, median, and maximum temperatures (°C) are also logged to *Temperature_Stats.csv* in the same directory, one row per snapshot.

Help
----

You can use the ``-h`` flag to explore addtional flags and functionality.

.. code-block:: console

   (.venv) C:\Users\username> leprun -h

In Code
-------
To interact with the Lepton in code, use the :ref:`Stream Class <stream-class>`. Example usage is given in :ref:`Examples <examples>`.


