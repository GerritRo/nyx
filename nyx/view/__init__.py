"""Looking at a nyx sky, rather than measuring it.

Where :mod:`nyx.core` and :mod:`nyx.instrument` answer *what does my
telescope record*, this subpackage answers *what does that sky look like*
over the whole hemisphere, emitter by emitter.  Both sit on the same sky
model and neither depends on the other; the one thing that passes between
them is a value, not a dependency, since an instrument's bandpass is a
perfectly good :class:`~nyx.view.response.SpectralResponse`.

One renderer serves both presentations, and the width of that response
decides which you get:

- one channel -- a telescope's own passband -- gives the HEALPix maps
  behind :meth:`~nyx.core.scene.Scene.sky_view`, drawn by
  :func:`~nyx.view.display.plot_maps`
- three channels -- :meth:`~nyx.view.response.SpectralResponse.cie` --
  give colour, projected through a :class:`~nyx.view.camera.Camera` into
  a photograph

Examples
--------
::

    from nyx.view import Camera, SpectralResponse, ToneCurve, render_sky

    sky = render_sky(obs, atmosphere, emitters, SpectralResponse.cie(geo.wvls))
    sky.plot()                                   # the hemisphere, as panels

    cam = Camera(az=200 * u.deg, alt=25 * u.deg, fov=70 * u.deg, size=(1600, 1000))
    image = cam.expose(sky)                      # linear sRGB
    picture = ToneCurve.fit(image)(image, alpha=cam.horizon())

Plotting needs matplotlib, declared as the ``view`` extra and imported
lazily, so nothing here requires it until you draw something.
"""

from .allsky import render_prepared, render_sky
from .camera import Camera
from .display import ToneCurve, add_noise, plot_maps, tonemap
from .response import XYZ_TO_SRGB, SpectralResponse, cie_xyz, to_linear_srgb
from .skyrender import PointField, SkyRender

__all__ = [
    "XYZ_TO_SRGB",
    "Camera",
    "PointField",
    "SkyRender",
    "SpectralResponse",
    "ToneCurve",
    "add_noise",
    "cie_xyz",
    "plot_maps",
    "render_prepared",
    "render_sky",
    "to_linear_srgb",
    "tonemap",
]
