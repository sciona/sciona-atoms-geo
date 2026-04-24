"""Geospatial atom providers.

Atoms derived from geospatial competition solutions and overhead-imagery
pipelines. Framework-agnostic atoms use numpy/OpenCV only. Framework-aware
ports are exposed separately where autograd is required.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
