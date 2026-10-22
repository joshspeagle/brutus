"""Keep numba's caches of brutus kernels in step with the whole package.

Numba reuses a cached kernel while the file that defines it is unchanged. A
kernel that calls jitted code from another brutus module, or that reads a
data file at import, would therefore keep a stale compiled copy after that
module or file changed. Importing this module installs cache locators whose
source stamp also covers the package's code and data files, so numba itself
rejects such caches. It must be imported before any brutus kernel is
decorated, because numba picks a kernel's locator then.
"""

import functools
import hashlib
import os
import warnings

from numba.core import caching

# Files whose content can change what a kernel computes.
STAMPED_SUFFIXES = (".py", ".dat")

_installed = {}


def _canonical(path):
    return os.path.normcase(os.path.realpath(path))


@functools.lru_cache
def package_stamp(root):
    """Return a SHA-256 over the relative path and the SHA-256 of the
    content of each code or data file under ``root``, skipping
    ``__pycache__``. A file that cannot be read is left out."""
    digest = hashlib.sha256()
    for directory, subdirs, names in os.walk(root):
        # Sorted in place, so the walk visits them in a fixed order.
        subdirs[:] = sorted(d for d in subdirs if d != "__pycache__")
        for name in sorted(n for n in names if n.endswith(STAMPED_SUFFIXES)):
            path = os.path.join(directory, name)
            try:
                with open(path, "rb") as f:
                    content = hashlib.sha256(f.read()).digest()
            except OSError:  # a dangling editor lock file, for example
                continue
            digest.update(os.path.relpath(path, root).encode() + b"\0" + content)
    return digest.hexdigest()


def _package_locator(base, root):
    """Return a subclass of the numba locator ``base`` that serves only the
    files under ``root`` and adds their ``package_stamp`` to the stamp."""

    class PackageLocator(base):
        package_root = root

        @classmethod
        def from_function(cls, py_func, py_file):
            if not _canonical(py_file).startswith(root + os.sep):
                return None
            return super().from_function(py_func, py_file)

        def get_source_stamp(self):
            return super().get_source_stamp(), package_stamp(root)

    return PackageLocator


def install_cache_locators(root):
    """Put package-wide locators for the files under ``root`` ahead of
    numba's own source-file locators, once per ``root``.

    This uses numba internals. If they are missing, it warns and leaves
    numba's locators alone. Numba's ``NUMBA_CACHE_LOCATOR_CLASSES``
    override bypasses the new locators.
    """
    root = _canonical(root)
    if root in _installed:
        return
    try:
        # numba 0.59 names it _CacheImpl; later versions CacheImpl.
        impl = getattr(caching, "CacheImpl", None) or caching._CacheImpl
        sourced = [
            c
            for c in impl._locator_classes
            if issubclass(c, caching._SourceFileBackedLocatorMixin)
            and not hasattr(c, "package_root")
        ]
    except AttributeError:
        warnings.warn(
            "numba's cache internals changed; cached brutus kernels may not "
            "see changes in the modules they call",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    _installed[root] = [_package_locator(c, root) for c in sourced]
    impl._locator_classes = _installed[root] + list(impl._locator_classes)


install_cache_locators(os.path.dirname(__file__))
