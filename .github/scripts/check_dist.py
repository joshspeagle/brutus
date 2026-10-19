#!/usr/bin/env python
"""Check that every trove classifier in the built wheel is valid.

`twine check` does not validate classifiers, but PyPI rejects unknown ones at
upload time with a 400. An invalid classifier broke the 1.1.0 and 1.2.0
releases this way.

Usage: python .github/scripts/check_dist.py [dist directory, default dist]
"""

import glob
import sys
import zipfile
from email.parser import BytesParser

from trove_classifiers import classifiers as VALID


def main():
    dist = sys.argv[1] if len(sys.argv) > 1 else "dist"
    (wheel,) = glob.glob(f"{dist}/*.whl")
    with zipfile.ZipFile(wheel) as z:
        (meta_name,) = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
        meta = BytesParser().parsebytes(z.read(meta_name))
    found = meta.get_all("Classifier") or []
    bad = [c for c in found if c not in VALID]
    for c in bad:
        print(f"::error::invalid trove classifier: {c}")
    if bad:
        sys.exit(1)
    print(f"all {len(found)} classifiers valid")


if __name__ == "__main__":
    main()
