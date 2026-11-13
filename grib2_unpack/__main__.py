import sys
from pathlib import Path

# Support both ``python -m grib2_unpack`` and isolated direct execution:
# ``python -I grib2_unpack/__main__.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from grib2_unpack.cli import main
else:
    from .cli import main

raise SystemExit(main())
