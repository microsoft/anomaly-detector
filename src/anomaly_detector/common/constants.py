# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import enum

TIMESTAMP = "timestamp"
VALUE = "value"


class AlignMode(enum.Enum):
    Inner = 1
    Outer = 2


class FillNAMethod(enum.Enum):
    Previous = 1
    Subsequent = 2
    Linear = 3
    Zero = 4
    Fixed = 5
