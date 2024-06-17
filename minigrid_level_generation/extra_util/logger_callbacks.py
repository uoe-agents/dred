# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT

import logging
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class GraphVAELogger(pl.Callback):
    def __init__(self):
        raise NotImplementedError("Implement your logger here.")