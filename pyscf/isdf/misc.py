#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

from pyscf import lib


def _benchmark_time(t1, t2, label, rec, rank=None):
    if rank is None or rank == 0:
        lib.logger.debug4(
            rec,
            "%20s wall time: %12.6f CPU time: %12.6f"
            % (label, t2[1] - t1[1], t2[0] - t1[0]),
        )


def _info(rec, rank, msg, *args):
    if rank == 0 or rank is None:
        lib.logger.info(rec, msg, *args)


def _warn(rec, rank, msg, *args):
    if rank == 0 or rank is None:
        lib.logger.warn(rec, msg, *args)


def _debug4(rec, rank, msg, *args):
    if rank == 0 or rank is None:
        lib.logger.debug4(rec, msg, *args)
