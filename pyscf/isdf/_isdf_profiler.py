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

import time
from functools import wraps


class PerformanceProfiler:
    def __init__(self, metrics):
        self.metrics = {metric: {"cpu": 0.0, "wall": 0.0} for metric in metrics}

    def add_time(self, metric, cpu_time, wall_time):
        if metric not in self.metrics:
            raise ValueError(f"Invalid metric: {metric}")
        self.metrics[metric]["cpu"] += cpu_time
        self.metrics[metric]["wall"] += wall_time

    def reset(self):
        for metric in self.metrics:
            self.metrics[metric]["cpu"] = 0.0
            self.metrics[metric]["wall"] = 0.0

    def log_profile(self, logger, use_mpi=False):
        def log_metric(metric):
            cpu_time = self.metrics[metric]["cpu"]
            wall_time = self.metrics[metric]["wall"]
            parallel = cpu_time / wall_time if wall_time != 0 else 0
            return f"Metric {metric:<5} cputime = {cpu_time:16.3f} walltime = {wall_time:16.3f} parallel = {parallel:4.2f}"

        if not use_mpi:
            for metric in self.metrics:
                logger.info(log_metric(metric))
        else:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            if comm.Get_rank() == 0:
                for metric in self.metrics:
                    logger.info(log_metric(metric))
            comm.Barrier()

    def profile(self, metric):
        if metric not in self.metrics:
            raise ValueError(f"Invalid metric: {metric}")

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cpu_start = time.process_time()
                wall_start = time.perf_counter()
                result = func(*args, **kwargs)
                cpu_time = time.process_time() - cpu_start
                wall_time = time.perf_counter() - wall_start
                self.add_time(metric, cpu_time, wall_time)
                return result

            return wrapper

        return decorator


### usage ###

# # Usage example:
# metrics_to_track = ['RgAO', 'V', 'W', 'RgR', 'Ktmp1', 'Ktmp2']
# profiler = PerformanceProfiler(metrics_to_track)
# @profiler.profile('RgAO')
# def some_function():
#     # Function implementation
#     pass
# # To reset profiler
# profiler.reset()
# profiler.log_profile(logger, use_mpi=False)
#
# try:
#     profiler.add_time('RgAO', 1.0, 2.0)
# except ValueError as e:
#     print(f"Error: {e}")
# # Trying to use an invalid metric
# try:
#     @profiler.profile('InvalidMetric')
#     def invalid_function():
#         pass
# except ValueError as e:
#     print(f"Error: {e}")
