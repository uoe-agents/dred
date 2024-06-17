# Copyright (c) 2022-2024 Samuel Garcin
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT



# Replace ProcessPoolExecutor with this for debugging without parallel execution
class MockProcessPoolExecutor():
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def submit(self, fn, *args, **kwargs):
        # execute functions in series without creating threads
        # for easier unit testing
        result = fn(*args, **kwargs)
        return result

    def shutdown(self, wait=True):
        pass


class Parser():
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return self.fn(*args)
