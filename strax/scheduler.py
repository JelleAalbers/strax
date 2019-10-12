from collections import defaultdict
import concurrent.futures as cf
from functools import partial
import os
from random import random
import time
import typing as ty
from types import SimpleNamespace

import psutil
import numpy as np


class Task(SimpleNamespace):
    # submit_to: str                # user, thread, or process
    content: ty.Union[np.ndarray, ty.Callable]
                                  # Task to submit to thread/process pool
                                  # or data to yield to user
    chunk_i: int                  # Incremental chunk number
    # dtypes: ty.Tuple[str]         # Keys / names of data types in result
    is_final: False               # If true, mark TaskGenerator as finished
                                  # on completion
    generator: ty.Any             # Generator that made the task
                                  # (need python 3.7 to annotate properly)

    future: cf.Future = None      # Result will be {dtypename: array, ...}

    def __repr__(self):
        return f"{self.dtypes[0]}:{self.chunk_i}"


class TaskGenerator(SimpleNamespace):
    dtypes: ty.Tuple[str]     # Produced data types
    wants_input: ty.Tuple[ty.Tuple[str, int]]
                                # [(dtype, chunk_i), ...] of inputs
                                # needed to make progress
    is_source = False           # True if loader/producer of data without deps
    submit_to = 'thread'        # thread, process, or user
    parallel = False            # Can we start more than one task at a time?

    priority = 0                # 0 = saver/target, 1 = source, 2 = other
    depth = 0                   # Dependency hops to final target

    chunk_i = 0                 # Chunk number of last emitted task
    need_result_chunk = -1      # Cannot generate new tasks until this result
                                # chunk number has arrived
    input_cache: ty.Dict[str, np.ndarray]
                                # Inputs we could not yet pass to computation
    has_final_task = False
    has_finished = False

    def task_function(self, chunk_i: int, **kwargs) -> ty.Dict[str: np.ndarray]:
        """Function executing the task"""
        raise NotImplementedError

    def __repr__(self):
        return f"TaskGenerator[{self.dtypes[0]}]"

    def inputs_exhausted(self):
        return False

    def is_ready(self):
        return True

    def finish(self):
        pass

    def task(self, inputs) -> Task:
        self.chunk_i += 1
        assert self.chunk_i > self.need_result_chunk
        assert not self.has_finished
        if not self.parallel:
            self.need_result_chunk = self.chunk_i

        if self.is_source:
            assert inputs is None, "Passed inputs to source"
            task_f = partial(self.task_function, chunk_i=self.chunk_i)
        else:
            assert all([k in self.wants_input for k in inputs]), "unwanted input"
            assert all([k in inputs for k in self.wants_input]), "missing input"
            task_f = partial(self.task_function, chunk_i=self.chunk_i, **inputs)
        return Task(content=task_f,
                    generator=self,
                    chunk_i=self.chunk_i,
                    is_final=False)

    def final_task(self) -> Task:
        raise NotImplementedError

    def finish_on_exception(self, exception):
        self.finish()


class StoredData(ty.NamedTuple):
    dtype: str                # Name of data type
    stored: ty.Dict[int, np.ndarray]
                              # [(chunk_i, data), ...]
    seen_by_consumers: ty.Dict[TaskGenerator, int]
                              # Last chunk seen by each of the generators
                              # that need it
    last_contiguous = -1      # Latest chunk that has arrived
                              # for which all previous chunks have arrived
    source_exhausted = False  # Whether source will produce more data

class Scheduler:
    pending_tasks: ty.List[Task]
    stored_data: ty.Dict[str, StoredData]  # {dtypename: StoredData}
    final_target: str
    task_generators: ty.List[TaskGenerator]
    this_process: psutil.Process
    threshold_mb = 1000

    def __init__(self, task_generators):
        self.task_generators = task_generators
        self.task_generators.sort(key=lambda tg: (tg.priority, tg.depth))
        self.pending_tasks = []
        self.stored_data = dict()
        self.this_process = psutil.Process(os.getpid())

        raise NotImplementedError

    def main_loop(self):
        while True:
            self._receive_from_done_tasks()
            task = self._get_new_task()
            if task is None:
                # No more work, except pending tasks
                # and tasks that may follow from their results.
                if not self.pending_tasks:
                    if self.all_exhausted():
                        break  # All done. We win!
                    self.exit_with_exception(RuntimeError(
                        "No available or pending tasks, "
                        "but data is not exhausted!"))
                # Wait for a pending task to complete
            else:
                if task.submit_to == 'user':
                    yield task.content  # Give back a piece of the final target
                else:
                    self._submit_task(task)
                if self.has_available_workers():
                    continue            # Find another task
            self.wait_until_task_done()

    def wait_until_task_done(self):
        while True:
            done, not_done = cf.wait(
                [t.future for t in self.pending_tasks],
                return_when=cf.FIRST_COMPLETED,
                timeout=5)
            if len(done):
                break
            self._emit_status("Waiting for a task to complete")

    def _emit_status(self, msg):
        print(msg)
        print(f"\tPending tasks: {self.pending_tasks}")

    def _receive_from_done_tasks(self):
        still_pending = []
        for task in self.pending_tasks:
            f = task.future
            if not f.done():
                still_pending.append(task)
                continue
            if task.is_final:
                # Note we do this BEFORE the exception checking
                # so we do not retry a failed finishing task.
                task.generator.is_finished = True
            if f.exception() is not None:
                self.exit_with_exception(
                    f.exception(),
                    f"Exception while computing {task.dtypes}:{task.chunk_i}")
            if not task.dtypes:
                continue
            for dtype, result in f.result().items():
                d = self.stored_data[dtype]
                d.stored[task.chunk_i] = f.result()
                if d.last_contiguous == task.chunk_i - 1:
                    d.last_contiguous += 1
        self.pending_tasks = still_pending

    def _submit_task(self, task):
        raise NotImplementedError

    def _get_new_task(self):
        external_waits = []    # TaskGenerators waiting for external conditions
        sources = []           # Sources we could load more data from
        requests_for = defaultdict(int)  # Requests for particular inputs

        for tg in self.task_generators:
            if not tg.is_ready():
                external_waits.append(tg)
                continue
            if not tg.parallel and (
                    self.stored_data[tg.dtypes[0]].last_contiguous
                    < tg.need_result_chunk):
                continue        # Need previous task to finish first
            if tg.is_source:
                sources.append(tg)
                continue

            # Are the required inputs available?
            if tg.inputs_exhausted():
                if not tg.has_finished:
                    if tg.has_final_task:
                        return tg.final_task()   # Submit final task (no inputs)
                    else:
                        tg.has_finished = True
                continue
            for dtype, chunk_id in tg.wants_input:
                if chunk_id not in self.stored_data[dtype]:
                    requests_for[dtype] += 1
                    continue    # Need input to arrive first

            # Yes! Submit the task
            task_inputs = dict()
            for dtype, chunk_i in tg.wants_input:
                self.stored_data[dtype].seen_by_consumers[tg] = chunk_i
                task_inputs[dtype] = self.stored_data[dtype[chunk_i]]
            self._cleanup_cache()

            return tg.task(task_inputs)

        if sources:
            # No computation tasks to do, but we could load new data
            if (self.this_process.memory_info().rss / 1e6 > self.threshold_mb
                    and self.pending_tasks):
                # ... Let's not though; instead wait for current tasks.
                # (We could perhaps also wait for an external condition
                # but in all likelihood a task will complete soon enough)
                return None
            # Load data for the source that is blocking the most tasks
            # Jitter it a bit for better performance on ties..
            # TODO: There is a better way, but I'm too lazy now
            requests_for_source = [(s, sum([requests_for.get(dt, 0)
                                           for dt in s.dtypes]) + random())
                                   for s in sources]
            s, _ = max(requests_for_source, key=lambda q: q[1])
            return s.task(None)

        if external_waits:
            # We could wait for an external condition...
            if len(self.pending_tasks):
                # ... but probably an existing task will complete first
                return None
            # ... so let's do that.
            self._emit_status(f"{external_waits} waiting on external condition")
            time.sleep(5)
            # TODO: for very long waits this will trip the recursion limit!
            return self._get_new_task()

        # No work to do. Maybe a pending task will still generate some though.
        return None

    def cleanup_cache(self):
        """Remove any data from our stored_data that has been seen
        by all the consumers"""
        for d in self.stored_data:
            if not len(d.seen_by_consumers):
                raise RuntimeError(f"{d} is not consumed by anyone??")
            seen_by_all = min(d.seen_by_consumers.values())
            d.stored = {chunk_i: data
                        for chunk_i, data in d.stored.items()
                        if chunk_i > seen_by_all}

    def exit_with_exception(self, exception, extra_message=''):
        print(extra_message)
        for tg in self.task_generators:
            if not tg.finished:
                try:
                    tg.finish_on_exception(exception)
                except Exception as e:
                    print(f"Exceptional shutdown of {tg} failed")
                    print("Got eanother xception: {e}")
                    pass   # These are exceptional times...
            raise exception

    def all_exhausted(self):
        raise NotImplementedError