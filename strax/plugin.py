"""Plugin system for strax

A 'plugin' is something that outputs an array and gets arrays
from one or more other plugins.
"""
from concurrent.futures import wait
from enum import IntEnum
import inspect
import itertools
import logging
import time
import typing
from warnings import warn
from immutabledict import immutabledict
import numpy as np

import strax
export, __all__ = strax.exporter()


@export
class SaveWhen(IntEnum):
    """Plugin's preference for having it's data saved"""
    NEVER = 0         # Throw an error if the user lists it
    EXPLICIT = 1      # Save ONLY if the user lists it explicitly
    TARGET = 2        # Save if the user asks for it as a final target
    ALWAYS = 3        # Save even if the user does not list it


@export
class InputTimeoutExceeded(Exception):
    pass


@export
class PluginGaveWrongOutput(Exception):
    pass


@export
class Plugin:
    """Plugin containing strax computation

    You should NOT instantiate plugins directly.
    Do NOT add unpickleable things (e.g. loggers) as attributes.
    """
    __version__ = '0.0.0'

    # For multi-output plugins these should be (immutable)dicts
    data_kind: typing.Union[str, immutabledict, dict]
    dtype: typing.Union[tuple, np.dtype, immutabledict, dict]

    depends_on: tuple
    provides: tuple
    input_buffer: typing.Dict[str, strax.Chunk]
    
    # Needed for plugins which are inherited from an already existing plugins,
    # indicates such an inheritance.
    child_plugin = False
    
    compressor = 'blosc'

    rechunk_on_save = True    # Saver is allowed to rechunk
    # How large (uncompressed) should re-chunked chunks be?
    # Meaningless if rechunk_on_save is False
    chunk_target_size_mb = strax.default_chunk_size_mb


    # For a source with online input (e.g. DAQ readers), crash if no new input
    # has appeared for this many seconds
    # This should be smaller than the mailbox timeout (which is intended as
    # a deep fallback)
    input_timeout = 80

    save_when = SaveWhen.ALWAYS

    # Instructions how to parallelize
    #   False: never parallellize;
    #   'process': use processpool;
    #   'thread' (or just True): use threadpool.
    parallel = False              # For the computation itself

    # Maximum number of output messages
    max_messages = None   # use default

    # Do not specify attributes below

    # Set using the takes_config decorator
    takes_config = immutabledict()

    # These are set on plugin initialization, which is done in the core
    run_id: str
    run_i: int
    config: typing.Dict
    deps: typing.Dict       # Dictionary of dependency plugin instances

    compute_takes_chunk_i = False    # Autoinferred, no need to set yourself
    compute_takes_start_end = False

    def __init__(self):
        if not hasattr(self, 'depends_on'):
            raise ValueError('depends_on not provided for '
                             f'{self.__class__.__name__}')

        self.depends_on = strax.to_str_tuple(self.depends_on)

        # Store compute parameter names, see if we take chunk_i too
        compute_pars = list(
            inspect.signature(self.compute).parameters.keys())
        if 'chunk_i' in compute_pars:
            self.compute_takes_chunk_i = True
            del compute_pars[compute_pars.index('chunk_i')]
        if 'start' in compute_pars:
            if 'end' not in compute_pars:
                raise ValueError(f"Compute of {self} takes start, "
                                 f"so it should also take end.")
            self.compute_takes_start_end = True
            del compute_pars[compute_pars.index('start')]
            del compute_pars[compute_pars.index('end')]

        self.compute_pars = compute_pars
        self.input_buffer = dict()

    def fix_dtype(self):
        if not hasattr(self, 'dtype'):
            self.dtype = self.infer_dtype()

        if self.multi_output:
            # Convert to a dict of numpy dtypes
            if (not hasattr(self, 'data_kind')
                    or not isinstance(self.data_kind, (dict, immutabledict))):
                raise ValueError(
                    f"{self.__class__.__name__} has multiple outputs and "
                    "must declare its data kind as a dict: "
                    "{dtypename: data kind}.")
            if not isinstance(self.dtype, dict):
                raise ValueError(
                    f"{self.__class__.__name__} has multiple outputs, so its "
                    "dtype must be specified as a dict: {output: dtype}.")
            self.dtype = {k: strax.to_numpy_dtype(dt)
                          for k, dt in self.dtype.items()}
        else:
            # Convert to a numpy dtype
            self.dtype = strax.to_numpy_dtype(self.dtype)

        # Check required time information is present
        for d in self.provides:
            fieldnames = self.dtype_for(d).names
            ok = 'time' in fieldnames and (
                    ('dt' in fieldnames and 'length' in fieldnames)
                    or 'endtime' in fieldnames)
            if not ok:
                raise ValueError(
                    f"Missing time and endtime information for {d}")

    @property
    def multi_output(self):
        return len(self.provides) > 1

    def setup(self):
        """Hook if plugin wants to do something on initialization
        """
        pass

    def infer_dtype(self):
        """Return dtype of computed data;
        used only if no dtype attribute defined"""
        # Don't raise NotImplementedError, IDE will complain you're not
        # implementing all abstract methods...
        raise RuntimeError("No infer dtype method defined")

    def version(self, run_id=None):
        """Return version number applicable to the run_id.
        Most plugins just have a single version (in .__version__)
        but some may be at different versions for different runs
        (e.g. time-dependent corrections).
        """
        return self.__version__

    def __repr__(self):
        return self.__class__.__name__

    def dtype_for(self, data_type):
        """
        Provide the dtype of one of the provide arguments of the plugin.
        NB: does not simply provide the dtype of any datatype but must
        be one of the provide arguments known to the plugin.
        """
        if self.multi_output:
            if data_type in self.dtype:
                return self.dtype[data_type]
            else:
                raise ValueError(f'dtype_for provides the dtype of one of the '
                                 f'provide datatypes specified in this plugin '
                                 f'{data_type} is not provided by this plugin')
        return self.dtype

    def can_rechunk(self, data_type):
        if isinstance(self.rechunk_on_save, bool):
            return self.rechunk_on_save
        if isinstance(self.rechunk_on_save, (dict, immutabledict)):
            return self.rechunk_on_save[data_type]
        raise ValueError("rechunk_on_save must be a bool or an immutabledict")

    def empty_result(self):
        if self.multi_output:
            return {d: np.empty(0, self.dtype_for(d))
                    for d in self.provides}
        return np.empty(0, self.dtype)

    def data_kind_for(self, data_type):
        if self.multi_output:
            return self.data_kind[data_type]
        return self.data_kind

    def metadata(self, run_id, data_type):
        """Metadata to save along with produced data"""
        if not data_type in self.provides:
            raise RuntimeError(f"{data_type} not in {self.provides}?")
        return dict(
            run_id=run_id,
            data_type=data_type,
            data_kind=self.data_kind_for(data_type),
            dtype=self.dtype_for(data_type),
            lineage_hash=strax.DataKey(
                run_id, data_type, self.lineage).lineage_hash,
            compressor=self.compressor,
            lineage=self.lineage,
            chunk_target_size_mb=self.chunk_target_size_mb)

    def dependencies_by_kind(self):
        """Return dependencies grouped by data kind
        i.e. {kind1: [dep0, dep1], kind2: [dep, dep]}
        :param require_time: If True, one dependency of each kind
        must provide time information. It will be put first in the list.

        If require_time is omitted, we will require time only if there is
        more than one data kind in the dependencies.
        """
        return strax.group_by_kind(
            self.depends_on,
            plugins=self.deps)

    def is_ready(self, chunk_i):
        """Return whether the chunk chunk_i is ready for reading.
        Returns True by default; override if you make an online input plugin.
        """
        return True

    def source_finished(self):
        """Return whether all chunks the plugin wants to read have been written.
        Only called for online input plugins.
        """
        # Don't raise NotImplementedError, IDE complains
        raise RuntimeError("source_finished called on a regular plugin")

    def _fetch_chunk(self, d, iters, check_end_not_before=None):
        """Add a chunk of the datatype d to the input buffer.
        Return True if this succeeded, False if the source is exhausted.
        :param d: data type to fetch
        :param iters: iterators that produce data
        :param check_end_not_before: Raise a runtimeError if the source 
        is exhausted, but the input buffer ends before this time.
        """
        try:
            # print(f"Fetching {d} in {self}, hope to see {hope_to_see}")
            self.input_buffer[d] = strax.Chunk.concatenate(
                [self.input_buffer[d], next(iters[d])])
            # print(f"Fetched {d} in {self}, "
            #      f"now have {self.input_buffer[d]}")
            return True
        except StopIteration:
            # print(f"Got StopIteration while fetching for {d} in {self}")
            if (check_end_not_before is not None
                    and self.input_buffer[d].end < check_end_not_before):
                raise RuntimeError(
                    f"Tried to get data until {check_end_not_before}, but {d} "
                    f"ended prematurely at {self.input_buffer[d].end}")
            return False

    def iter(self, iters, executor=None):
        """Iterate over dependencies and yield results

        :param iters: dict with iterators over dependencies
        :param executor: Executor to punt computation tasks to. If None,
            will compute inside the plugin's thread.
        """
        pending_futures = []
        last_input_received = time.time()
        self.input_buffer = {d: None
                             for d in self.depends_on}

        # Fetch chunks from all inputs. Whoever is the slowest becomes the
        # pacemaker
        pacemaker = None
        _end = float('inf')
        for d in self.depends_on:
            self._fetch_chunk(d, iters)
            if self.input_buffer[d] is None:
                raise ValueError(f'Cannot work with empty input buffer {self.input_buffer}')
            if self.input_buffer[d].end < _end:
                pacemaker = d
                _end = self.input_buffer[d].end

        for chunk_i in itertools.count():

            # Online input support
            while not self.is_ready(chunk_i):
                if self.source_finished():
                    # Chunk_i does not exist. We are done.
                    print("Source finished!")
                    self.cleanup(iters, wait_for=pending_futures)
                    return

                if time.time() > last_input_received + self.input_timeout:
                    raise InputTimeoutExceeded(
                        f"{self.__class__.__name__}:{id(self)} waited for "
                        f"more  than {self.input_timeout} sec for arrival of "
                        f"input chunk {chunk_i}, and has given up.")

                print(f"{self.__class__.__name__} with object id: {id(self)} "
                      f"waits for chunk {chunk_i}")
                time.sleep(2)
            last_input_received = time.time()

            if pacemaker is None:
                inputs_merged = dict()
            else:
                if chunk_i != 0:
                    # Fetch the pacemaker, to figure out when this chunk ends
                    # (don't do it for chunk 0, for which we already fetched)
                    if not self._fetch_chunk(pacemaker, iters):
                        # Source exhausted. Cleanup will do final checks.
                        self.cleanup(iters, wait_for=pending_futures)
                        return
                this_chunk_end = self.input_buffer[pacemaker].end

                inputs = dict()
                # Fetch other inputs (when needed)
                for d in self.depends_on:
                    if d != pacemaker:
                        while (self.input_buffer[d] is None
                               or self.input_buffer[d].end < this_chunk_end):
                            self._fetch_chunk(
                                d, iters,
                                check_end_not_before=this_chunk_end)
                    inputs[d], self.input_buffer[d] = \
                        self.input_buffer[d].split(
                            t=this_chunk_end,
                            allow_early_split=True)
                # If any of the inputs were trimmed due to early splits,
                # trim the others too.
                # In very hairy cases this can take multiple passes.
                # TODO: can we optimize this, or code it more elegantly?
                max_passes_left = 10
                while max_passes_left > 0:
                    this_chunk_end = min([x.end for x in inputs.values()]
                                         + [this_chunk_end])
                    if len(set([x.end for x in inputs.values()])) <= 1:
                        break
                    for d in self.depends_on:
                        inputs[d], back_to_buffer = \
                            inputs[d].split(
                                t=this_chunk_end,
                                allow_early_split=True)
                        self.input_buffer[d] = strax.Chunk.concatenate(
                            [back_to_buffer, self.input_buffer[d]])
                    max_passes_left -= 1
                else:
                    raise RuntimeError(
                        f"{self} was unable to get time-consistent "
                        f"inputs after ten passess. Inputs: \n{inputs}\n"
                        f"Input buffer:\n{self.input_buffer}")

                # Merge inputs of the same kind
                inputs_merged = {
                    kind: strax.Chunk.merge([inputs[d] for d in deps_of_kind])
                    for kind, deps_of_kind in self.dependencies_by_kind().items()}

            # Submit the computation
            # print(f"{self} calling with {inputs_merged}")
            if self.parallel and executor is not None:
                new_future = executor.submit(
                    self.do_compute,
                    chunk_i=chunk_i,
                    **inputs_merged)
                pending_futures.append(new_future)
                pending_futures = [f for f in pending_futures if not f.done()]
                to_yield = new_future
            else:
                to_yield = self.do_compute(chunk_i=chunk_i, **inputs_merged)

            try:
                yield to_yield
            except Exception:
                # Sender has .throw()'n an exception in rather than asking
                # for another message
                self.cleanup(iters, wait_for=pending_futures)
                raise

        raise RuntimeError("This cannot happen.")

    def cleanup(self,
                iters: typing.Dict[str, typing.Iterable],
                wait_for):
        # The wait_for option is only used in child classes;
        # A standard plugin doesn't need to do anything with the computation
        # future results.

        # Check all sources are exhausted.
        # This is more than a check though -- it ensure the content of
        # all sources are requested all the way (including the final
        # Stopiteration), as required by lazy-mode processing requires
        for d in iters.keys():
            if self._fetch_chunk(d, iters):
                raise RuntimeError(
                    f"Plugin {d} terminated without fetching last {d}!")

        # This can happen especially in time range selections
        if int(self.save_when) != strax.SaveWhen.NEVER:
            for d, buffer in self.input_buffer.items():
                # Check the input buffer is empty
                if buffer is not None and len(buffer):
                    raise RuntimeError(
                        f"Plugin {d} terminated with leftover {d}: {buffer}")

    def _check_dtype(self, x, d=None):
        # There is an additional 'last resort' data type check
        # in the chunk initialization.
        # This one is broader and gives a more context-aware message.
        if d is None:
            assert not self.multi_output
            d = self.provides[0]
        pname = self.__class__.__name__
        if not isinstance(x, np.ndarray):
            raise strax.PluginGaveWrongOutput(
                f"Plugin {pname} did not deliver "
                f"data type {d} as promised.\n"
                f"Delivered a {type(x)}")

        expect = strax.remove_titles_from_dtype(self.dtype_for(d))
        if not isinstance(expect, np.dtype):
            raise ValueError(f"Plugin {pname} expects {expect} as dtype??")
        got = strax.remove_titles_from_dtype(x.dtype)
        if got != expect:
            raise strax.PluginGaveWrongOutput(
                f"Plugin {pname} did not deliver "
                f"data type {d} as promised.\n"
                f"Promised:  {expect}\n"
                f"Delivered: {got}.")

    def do_compute(self, chunk_i=None, **kwargs):
        """Wrapper for the user-defined compute method

        This is the 'job' that gets executed in different processes/threads
        during multiprocessing
        """
        for k, v in kwargs.items():
            if not isinstance(v, strax.Chunk):
                raise RuntimeError(
                    f"do_compute of {self.__class__.__name__} got a {type(v)} "
                    f"instead of a strax Chunk for {k}")

        if len(kwargs):
            # Check inputs describe the same time range
            tranges = {k: (v.start, v.end) for k, v in kwargs.items()}
            start, end = list(tranges.values())[0]

            # For non-saving plugins, don't be strict, just take whatever
            # endtimes are available and don't check time-consistency
            if int(self.save_when) == strax.SaveWhen.NEVER:
                # </start>This warning/check will be deleted, see UserWarning
                if len(set(tranges.values())) != 1:
                    end = max([v.end for v in kwargs.values()])  # Don't delete
                    message = (
                        f"New feature, we are ignoring inconsistent the "
                        f"possible ValueError in time ranges for "
                        f"{self.__class__.__name__} of inputs: {tranges}"
                        f"because this occurred in a save_when.NEVER "
                        f"plugin. Report any findings in "
                        f"github.com/AxFoundation/strax/issues/247")
                    warn(message, UserWarning)
                # This block will be deleted </end>
            elif len(set(tranges.values())) != 1:
                message = (f"{self.__class__.__name__} got inconsistent time "
                           f"ranges of inputs: {tranges}")
                raise ValueError(message)
        else:
            # This plugin starts from scratch
            start, end = None, None

        kwargs = {k: v.data for k, v in kwargs.items()}
        if self.compute_takes_chunk_i:
            kwargs['chunk_i'] = chunk_i
        if self.compute_takes_start_end:
            kwargs['start'] = start
            kwargs['end'] = end
        result = self.compute(**kwargs)

        return self._fix_output(result, start, end)

    def _fix_output(self, result, start, end, _dtype=None):
        if self.multi_output and _dtype is None:
            if not isinstance(result, dict):
                raise ValueError(
                    f"{self.__class__.__name__} is multi-output and should "
                    "provide a dict output {dtypename: result}")
            return {d: self._fix_output(result[d], start, end, _dtype=d)
                    for d in self.provides}

        if _dtype is None:
            assert not self.multi_output
            _dtype = self.provides[0]

        if not isinstance(result, strax.Chunk):
            if start is None:
                assert len(self.depends_on) == 0
                raise ValueError(
                    "Plugins without dependencies must return full strax "
                    f"Chunks, but {self.__class__.__name__} produced a "
                    f"{type(result)}!")
            if isinstance(result, dict) and len(result) == 1:
                raise ValueError(
                    f'Ran into single key results dict with key: '
                    f'{list(result.keys())}, cannot convert this to array of '
                    f'dtype {self.dtype_for(_dtype)}.\nSee '
                    f'github.com/AxFoundation/strax/issues/238 for more info')
            result = strax.dict_to_rec(result, dtype=self.dtype_for(_dtype))
            self._check_dtype(result, _dtype)
            result = self.chunk(
                start=start,
                end=end,
                data_type=_dtype,
                data=result)
        return result

    def chunk(self, *, start, end, data, data_type=None, run_id=None):
        if data_type is None:
            if self.multi_output:
                raise ValueError("Must give data_type when making chunks from "
                                 "a multi-output plugin")
            data_type = self.provides[0]
        if run_id is None:
            run_id = self.run_id
        return strax.Chunk(
            start=start,
            end=end,
            run_id=run_id,
            data_kind=self.data_kind_for(data_type),
            data_type=data_type,
            dtype=self.dtype_for(data_type),
            data=data,
            target_size_mb=self.chunk_target_size_mb)

    def compute(self, **kwargs):
        raise NotImplementedError


##
# Special plugins
##

@export
class OverlapWindowPlugin(Plugin):
    """Plugin whose computation depends on having its inputs extend
    a certain window on both sides.

    Current implementation assumes:
    - All inputs are sorted by *endtime*. Since everything in strax is sorted
    by time, this only works for disjoint intervals such as peaks or events,
    but NOT records!
    - You must read time info for your data kind, or create a new data kind.
    """
    parallel = False

    def __init__(self):
        super().__init__()
        self.cached_input = {}
        self.cached_results = None
        self.sent_until = 0
        # This guy can have a logger, it's not parallelized anyway
        self.log = logging.getLogger(self.__class__.__name__)

    def get_window_size(self):
        """Return the required window size in nanoseconds"""
        raise NotImplementedError

    def iter(self, iters, executor=None):
        yield from super().iter(iters, executor=executor)

        # Yield final results, kept at bay in fear of a new chunk
        if self.cached_results is not None:
            yield self.cached_results

    def do_compute(self, chunk_i=None, **kwargs):
        if not len(kwargs):
            raise RuntimeError("OverlapWindowPlugin must have a dependency")

        # Add cached inputs to compute arguments
        for k, v in kwargs.items():
            if len(self.cached_input):
                kwargs[k] = strax.Chunk.concatenate(
                    [self.cached_input[k], v])

        # Compute new results
        result = super().do_compute(chunk_i=chunk_i, **kwargs)

        # Throw away results we already sent out
        _, result = result.split(t=self.sent_until,
                                 allow_early_split=False)

        # When does this batch of inputs end?
        ends = [v.end for v in kwargs.values()]
        if not len(set(ends)) == 1:
            raise RuntimeError(
                f"OverlapWindowPlugin got incongruent inputs: {kwargs}")
        end = ends[0]

        # When can we no longer trust our results?
        # Take slightly larger windows for safety: it is very easy for me
        # (or the user) to have made an off-by-one error
        invalid_beyond = int(end - self.get_window_size() - 1)

        # Prepare to send out valid results, cache the rest
        # Do not modify result anymore after this
        # Note result.end <= invalid_beyond, with equality if there are
        # no overlaps
        result, self.cached_results = result.split(t=invalid_beyond,
                                                   allow_early_split=True)
        self.sent_until = result.end

        # Cache a necessary amount of input for next time
        # Again, take a bit of overkill for good measure
        cache_inputs_beyond = int(self.sent_until
                                  - 2 * self.get_window_size() - 1)
        for k, v in kwargs.items():
            _, self.cached_input[k] = v.split(t=cache_inputs_beyond,
                                              allow_early_split=True)

        return result


@export
class LoopPlugin(Plugin):
    """Plugin that disguises multi-kind data-iteration by an event loop
    """
    def compute(self, **kwargs):
        # If not otherwise specified, data kind to loop over
        # is that of the first dependency (e.g. events)
        # Can't be in __init__: deps not initialized then
        if hasattr(self, 'loop_over'):
            loop_over = self.loop_over
        else:
            loop_over = self.deps[self.depends_on[0]].data_kind
        if not isinstance(loop_over, str):
            raise TypeError("Please add \"loop_over = <base>\""
                            " to your plugin definition")

        # Group into lists of things (e.g. peaks)
        # contained in the base things (e.g. events)
        base = kwargs[loop_over]
        if len(base) > 1:
            assert np.all(base[1:]['time'] >= strax.endtime(base[:-1])), \
                f'{base}s overlap'

        for k, things in kwargs.items():
            # Check for sorting
            difs = np.diff(things['time'])
            if difs.min(initial=0) < 0:
                i_bad = np.argmin(difs)
                examples = things[i_bad-1:i_bad+3]
                t0 = examples['time'].min()
                raise ValueError(
                    f'Expected {k} to be sorted, but found ' +
                    str([(x['time'] - t0, strax.endtime(x) - t0)
                         for x in examples]))

            if k != loop_over:
                r = strax.split_by_containment(things, base)
                if len(r) != len(base):
                    raise RuntimeError(f"Split {k} into {len(r)}, "
                                       f"should be {len(base)}!")
                kwargs[k] = r

        if self.multi_output:
            # This is the a-typical case. Most of the time you just have
            # one output. Just doing the same as below but this time we
            # need to create a dict for the outputs.
            # NB: both outputs will need to have the same length as the
            # base!
            results = {k: np.zeros(len(base), dtype=self.dtype[k]) for k in self.provides}
            deps_by_kind = self.dependencies_by_kind()

            for i, base_chunk in enumerate(base):
                res = self.compute_loop(base_chunk,
                                        **{k: kwargs[k][i]
                                           for k in deps_by_kind
                                           if k != loop_over})
                if not isinstance(res, (dict, immutabledict)):
                    raise AttributeError('Please provide result in '
                                         'compute loop as dict')
                # Convert from dict to array row:
                for provides, r in res.items():
                    for k, v in r.items():
                        if np.shape(v) != np.shape(results[provides][i][k]):
                            # Make sure that the buffer length as
                            # defined by the base matches the output of
                            # the compute argument.
                            raise ValueError(
                                f'{provides} returned an improper length array '
                                f'that is not equal to the {loop_over} '
                                'data-kind! Are you sure a LoopPlugin is the '
                                'right Plugin for your application?')
                        results[provides][i][k] = v
        else:
            # Normally you end up here were we are going to loop over
            # base and add the results to the right format.
            results = np.zeros(len(base), dtype=self.dtype)
            deps_by_kind = self.dependencies_by_kind()

            for i, base_chunk in enumerate(base):
                r = self.compute_loop(base_chunk,
                                      **{k: kwargs[k][i]
                                         for k in deps_by_kind
                                         if k != loop_over})
                if not isinstance(r, (dict, immutabledict)):
                    raise AttributeError('Please provide result in '
                                         'compute loop as dict')
                # Convert from dict to array row:
                for k, v in r.items():
                    results[i][k] = v
        return results

    def compute_loop(self, *args, **kwargs):
        raise NotImplementedError


@export
class CutPlugin(Plugin):
    """Generate a plugin that provides a boolean for a given cut specified by 'cut_by'"""
    save_when = SaveWhen.NEVER

    def __init__(self):
        super().__init__()

        _name = strax.camel_to_snake(self.__class__.__name__)
        if not hasattr(self, 'provides'):
            self.provides = _name
        if not hasattr(self, 'cut_name'):
            self.cut_name = _name
        if not hasattr(self, 'cut_description'):
            _description = _name
            if 'cut_' not in _description:
                _description = 'Cut by ' + _description
            else:
                _description = " ".join(_description.split("_"))
            self.cut_description = _description

    def infer_dtype(self):
        dtype = [(self.cut_name, np.bool_, self.cut_description)]
        # Alternatively one could use time_dt_fields for low level plugins.
        dtype = dtype + strax.time_fields
        return dtype

    def compute(self, **kwargs):
        if hasattr(self, 'cut_by'):
            cut_by = self.cut_by
        else:
            raise NotImplementedError(f"{self.cut_name} does not have attribute 'cut_by'")

        # Take shape of the first data_type like in strax.plugin
        buff = list(kwargs.values())[0]

        # Generate result buffer
        r = np.zeros(len(buff), self.dtype)
        r['time'] = buff['time']
        r['endtime'] = strax.endtime(buff)
        r[self.cut_name] = cut_by(**kwargs)
        return r

    def cut_by(self, **kwargs):
        # This should be provided by the user making a CutPlugin
        raise NotImplementedError()


##
# "Plugins" for internal use
# These do not actually do computations, but do other tasks
# for which posing as a plugin is helpful.
# Do not subclass unless you know what you are doing..
##

@export
class MergeOnlyPlugin(Plugin):
    """Plugin that merges data from its dependencies
    """
    save_when = SaveWhen.NEVER

    def infer_dtype(self):
        deps_by_kind = self.dependencies_by_kind()
        if len(deps_by_kind) != 1:
            raise ValueError("MergeOnlyPlugins can only merge data "
                             "of the same kind, but got multiple kinds: "
                             + str(deps_by_kind))

        return strax.merged_dtype([
            self.deps[d].dtype_for(d)
            # Sorting is needed here to match what strax.Chunk does in merging
            for d in sorted(self.depends_on)])

    def compute(self, **kwargs):
        return kwargs[list(kwargs.keys())[0]]


@export
class ParallelSourcePlugin(Plugin):
    """An plugin that inlines the computations of other plugins
    and the saving of their results.

    This evades data transfer (pickling and/or memory copy) penalties
    while multiprocessing.
    """
    parallel = 'process'

    @classmethod
    def inline_plugins(cls, components, start_from, log):
        plugins = components.plugins.copy()

        sub_plugins = {start_from: plugins[start_from]}
        del plugins[start_from]

        # Gather all plugins that do not rechunk and which branch out as a
        # simple tree from the input plugin.
        # We'll run these all together in one process.
        while True:
            # Scan for plugins we can inline
            for p in plugins.values():
                if (p.parallel
                        and all([d in sub_plugins for d in p.depends_on])):
                    for d in p.provides:
                        sub_plugins[d] = p
                        if d in plugins:
                            del plugins[d]
                    # Rescan
                    break
            else:
                # No more plugins we can inline
                break

        if len(set(list(sub_plugins.values()))) == 1:
            # Just one plugin to inline: no use
            log.debug("Just one plugin to inline: skipping")
            return components

        # Which data types should we output? Three cases follow.
        outputs_to_send = set()

        # Case 1. Requested as a final target
        for p in sub_plugins.values():
            outputs_to_send.update(set(components.targets)
                                   .intersection(set(p.provides)))
        # Case 2. Requested by a plugin we did not inline
        for d, p in plugins.items():
            outputs_to_send.update(set(p.depends_on))
        outputs_to_send &= sub_plugins.keys()

        # Inline savers that do not require rechunking
        savers = components.savers
        sub_savers = dict()
        for p in sub_plugins.values():
            for d in p.provides:
                if d not in savers:
                    continue
                if p.can_rechunk(d):
                    # Case 3. has a saver we can't inline
                    outputs_to_send.add(d)
                    continue

                remaining_savers = []
                for s_i, s in enumerate(savers[d]):
                    if not s.allow_fork:
                        # Case 3 again, cannot inline saver
                        outputs_to_send.add(d)
                        remaining_savers.append(s)
                        continue
                    if d not in sub_savers:
                        sub_savers[d] = []
                    s.is_forked = True
                    sub_savers[d].append(s)
                savers[d] = remaining_savers

                if not len(savers[d]):
                    del savers[d]

        p = cls(depends_on=sub_plugins[start_from].depends_on)
        p.sub_plugins = sub_plugins
        assert len(outputs_to_send)
        p.provides = tuple(outputs_to_send)
        p.sub_savers = sub_savers
        p.start_from = start_from
        if p.multi_output:
            p.dtype = {}
            for d in outputs_to_send:
                if d in p.sub_plugins:
                    p.dtype[d] = p.sub_plugins[d].dtype_for(d)
                else:
                    log.debug(f'Finding plugin that provides {d}')
                    # Need to do some more work to get the plugin that
                    # provides this data-type.
                    for sp in p.sub_plugins.values():
                        if d in sp.provides:
                            log.debug(f'{sp} provides {d}')
                            p.dtype[d] = sp.dtype_for(d)
                            break
        else:
            to_send = list(outputs_to_send)[0]
            p.dtype = p.sub_plugins[to_send].dtype_for(to_send)
        for d in p.provides:
            plugins[d] = p
        p.deps = {d: plugins[d] for d in p.depends_on}

        log.debug(f"Inlined plugins: {p.sub_plugins}."
                  f"Inlined savers: {p.sub_savers}")

        return strax.ProcessorComponents(
            plugins, components.loaders, savers, components.targets)

    def __init__(self, depends_on):
        self.depends_on = depends_on
        super().__init__()

    def source_finished(self):
        return self.sub_plugins[self.start_from].source_finished()

    def is_ready(self, chunk_i):
        return self.sub_plugins[self.start_from].is_ready(chunk_i)

    def do_compute(self, chunk_i=None, **kwargs):
        results = kwargs

        # Run the different plugin computations
        while True:
            for output_name, p in self.sub_plugins.items():
                if output_name in results:
                    continue
                if any([d not in results for d in p.depends_on]):
                    continue
                compute_kwargs = dict(chunk_i=chunk_i)

                for kind, d_of_kind in p.dependencies_by_kind().items():
                    compute_kwargs[kind] = strax.Chunk.merge(
                        [results[d] for d in d_of_kind])

                # Store compute result(s)
                r = p.do_compute(**compute_kwargs)
                if p.multi_output:
                    for d in r:
                        results[d] = r[d]
                else:
                    results[output_name] = r

                # Rescan plugins to see if we can compute anything more
                break

            else:
                # Nothing further to compute
                break
        for d in self.provides:
            assert d in results, f"Output {d} missing!"

        # Save anything we can through the inlined savers
        for d, savers in self.sub_savers.items():
            for s in savers:
                s.save(chunk=results[d], chunk_i=chunk_i)

        # Remove results we do not need to send
        for d in list(results.keys()):
            if d not in self.provides:
                del results[d]

        if self.multi_output:
            for k in self.provides:
                assert k in results
                assert isinstance(results[k], strax.Chunk)
                r0 = results[k]
        else:
            results = r0 = results[self.provides[0]]
            assert isinstance(r0, strax.Chunk)

        return self._fix_output(results, start=r0.start, end=r0.end)

    def cleanup(self, iters, wait_for):
        print(f"{self.__class__.__name__} exhausted. "
              f"Waiting for {len(wait_for)} pending futures.")
        for savers in self.sub_savers.values():
            for s in savers:
                s.close(wait_for=wait_for)
        super().cleanup(iters, wait_for)
