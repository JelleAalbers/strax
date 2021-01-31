import glob
import shutil
import tempfile
import os
import os.path as osp
import pytest

from strax.testutils import *


def test_core():
    for allow_multiprocess in (False, True):
        for max_workers in [1, 2]:
            mystrax = strax.Context(storage=[],
                                    register=[Records, Peaks],
                                    allow_multiprocess=allow_multiprocess)
            bla = mystrax.get_array(run_id=run_id, targets='peaks',
                                    max_workers=max_workers)
            assert len(bla) == recs_per_chunk * n_chunks
            assert bla.dtype == strax.peak_dtype()


def test_multirun():
    for max_workers in [1, 2]:
        mystrax = strax.Context(storage=[],
                                register=[Records, Peaks],)
        bla = mystrax.get_array(run_id=['0', '1'], targets='peaks',
                                max_workers=max_workers)
        n = recs_per_chunk * n_chunks
        assert len(bla) == n * 2
        np.testing.assert_equal(
            bla['run_id'],
            np.array(['0'] * n + ['1'] * n))


def test_filestore():
    with tempfile.TemporaryDirectory() as temp_dir:
        mystrax = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                            deep_scan=True),
                                register=[Records, Peaks])

        assert not mystrax.is_stored(run_id, 'peaks')
        mystrax.scan_runs()
        assert mystrax.list_available('peaks') == []

        # Create it
        peaks_1 = mystrax.get_array(run_id=run_id, targets='peaks')
        assert len(peaks_1) == recs_per_chunk * n_chunks

        assert mystrax.is_stored(run_id, 'peaks')
        mystrax.scan_runs()
        assert mystrax.list_available('peaks') == [run_id]
        assert mystrax.scan_runs()['name'].values.tolist() == [run_id]

        # We should have two directories
        data_dirs = sorted(glob.glob(osp.join(temp_dir, '*/')))
        assert len(data_dirs) == 2

        # The first dir contains peaks.
        # It should have one data chunk (rechunk is on) and a metadata file
        prefix = strax.dirname_to_prefix(data_dirs[0])
        assert sorted(os.listdir(data_dirs[0])) == [
            f'{prefix}-000000',
            f'{prefix}-metadata.json']

        # Check metadata got written correctly.
        metadata = mystrax.get_meta(run_id, 'peaks')
        assert len(metadata)
        assert 'writing_ended' in metadata
        assert 'exception' not in metadata
        assert len(metadata['chunks']) == 1

        # Check data gets loaded from cache, not rebuilt
        md_filename = osp.join(data_dirs[0], f'{prefix}-metadata.json')
        mtime_before = osp.getmtime(md_filename)
        peaks_2 = mystrax.get_array(run_id=run_id, targets='peaks')
        np.testing.assert_array_equal(peaks_1, peaks_2)
        assert mtime_before == osp.getmtime(md_filename)

        # Test the zipfile store. Zipping is still awkward...
        zf = osp.join(temp_dir, f'{run_id}.zip')
        strax.ZipDirectory.zip_dir(temp_dir, zf, delete=True)
        assert osp.exists(zf)

        mystrax = strax.Context(storage=strax.ZipDirectory(temp_dir),
                                register=[Records, Peaks])
        metadata_2 = mystrax.get_meta(run_id, 'peaks')
        assert metadata == metadata_2


def test_multichunk_file():
    """Test FileStore's support of multichunk files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                       deep_scan=True),
                           register=[Records])

        records = st.get_array(run_id, 'records')
        meta = st.get_meta(run_id, 'records')

        peak_dir = st.storage[0].find(st.key_for(run_id, 'records'))[1]
        assert os.path.exists(peak_dir)
        assert st.is_stored(run_id, 'records')
        n_chunks = len(meta['chunks'])
        assert n_chunks > 2, "Can't test properly if you make just one chunk"

        chunks_per_file = 3
        strax.FileSytemBackend().pack_dir(peak_dir,
                                          chunks_per_file=chunks_per_file)

        # Data is still there
        assert os.path.exists(peak_dir), "Data is gone"
        assert st.is_stored(run_id, 'records'), "Strax can't find data"

        # .. and unchanged
        new_recs = st.get_array(run_id, 'records')
        assert len(new_recs) == len(records), "Data mangled"
        assert np.all(new_recs == records), "Data mangled"

        # Chunks are still there
        new_meta = st.get_meta(run_id, 'records')
        print(new_meta)
        assert all(['row_range' in x for x in new_meta['chunks']])
        assert len(meta['chunks']) == n_chunks

        # .. but stored in fewer files
        new_n = len(os.listdir(peak_dir)) - 1   # Skip metadata json
        assert new_n == int(np.ceil(n_chunks / chunks_per_file)), 'Didnt pack'


def test_datadirectory_deleted():
    """Test deleting the data directory does not cause crashes
    or silent failures to save (#93)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = osp.join(temp_dir, 'bla')
        os.makedirs(data_dir)

        mystrax = strax.Context(storage=strax.DataDirectory(data_dir,
                                                            deep_scan=True),
                                register=[Records, Peaks])

        # Delete directory AFTER context is created
        shutil.rmtree(data_dir)

        mystrax.scan_runs()
        assert not mystrax.is_stored(run_id, 'peaks')
        assert mystrax.list_available('peaks') == []

        mystrax.make(run_id=run_id, targets='peaks')

        mystrax.scan_runs()
        assert mystrax.is_stored(run_id, 'peaks')
        assert mystrax.list_available('peaks') == [run_id]


def test_fuzzy_matching():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir,
                                                       deep_scan=True),
                           register=[Records, Peaks])

        st.make(run_id=run_id, targets='peaks')

        # Changing option causes data not to match
        st.set_config(dict(base_area=1))
        assert not st.is_stored(run_id, 'peaks')
        assert st.list_available('peaks') == []

        # In fuzzy context, data does match
        st2 = st.new_context(fuzzy_for=('peaks',))
        assert st2.is_stored(run_id, 'peaks')
        assert st2.list_available('peaks') == [run_id]

        # And we can actually load it
        st2.get_meta(run_id, 'peaks')
        st2.get_array(run_id, 'peaks')

        # Fuzzy for options also works
        st3 = st.new_context(fuzzy_for_options=('base_area',))
        assert st3.is_stored(run_id, 'peaks')

    # No saving occurs at all while fuzzy matching
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks],
                           fuzzy_for=('records',))
        st.make(run_id, 'peaks')
        assert not st.is_stored(run_id, 'peaks')
        assert not st.is_stored(run_id, 'records')


def test_storage_converter():
    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks])
        st.make(run_id=run_id, targets='peaks')

        with tempfile.TemporaryDirectory() as temp_dir_2:
            st = strax.Context(
                storage=[strax.DataDirectory(temp_dir,
                                             readonly=True),
                         strax.DataDirectory(temp_dir_2)],
                register=[Records, Peaks],
                storage_converter=True)
            store_1, store_2 = st.storage

            # Data is now in store 1, but not store 2
            key = st.key_for(run_id, 'peaks')
            store_1.find(key)
            with pytest.raises(strax.DataNotAvailable):
                store_2.find(key)

            st.make(run_id, 'peaks', _skip_if_built=False)

            # Data is now in both stores
            store_1.find(key)
            store_2.find(key)


def test_exception():
    for allow_multiprocess, max_workers in zip((False, True), (1, 2)):
        with tempfile.TemporaryDirectory() as temp_dir:
            st = strax.Context(storage=strax.DataDirectory(temp_dir),
                               register=[Records, Peaks],
                               allow_multiprocess=allow_multiprocess,
                               config=dict(crash=True))

            # Check correct exception is thrown
            with pytest.raises(SomeCrash):
                st.make(run_id=run_id,
                        targets='peaks',
                        max_workers=max_workers)

            # Check exception is recorded in metadata
            # in both its original data type and dependents
            for target in ('peaks', 'records'):
                assert 'SomeCrash' in st.get_meta(run_id, target)['exception']

            # Check corrupted data does not load
            st.context_config['forbid_creation_of'] = ('peaks',)
            with pytest.raises(strax.DataNotAvailable):
                st.get_df(run_id=run_id,
                          targets='peaks',
                          max_workers=max_workers)


def test_exception_in_saver(caplog):
    import logging
    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as temp_dir:
        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks])

        def kaboom(*args, **kwargs):
            raise SomeCrash

        old_save = strax.save_file
        try:
            strax.save_file = kaboom
            with pytest.raises(SomeCrash):
                st.make(run_id=run_id, targets='records')
        finally:
            strax.save_file = old_save


def test_random_access():
    """Test basic random access
    TODO: test random access when time info is not provided directly
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Hack to enable testing if only required chunks are loaded
        Peaks.rechunk_on_save = False

        st = strax.Context(storage=strax.DataDirectory(temp_dir),
                           register=[Records, Peaks, PeakClassification])

        with pytest.raises(strax.DataNotAvailable):
            # Time range selection requires data already available
            st.get_df(run_id, 'peaks', time_range=(3, 5))

        st.make(run_id=run_id, targets=('peaks', 'peak_classification'))

        # Second part of hack: corrupt data by removing one chunk
        dirname = str(st.key_for(run_id, 'peaks'))
        os.remove(os.path.join(temp_dir,
                               dirname,
                               strax.dirname_to_prefix(dirname) + '-000000'))

        with pytest.raises(FileNotFoundError):
            st.get_array(run_id, 'peaks')

        df = st.get_array(run_id, 'peaks', time_range=(3, 5))
        # Also test without the progress-bar
        df_pbar = st.get_array(run_id, 'peaks', time_range=(3, 5), progress_bar = False)
        assert len(df) == 2 * recs_per_chunk
        assert df['time'].min() == 3
        assert df['time'].max() == 4
        assert np.all(df == df_pbar), 'progress-bar changes the result?!?'

        # Try again with unaligned chunks
        df = st.get_array(run_id, ['peaks', 'peak_classification'],
                          time_range=(3, 5))
        assert len(df) == 2 * recs_per_chunk
        assert df['time'].min() == 3
        assert df['time'].max() == 4

    # Remove hack
    Peaks.rechunk_on_save = True


def test_rechunk_on_save():
    """Test saving works with and without rechunk on save

    Doesn't test whether rechunk on save actually rechunks:
    that's done in test_filestore
    """
    for do_rechunk in (False, True):
        Peaks.rechunk_on_save = do_rechunk

        with tempfile.TemporaryDirectory() as temp_dir:
            st = strax.Context(storage=strax.DataDirectory(path=temp_dir),
                               register=[Records, Peaks])

            peaks_0 = st.get_array('0', 'peaks')
            peaks_0a = st.get_array('0', 'peaks')
            np.testing.assert_array_equal(peaks_0, peaks_0a)


def test_run_selection():
    mock_rundb = [
        dict(name='0', mode='funny', tags=[dict(name='bad')]),
        dict(name='1', mode='nice', tags=[dict(name='interesting'),
                                          dict(name='bad')]),
        dict(name='2', mode='nice', tags=[dict(name='interesting')])]

    with tempfile.TemporaryDirectory() as temp_dir:
        sf = strax.DataDirectory(path=temp_dir,
                                 deep_scan=True, provide_run_metadata=True)

        # Write mock runs db
        for d in mock_rundb:
            sf.write_run_metadata(d['name'], d)

        st = strax.Context(storage=sf)
        assert len(st.scan_runs()) == len(mock_rundb)
        assert st.run_metadata('0') == mock_rundb[0]
        assert st.run_metadata('0', projection='name') == {'name': '0'}

        assert len(st.select_runs(run_mode='nice')) == 2
        assert len(st.select_runs(include_tags='interesting')) == 2
        assert len(st.select_runs(include_tags='interesting',
                                  exclude_tags='bad')) == 1
        assert len(st.select_runs(include_tags='interesting',
                                  run_mode='nice')) == 2

        assert len(st.select_runs(run_id='0')) == 1
        assert len(st.select_runs(run_id='*',
                                  exclude_tags='bad')) == 1


def test_run_defaults():
    mock_rundb = [{'name': '0', strax.RUN_DEFAULTS_KEY: dict(base_area=43)}]

    with tempfile.TemporaryDirectory() as temp_dir:
        sf = strax.DataDirectory(path=temp_dir,
                                 deep_scan=True,
                                 provide_run_metadata=True)
        for d in mock_rundb:
            sf.write_run_metadata(d['name'], d)
        st = strax.Context(storage=sf, register=[Records, Peaks],
                           use_per_run_defaults=True)

        # The run defaults get used
        peaks = st.get_array('0', 'peaks')
        assert np.all(peaks['area'] == 43)

        # ... but the user can still override them
        peaks = st.get_array('0', 'peaks', config=dict(base_area=44))
        assert np.all(peaks['area'] == 44)


def test_dtype_mismatch():
    mystrax = strax.Context(storage=[],
                            register=[Records, Peaks],
                            config=dict(give_wrong_dtype=True))
    with pytest.raises(strax.PluginGaveWrongOutput):
        mystrax.get_array(run_id=run_id, targets='peaks')


def test_get_single_plugin():
    mystrax = strax.Context(storage=[],
                            register=[Records, Peaks])
    p = mystrax.get_single_plugin('0', 'peaks')
    assert isinstance(p, Peaks)
    assert len(p.config)
    assert p.config['base_area'] == 0


def test_superrun():
    # TODO: duplicated init with test_run_selection
    mock_rundb = [
        dict(name='0', start=0, end=int(1e9)),
        dict(name='1', start=int(2e9), end=int(3e9))
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test run definition
        sf = strax.DataDirectory(path=temp_dir,
                                 provide_run_metadata=True,
                                 deep_scan=True)
        for d in mock_rundb:
            sf.write_run_metadata(d['name'], d)

        st = strax.Context(storage=sf, register=[Records, Peaks])
        st.define_run('super', ['0', '1'])

        md = st.run_metadata('_super')
        assert md['start'] == 0
        assert md['end'] == int(3e9)
        assert md['livetime'] == int(2e9)
        assert md['sub_run_spec'] == {'0': 'all', '1': 'all'}

        # Test superrun loading
        r1 = st.get_array('0', 'records')
        r2 = st.get_array('1', 'records', config=dict(secret_time_offset=int(1e9)))
        rs = st.get_array('_super', 'records')
        np.testing.assert_array_equal(rs, np.concatenate([r1, r2]))

        # Test that superrun loading triggers subrun processing
        ps = st.get_array('_super', 'peaks')
        p1 = st.get_array('0', 'peaks')
        p2 = st.get_array('1', 'peaks')
        np.testing.assert_array_equal(p1['area'], np.zeros(len(p1)))
        np.testing.assert_array_equal(p2['area'], np.ones(len(p2)))
        np.testing.assert_array_equal(ps, np.concatenate([p1, p2]))

        # Test superrun processing can override run-dependent options
        st.set_config(config=dict(bonus_area=0))
        ps = st.get_array('_super', 'peaks')
        p1 = st.get_array('0', 'peaks')
        p2 = st.get_array('1', 'peaks')
        np.testing.assert_array_equal(p1['area'], np.zeros(len(p1)))
        np.testing.assert_array_equal(p2['area'], np.zeros(len(p2)))
        np.testing.assert_array_equal(ps, np.concatenate([p1, p2]))
