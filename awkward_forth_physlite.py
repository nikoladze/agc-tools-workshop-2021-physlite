"""
extracted from https://gitlab.cern.ch/nihartma/physlite-experiments/-/blob/master/physlite_experiments/deserialization_hacks.py
"""

import queue
import numpy as np
import uproot
from uproot import AsObjects, AsVector, AsString
import awkward as ak
import awkward.forth


_forth_machine_cache = {}

def _generate_forth_machine(*args):
    data_size, data_header_size, num_entries_size, ndim = args
    if args in _forth_machine_cache:
        return _forth_machine_cache[args]

    forth = [
        "input data",
        "input byte_offsets",
    ]
    for i in range(ndim):
        forth.append(f"output offsets{i} int64")
    forth.append("output content int8")
    for i in range(ndim):
        forth.append(f"0 offsets{i} <- stack")
    forth += [
        "begin",
        "  byte_offsets i-> stack",
        "  6 + data seek",
        "  data !i-> stack"
        "  dup offsets0 +<- stack",
    ]
    for i in range(1, ndim):
        forth.append("0 do")
        if num_entries_size == 4:
            forth.append("data !i-> stack")
        elif num_entries_size == 1:
            forth.append("data !b-> stack")
        else:
            raise NotImplementedError(
                f"No implementation for `num_entries_size` == {num_entries_size}"
            )
        forth.append(f"dup offsets{i} +<- stack")
    forth += [
        f"{data_size + data_header_size} *",
        "data #!b-> content"
    ]
    for i in range(ndim - 1):
        forth.append("loop")
    forth.append("again")
    forth = "\n".join(forth)
    machine = awkward.forth.ForthMachine32(forth)
    _forth_machine_cache[args] = machine
    return machine


def _read_nested_vector_forth(
    basket_data,
    num_entries,
    byte_offsets,
    data_size=4,
    data_header_size=0,
    num_entries_size=4,
    ndim=2,
):
    machine = _generate_forth_machine(data_size, data_header_size, num_entries_size, ndim)
    machine.run(
        {"data": basket_data, "byte_offsets": byte_offsets},
        raise_read_beyond=False,
        raise_seek_beyond=False,
    )
    content = machine.output_NumpyArray("content")
    if data_header_size != 0:
        content = np.asarray(content)
        content = content.view(
            [("header", f">V{data_header_size}"), ("data", f">V{data_size}")]
        )["data"]
    return [
        np.asarray(i) for i in [
            machine.output_Index64(f"offsets{j}")
            for j in range(ndim)
        ] + [content]
    ]


def _read_vector_vector(basket_data, num_entries, **kwargs):
    return _read_nested_vector_forth(np.array(basket_data), num_entries, ndim=2, **kwargs)


def _read_vector_vector_vector(basket_data, num_entries, **kwargs):
    return _read_nested_vector_forth(np.array(basket_data), num_entries, ndim=3, **kwargs)


def _get_baskets(branch, entry_start=None, entry_stop=None):
    notifications = queue.Queue()
    source = branch._file._source

    basket_chunks = []
    basket_ids = {}
    entry_starts, entry_stops = (
        branch.member("fBasketEntry")[:-1],
        branch.member("fBasketEntry")[1:],
    )
    basket_entries = branch.member("fBasketEntry")
    for i in range(branch.num_baskets):

        if entry_start is not None and entry_stops[i] <= entry_start:
            continue
        if entry_stop is not None and entry_starts[i] >= entry_stop:
            break

        start = branch.member("fBasketSeek")[i]
        stop = start + branch.basket_compressed_bytes(i)
        basket_chunks.append((int(start), int(stop)))
        basket_ids[start, stop] = i

    def chunk_to_basket(chunk, basket_num):
        cursor = uproot.source.cursor.Cursor(chunk.start)
        return uproot.models.TBasket.Model_TBasket.read(
            chunk,
            cursor,
            {"basket_num": basket_num},
            branch._file,
            branch._file,
            branch,
        )

    source.chunks(basket_chunks, notifications)
    result_baskets = {}
    for i in range(len(basket_chunks)):
        chunk = notifications.get(timeout=10)
        basket_num = basket_ids[chunk.start, chunk.stop]
        result_baskets[basket_num] = chunk_to_basket(chunk, basket_num)

    return result_baskets


def _get_start_stop(first_basket_start, num_entries, entry_start, entry_stop):
    stop = entry_stop or num_entries
    start = entry_start or 0
    num_entries = stop - start
    this_entry_start = start - first_basket_start
    this_entry_stop = this_entry_start + num_entries
    return this_entry_start, this_entry_stop


def _branch_to_array_vector_vector(
    branch,
    dtype=np.dtype(">i4"),
    data_size=4,
    data_header_size=0,
    num_entries_size=4,
    entry_start=None,
    entry_stop=None,
):
    offsets_lvl1, offsets_lvl2, data = [], [], []
    baskets = _get_baskets(branch, entry_start=entry_start, entry_stop=entry_stop)
    for i in sorted(baskets):
        basket = baskets[i]
        offsets_lvl1_i, offsets_lvl2_i, data_i = _read_vector_vector(
            basket.data,
            basket.num_entries,
            byte_offsets=basket.byte_offsets,
            data_size=data_size,
            data_header_size=data_header_size,
            num_entries_size=num_entries_size,
        )
        data.append(data_i)
        if len(offsets_lvl1) == 0:
            offsets_lvl1.append(offsets_lvl1_i)
            offsets_lvl2.append(offsets_lvl2_i)
        else:
            # add last offset from previous basket
            if len(offsets_lvl1_i) > 1:
                offsets_lvl1.append(offsets_lvl1_i[1:] + offsets_lvl1[-1][-1])
            if len(offsets_lvl2_i) > 1:
                offsets_lvl2.append(offsets_lvl2_i[1:] + offsets_lvl2[-1][-1])
    offsets_lvl1, offsets_lvl2, data = [
        np.concatenate(i) for i in [offsets_lvl1, offsets_lvl2, data]
    ]
    data = np.frombuffer(data.tobytes(), dtype=dtype)
    # storing in parquet needs contiguous arrays
    if data.dtype.fields is None:
        data = ak.Array(data.newbyteorder().byteswap()).layout
    else:
        data = ak.zip(
            {
                k: np.ascontiguousarray(data[k]).newbyteorder().byteswap()
                for k in data.dtype.fields
            }
        ).layout
    if entry_start is not None or entry_stop is not None:
        start, stop = _get_start_stop(
            baskets[min(baskets)].entry_start_stop[0],
            branch.num_entries,
            entry_start,
            entry_stop,
        )
        offsets_lvl1 = offsets_lvl1[start: stop + 1]
        offsets_lvl2 = offsets_lvl2[offsets_lvl1[0]: offsets_lvl1[-1] + 1]
        data = data[offsets_lvl2[0]: offsets_lvl2[-1]]
        offsets_lvl1 -= offsets_lvl1[0]
        offsets_lvl2 -= offsets_lvl2[0]
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(offsets_lvl1),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(offsets_lvl2),
                data,
            ),
        )
    )
    return array


def _branch_to_array_vector_vector_vector(
    branch, dtype=np.dtype(">i4"), data_size=4, data_header_size=0, num_entries_size=4
):
    offsets_lvl1, offsets_lvl2, offsets_lvl3, data = [], [], [], []
    baskets = _get_baskets(branch)
    for i in range(branch.num_baskets):
        basket = baskets[i]
        (
            offsets_lvl1_i,
            offsets_lvl2_i,
            offsets_lvl3_i,
            data_i,
        ) = _read_vector_vector_vector(
            basket.data,
            basket.num_entries,
            data_size=data_size,
            data_header_size=data_header_size,
            num_entries_size=num_entries_size,
            byte_offsets=basket.byte_offsets,
        )
        data.append(data_i)
        if len(offsets_lvl1) == 0:
            offsets_lvl1.append(offsets_lvl1_i)
            offsets_lvl2.append(offsets_lvl2_i)
            offsets_lvl3.append(offsets_lvl3_i)
        else:
            # add last offset from previous basket
            if len(offsets_lvl1_i) > 1:
                offsets_lvl1.append(offsets_lvl1_i[1:] + offsets_lvl1[-1][-1])
            if len(offsets_lvl2_i) > 1:
                offsets_lvl2.append(offsets_lvl2_i[1:] + offsets_lvl2[-1][-1])
            if len(offsets_lvl3_i) > 1:
                offsets_lvl3.append(offsets_lvl3_i[1:] + offsets_lvl3[-1][-1])
    offsets_lvl1, offsets_lvl2, offsets_lvl3, data = [
        np.concatenate(i) for i in [offsets_lvl1, offsets_lvl2, offsets_lvl3, data]
    ]
    data = np.frombuffer(data.tobytes(), dtype=dtype)
    # storing in parquet needs contiguous arrays
    if data.dtype.fields is None:
        data = ak.Array(data.newbyteorder().byteswap()).layout
    else:
        data = ak.zip(
            {
                k: np.ascontiguousarray(data[k]).newbyteorder().byteswap()
                for k in data.dtype.fields
            }
        ).layout
    return ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(offsets_lvl1),
            ak.layout.ListOffsetArray64(
                ak.layout.Index64(offsets_lvl2),
                ak.layout.ListOffsetArray64(
                    ak.layout.Index64(offsets_lvl3),
                    data,
                ),
            ),
        ),
    )


def _branch_to_array_vector_vector_elementlink(branch, **kwargs):
    return _branch_to_array_vector_vector(
        branch,
        dtype=np.dtype([("m_persKey", ">i4"), ("m_persIndex", ">i4")]),
        data_size=8,
        data_header_size=20,
        **kwargs,
    )


def _branch_to_array_vector_string(branch, **kwargs):
    array = _branch_to_array_vector_vector(
        branch, dtype=np.uint8, data_size=1, num_entries_size=1, **kwargs
    )
    array.layout.content.setparameter("__array__", "string")
    array.layout.content.content.setparameter("__array__", "char")
    return array


def interpretation_is_vector_vector(interpretation):
    "... there is probably a better way"
    if not isinstance(interpretation, AsObjects):
        return False
    if not hasattr(interpretation, "_model"):
        return False
    if not isinstance(interpretation._model, AsVector):
        return False
    if not interpretation._model.header:
        return False
    if not isinstance(interpretation._model.values, AsVector):
        return False
    if interpretation._model.values.header:
        return False
    if isinstance(interpretation._model.values.values, AsVector):
        # vector<vector<vector
        return False
    return True


_other_custom = {
    "AsObjects(AsVector(True, AsVector(False, AsVector(False, dtype('>u8')))))": (
        lambda branch, **kwargs: _branch_to_array_vector_vector_vector(
            branch, dtype=np.dtype(">u8"), data_size=8, **kwargs
        )
    ),
    "AsObjects(AsVector(True, AsVector(False, AsVector(False, dtype('uint8')))))": (
        lambda branch, **kwargs: _branch_to_array_vector_vector_vector(
            branch, dtype=np.dtype(">i1"), data_size=1, **kwargs
        )
    ),
    "AsObjects(AsVector(True, AsSet(False, dtype('>u4'))))": (
        lambda branch, **kwargs: _branch_to_array_vector_vector(
            branch, dtype=np.dtype(">u4"), data_size=4, **kwargs
        )
    ),
}


def branch_to_array(branch, force_custom=False, **kwargs):
    "Try to deserialize with the custom functions and fall back to uproot"
    if branch.interpretation == AsObjects(AsVector(True, AsString(False))):
        return _branch_to_array_vector_string(branch, **kwargs)
    elif interpretation_is_vector_vector(branch.interpretation):
        values = branch.interpretation._model.values.values
        if isinstance(values, np.dtype):
            return _branch_to_array_vector_vector(
                branch,
                dtype=values,
                data_size=values.itemsize,
                data_header_size=0,
                **kwargs,
            )
        else:
            if "ElementLink_3c_DataVector" in values.__name__:
                return _branch_to_array_vector_vector_elementlink(branch, **kwargs)
    elif str(branch.interpretation) in _other_custom:
        return _other_custom[str(branch.interpretation)](branch, **kwargs)
    if force_custom:
        raise TypeError(
            f"No custom deserialization for interpretation {branch.interpretation}"
        )
    return branch.array(**kwargs)


def _extract_base_form_no_fix(cls, tree, iteritems_options={}):
    """
    patched version for UprootSourceMapping._extract_base_form to skip fixing object branches

    needed to experiment with AwkwardForth before
    https://github.com/CoffeaTeam/coffea/pull/609
    """

    import json
    import warnings
    from coffea.nanoevents.mapping.uproot import CannotBeNanoEvents, _lazify_form

    branch_forms = {}
    for key, branch in tree.iteritems(**iteritems_options):
        if key in branch_forms:
            warnings.warn(
                f"Found duplicate branch {key} in {tree}, taking first instance"
            )
            continue
        if "," in key or "!" in key:
            warnings.warn(
                f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
            )
            continue
        if len(branch):
            # The branch is split and its sub-branches will be enumerated by tree.iteritems
            continue
        if isinstance(
            branch.interpretation,
            uproot.interpretation.identify.UnknownInterpretation,
        ):
            warnings.warn(f"Skipping {key} as it is not interpretable by Uproot")
            continue
        try:
            form = branch.interpretation.awkward_form(None)
        except uproot.interpretation.objects.CannotBeAwkward:
            warnings.warn(
                f"Skipping {key} as it is it cannot be represented as an Awkward array"
            )
            continue
        form = uproot._util.awkward_form_remove_uproot(awkward, form)
        form = json.loads(
            form.tojson()
        )  # normalizes form (expand NumpyArray classes)
        try:
            form = _lazify_form(form, f"{key},!load", docstr=branch.title)
        except CannotBeNanoEvents as ex:
            warnings.warn(
                f"Skipping {key} as it is not interpretable by NanoEvents\nDetails: {ex}"
            )
            continue
        branch_forms[key] = form

    return {
        "class": "RecordArray",
        "contents": branch_forms,
        "parameters": {"__doc__": tree.title},
        "form_key": "",
    }


def patch_nanoevents(verbose=False):
    """
    Patch the `extract_column` method of `UprootSourceMapping` in
    `coffea.nanoevents` to make use of the deserialization hacks
    """
    from coffea.nanoevents.mapping import UprootSourceMapping
    from coffea.nanoevents.schemas import PHYSLITESchema

    def extract_column(self, columnhandle, start, stop):
        if verbose:
            print("extracting", columnhandle)
        return branch_to_array(columnhandle, entry_start=start, entry_stop=stop)

    UprootSourceMapping.extract_column = extract_column
    if hasattr(PHYSLITESchema, "_hack_for_elementlink_int64"):
        PHYSLITESchema._hack_for_elementlink_int64 = False
    if hasattr(UprootSourceMapping, "_fix_awkward_form_of_iter"):
        UprootSourceMapping._fix_awkward_form_of_iter = False
    if (
            hasattr(uproot._util, "recursively_fix_awkward_form_of_iter")
            and not hasattr(UprootSourceMapping, "_fix_awkward_form_of_iter")
    ):
        # https://github.com/CoffeaTeam/coffea/pull/609 not yet applied
        UprootSourceMapping._extract_base_form = _extract_base_form_no_fix
