import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm
from pathlib import Path


class PMTInfo:
    def __init__(self, ntuple_fname):
        meta = uproot.open(ntuple_fname)['meta']
        pmtx = meta['pmtX'].array(library='numpy')[0]
        pmty = meta['pmtY'].array(library='numpy')[0]
        pmtz = meta['pmtZ'].array(library='numpy')[0]
        pmtu = meta['pmtU'].array(library='numpy')[0]
        pmtv = meta['pmtV'].array(library='numpy')[0]
        pmtw = meta['pmtW'].array(library='numpy')[0]
        pmtid = meta['pmtId'].array(library='numpy')[0]
        pmttype = meta['pmtType'].array(library='numpy')[0]
        pmt_lcn = meta['pmtChannel'].array(library='numpy')[0]

        pmt_pos = np.stack([pmtx, pmty, pmtz], axis=1)
        pmt_dir = np.stack([pmtu, pmtv, pmtw], axis=1)

        self.pos = pmt_pos
        self.dir = pmt_dir
        self.id = pmtid
        self.type = pmttype
        self.lcn = pmt_lcn
        self.x, self.y, self.z = pmtx, pmty, pmtz
        self.u, self.v, self.w = pmtu, pmtv, pmtw

    def time_of_flight(self, source_pos, rindex=1.34):
        tof = np.linalg.norm(self.pos - source_pos, axis=1) / 300 * rindex
        return tof

    def id_to_lcn(self, pmtid):
        return self.lcn[pmtid]

    def lcn_to_id(self, lcns):
        result = []
        for lcn in lcns:
            result.append(np.argwhere(self.lcn == lcn)[0][0])
        return np.asarray(result)
            

    def get_lcns_by_type(self, pmttype):
        return self.lcn[self.type == pmttype]


def get_flat_arrays(ntuple_fname, pmtinfo):
    events = uproot.open(ntuple_fname)['output'].arrays()
    data = {}
    for key, value in events.items():
        data[key] = ak.flatten(value).to_numpy()
    if 'digitid' in data:
        data['lcn'] = pmtinfo['lcn'][data['digitid']]
    return data


def aggregate_histogram(ttree, branch_name,
                        expressions=None,
                        bin_params=None,
                        flat_transform_func=None,
                        cut_func=None,
                        step_size='100 MB',
                        num_workers=1
                       ):
    '''
    expressions: list of branch names that needs to be included in the batch. Default is all branches
    bin_params: parameters fed to np.histogram
    transform_func: transform array before binning
    cut_func: f(batch)->True if include in histogram
    '''
    if bin_params is None:
        bin_params = {}
    result = None
    if isinstance(ttree, str):
        fglob, treename = ttree.split(':')
        flist = Path('/').glob(fglob[1:])
        treelist = [f'{fname}:{treename}' for fname in flist]
        nevts = np.sum([n for (_, _, n) in uproot.num_entries(treelist)])
        pbar = tqdm(total=nevts, unit="event", desc=ttree)
        iterator = uproot.iterate(ttree, step_size=step_size, num_workers=num_workers, expressions=expressions)
    else:
        pbar = tqdm(total=ttree.num_entries, unit="event")
        iterator = ttree.iterate(expressions, step_size=step_size)
    for batch in iterator:
        pbar.update(len(batch))
        if callable(branch_name):
            target_array = branch_name(batch)
        elif isinstance(branch_name, str):
            target_array = batch[branch_name]
        if cut_func is not None:
            cut = cut_func(batch)
            target_array = target_array[cut]
        target_array_flat = ak.flatten(target_array).to_numpy()
        if flat_transform_func is not None:
            target_array_flat = flat_transform_func(target_array_flat)
        hist = np.histogram(target_array_flat, **bin_params)
        if result is None:
            result = hist
        else:
            assert np.all(result[1] == hist[1])
            result = result[0] + hist[0], result[1]
    return result


def get_hits_by_lcn(hit_digitids, pmtinfo):
    digitid, counts = np.unique(hit_digitids, return_counts=True)
    digitlcn = pmtinfo['lcn'][digitid]
    p = np.argsort(digitlcn)
    return (digitlcn[p], counts[p])
