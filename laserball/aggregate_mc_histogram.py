import numpy as np
from pathlib import Path
import laserball_analysis as ana
import sys
import pickle

data_directory = Path("/nfs/disk1/users/jierans/eos/eos_laserball_sims")
files = list(data_directory.glob("*.ntuple.root"))
wvl_idx = int(sys.argv[1])

sim_ttrees = {}
scanned_wavelengths = np.arange(370, 521, 2)  # [374, 408, 442, 515]
scanned_zpos = np.sort(np.arange(600, -601, -100))
for wvl in scanned_wavelengths:
    sim_ttrees[wvl] = {}
    for zpos in scanned_zpos:
        zpos_txt = f"{zpos}_up" if zpos > 0 else f"{-zpos}_down"
        fname = f"eos_pbomb_{wvl}nm_{zpos_txt}.ntuple.root"
        # print(fname)
        full_path = (data_directory/fname)
        assert full_path in files, f"Sim file not found for {fname}"
        sim_ttrees[wvl][zpos] = full_path.as_posix() + ":output"

pmtinfo = ana.PMTInfo(files[0].as_posix())

def get_tresid(batch, zpos):
    tofs = pmtinfo.time_of_flight(np.asarray([0, 0, zpos]))
    # print(batch['digitPMTID'])
    return batch['fitTime'] - [tofs[ev] for ev in batch['digitPMTID']] - 6.0


def cut(batch, zpos):
    return batch['digitNCrossings'] == 1 & (np.abs(get_tresid(batch, zpos)) < 10)


def get_mchits_by_lcn(mcdata, zpos):
    # hit_histogram = ana.aggregate_histogram(mcdata, "mcPMTID",
    #                                         bin_params={"bins": np.arange(-0.5, 270.5, 1)},
    #                                         flat_transform_func=pmtinfo.id_to_lcn,
    #                                         num_workers=1,
    #                                         step_size="100 MB"
    #                                         )
    hit_histogram = ana.aggregate_histogram(mcdata, "digitPMTID",
                                            expressions = ["digitNCrossings", "digitPMTID", "fitTime"],
                                            bin_params = {"bins": np.arange(-0.5, 270.5, 1)},
                                            flat_transform_func = pmtinfo.id_to_lcn,
                                            cut_func = lambda batch: cut(batch, zpos)
                                            )
    return hit_histogram


bottom_8in = pmtinfo.lcn[(pmtinfo.type == 4)]
bottom_8in_pos = pmtinfo.pos[(pmtinfo.type == 4)]
edges_d8inch = np.arange(np.min(pmtinfo.get_lcns_by_type(1))-0.5, np.max(pmtinfo.get_lcns_by_type(1))+0.6)
bottom_8in_zpos = np.mean(pmtinfo.z[[lcn in bottom_8in for lcn in pmtinfo.lcn]])
bottom_10in_zpos = np.mean(pmtinfo.z[pmtinfo.type == 3])

mc_nhits = {}
mc_norms = {}
for i, zpos in enumerate(scanned_zpos):
    mc_nhits[zpos] = {}
    mc_norms[zpos] = {}
    wvl = scanned_wavelengths[wvl_idx]
    tree = sim_ttrees[wvl][zpos]
    nhits, edges = get_mchits_by_lcn(tree, zpos)
    bottom_8in_ly = np.mean(nhits[bottom_8in])
    norm = bottom_8in_ly
    # mc_norms[zpos][wvl] = norm
    mc_nhits[zpos][wvl] = nhits

with open(f"mc_aggregated_nhit_pickle/mc_aggregated_nhit.{wvl_idx}.pickle", "wb") as f:
    pickle.dump(mc_nhits, f)
