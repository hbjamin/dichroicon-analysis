import numpy as np
import pandas as pd
import uproot 
import awkward as ak
from tqdm import tqdm

pmtinfo = pd.read_pickle('pmtinfo_dataframe.pkl')
data_dir = "/nfs/disk1/users/jierans/eos/eos_laserball_sims_border"
scanned_zpos = np.arange(-600, 601, 100)
data = []

for zpos in tqdm(scanned_zpos):
    zpos_sign = "down" if zpos <= 0 else "up"
    rootfile = uproot.open(
        f"{data_dir}/eos_pbomb_514nm_{abs(zpos)}_{zpos_sign}.ntuple.root"
    )
    output = rootfile['output'].arrays()
    fit_times = ak.flatten(output.fit_time_Lognormal).to_numpy()
    hits = ak.flatten(output.fit_pmtid_Lognormal).to_numpy()
    # Prompt Cut
    pmtpositions = pmtinfo.iloc[hits][[ 'pmtx', 'pmty', 'pmtz' ]].to_numpy()
    pmtpositions[:, 2] -= zpos
    distances = np.linalg.norm(pmtpositions, axis=1)
    fit_times -= distances / (300 / 1.34)
    hits_cleaned = hits[np.abs(fit_times) < 10] # In time
    # Aggregate
    ids, counts = np.unique(hits_cleaned, return_counts=1)

    for id, count in zip(ids, counts):
        data.append({
            "id": id,
            "lcn": pmtinfo['lcn'][id],
            "type": pmtinfo['type'][id],
            "zpos": zpos,
            "nhits": count
        })
data = pd.DataFrame(data)
data.to_pickle("laserball_sim_dataframe.pkl")

