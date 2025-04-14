import numpy as np
import pandas as pd
import uproot 
import awkward as ak
from tqdm import tqdm


data_dir = "/nfs/disk1/users/jierans/eos/eos_laserball_angle_scan"

pmtinfo = pd.read_pickle('pmtinfo_dataframe.pkl')

scanned_zpos = np.arange(-600, 601, 100)
data = []

run_zpos = {
    149: 0,
    150: -100,
    151: -200,
    152: -300,
    153: -400,
    154: -500,
    155: -600,
    156: 100,
    157: 200,
    158: 300,
    159: 400,
    160: 500,
    161: 600,
}

for runnum, zpos in run_zpos.items():
    for output in tqdm(uproot.iterate(
            "/nfs/disk1/eos/eos-processed-data-nubar/rat_processed_data/"
            f"run{runnum}/*.ntuple.root:output",
            [
                "fit_pmtid_Lognormal",
                "fit_time_Lognormal",
             ]), desc=f"Run {runnum}"):
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
data.to_pickle("laserball_dataframe.pkl")

