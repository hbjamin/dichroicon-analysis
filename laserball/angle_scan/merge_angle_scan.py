import numpy as np
import pandas as pd
import uproot 
import awkward as ak
from tqdm import tqdm


data_dir = "/nfs/disk1/users/jierans/eos/eos_laserball_angle_scan"

# First open up a meta branch to get the PMT information
pmtinfo = pd.DataFrame()
with uproot.open(f'{data_dir}/pbomb_0_deg_0_down.ntuple.root') as f:
    meta: ak.Record = f['meta'].arrays()
    pmtinfo['pmtid'] = ak.flatten(meta.pmtId).to_numpy()
    pmtinfo.set_index('pmtid', inplace=True)
    pmtinfo['lcn'] = ak.flatten(meta.pmtChannel).to_numpy()
    pmtinfo['type'] = ak.flatten(meta.pmtType).to_numpy()
    pmtinfo['pmtx'] = ak.flatten(meta.pmtX).to_numpy()
    pmtinfo['pmty'] = ak.flatten(meta.pmtY).to_numpy()
    pmtinfo['pmtz'] = ak.flatten(meta.pmtZ).to_numpy()
    pmtinfo['pmtu'] = ak.flatten(meta.pmtU).to_numpy()
    pmtinfo['pmtv'] = ak.flatten(meta.pmtV).to_numpy()
    pmtinfo['pmtw'] = ak.flatten(meta.pmtW).to_numpy()

degrees = [-1, *np.arange(0, 91, 5)] # -1 is baseline
scanned_zpos = np.arange(-600, 601, 100)
data = []
for deg in tqdm(degrees, desc="Angle scan"):
    for zpos in tqdm(scanned_zpos, leave=bool(deg==degrees[-1]), desc="Z positions"):
        zpos_sign = "down" if zpos <= 0 else "up"
        if deg == -1:
            rootfile: uproot.ReadOnlyDirectory = uproot.open(
                f"{data_dir}/pbomb_baseline_{abs(zpos)}_{zpos_sign}.ntuple.root"
            )
        else:
            rootfile: uproot.ReadOnlyDirectory = uproot.open(
                f"{data_dir}/pbomb_{deg}_deg_{abs(zpos)}_{zpos_sign}.ntuple.root"
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
                "degree": deg,
                "nhits": count
            })
data = pd.DataFrame(data)
pmtinfo.to_pickle("pmtinfo_dataframe.pkl")
data.to_pickle("angle_scan_dataframe.pkl")

