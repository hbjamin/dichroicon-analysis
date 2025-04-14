import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


scan_data = pd.read_pickle("angle_scan_dataframe.pkl")
scan_mean_nhits = scan_data.groupby(['type', 'zpos', 'degree'], as_index=False)['nhits'].agg('mean')

# mask out all channels whose occupancy is less than 10% of the mean occupancy of its type.
scan_channel_online_mask = scan_data.nhits.values > (scan_mean_nhits.nhits[scan_data.type].values*0.1)
scan_data = scan_data[scan_channel_online_mask]
scan_mean_nhits = scan_data.groupby(['type', 'zpos', 'degree'], as_index=False)['nhits'].agg('mean')

for deg, dat_by_deg in scan_mean_nhits.groupby("degree"):
    # if deg >= 0: continue
    zpos_nhit = []
    for zpos, dat in dat_by_deg.groupby("zpos"):
        normalized_dichroicon_nhit = dat[dat.type==1]['nhits'].values[0]
        normalized_dichroicon_nhit /= dat[dat.type==0]['nhits'].values[0]
        zpos_nhit.append((zpos, normalized_dichroicon_nhit))
    zpos_nhit = np.asarray(zpos_nhit).T
    fmtstring = "--" if deg > 0 else 'g-'
    label = f"{deg} deg" if deg >= 0 else "no dichroic filter"
    plt.plot(*zpos_nhit, fmtstring, label=label)

run_data = pd.read_pickle("laserball_dataframe.pkl")
run_mean_nhits = run_data.groupby(['type', 'zpos'], as_index=False)['nhits'].mean()
run_channel_online_mask = run_data.nhits.values > (run_mean_nhits.nhits[run_data.type].values*0.1)
run_data = run_data[run_channel_online_mask]
run_mean_nhits = run_data.groupby(['type', 'zpos'], as_index=False)['nhits'].mean()
zpos_nhit = []
for zpos, dat in run_mean_nhits.groupby("zpos"):
    normalized_dichroicon_nhit = dat[dat.type==1]['nhits'].values[0]
    normalized_dichroicon_nhit /= dat[dat.type==0]['nhits'].values[0]
    zpos_nhit.append((zpos, normalized_dichroicon_nhit))
zpos_nhit = np.asarray(zpos_nhit).T
plt.plot(*zpos_nhit, "k-", label="Eos data")


sim_data = pd.read_pickle("laserball_sim_dataframe.pkl")
sim_mean_nhits = sim_data.groupby(['type', 'zpos'], as_index=False)['nhits'].mean()
sim_channel_online_mask = sim_data.nhits.values > (sim_mean_nhits.nhits[sim_data.type].values*0.1)
sim_data = sim_data[sim_channel_online_mask]
sim_mean_nhits = sim_data.groupby(['type', 'zpos'], as_index=False)['nhits'].mean()
zpos_nhit = []
for zpos, dat in sim_mean_nhits.groupby("zpos"):
    normalized_dichroicon_nhit = dat[dat.type==1]['nhits'].values[0]
    normalized_dichroicon_nhit /= dat[dat.type==0]['nhits'].values[0]
    zpos_nhit.append((zpos, normalized_dichroicon_nhit))
zpos_nhit = np.asarray(zpos_nhit).T
plt.plot(*zpos_nhit, "r-", label="Laserball Sim")

plt.legend()
plt.show()


