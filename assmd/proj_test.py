import numpy as np
import pytraj as pt

def projectTrajectory(traj:pt.Trajectory):
    feats = []
    #here extract features using pytraj
    ca_dist = []
    for resnum in range(1,294):
        dist = pt.distance(traj=traj, mask=f":BR :{resnum}@CA")
        ca_dist.append(dist)
    #assing features to feats variable in way that feats is a list of lists (list for each feature)
    feats = ca_dist
    combined = np.column_stack(feats)
    #np.array is a expected return rows:frames, columns:features
    return combined