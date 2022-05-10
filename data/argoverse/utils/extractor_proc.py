
import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path

# If the machine has enough RAM one might try if this speeds up loading
@lru_cache(300000)
def _read_csv(path):
    return pd.read_csv(path)


class ArgoDataExtractor:
    def __init__(self, args):
        self.align_image_with_target_x = args.align_image_with_target_x

    def get_displ(self, data):
        """
        Get x and y displacements (proportional to discrete velocities) for
        a given trajectory and update the valid flag for observed timesteps

        Args:
            data: Trajectories of all agents

        Returns:
            Displacements of all agents
        """
        res = np.zeros((data.shape[0], data.shape[1] - 1, data.shape[2]))

        for i in range(len(res)):
            # Replace  0 in first dimension with 2
            diff = data[i, 1:, :2] - data[i, :-1, :2]

            # Sliding window (size=2) with the valid flag
            valid = np.convolve(data[i, :, 2], np.ones(2), "valid")
            # Valid entries have the sum=2 (they get flag=1=valid), unvalid entries have the sum=1 or sum=2 (they get flag=0)
            valid = np.select(
                [valid == 2, valid == 1, valid == 0], [1, 0, 0], valid)

            res[i, :, :2] = diff
            res[i, :, 2] = valid

            # Set zeroes everywhere, where third dimension is = 0 (invalid)
            res[i, res[i, :, 2] == 0] = 0

        return np.float32(res), data[:, -1, :2]

    def extract_data(self, filename):
        """Load csv and extract the features required for CRAT-Pred

        Args:
            filename: Filename of the csv to load

        Returns:
            Feature dictionary required for CRAT-Pred
        """
        #df = _read_csv(filename)
        df = pd.read_csv(filename)
        argo_id = int(Path(filename).stem)

        city = df["CITY_NAME"].values[0]

        agt_ts = np.sort(np.unique(df["TIMESTAMP"].values))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)

        steps = [mapping[x] for x in df["TIMESTAMP"].values]
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(["TRACK_ID", "OBJECT_TYPE"]).groups
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]

        agnt_key = keys.pop(obj_type.index("AGENT"))
        av_key = keys.pop(obj_type.index("AV"))
        keys = [agnt_key, av_key] + keys

        res_trajs = []
        for key in keys:
            idcs = objs[key]
            tt = trajs[idcs]
            ts = steps[idcs]
            rt = np.zeros((50, 3))

            if 19 not in ts:
                continue

            rt[ts, :2] = tt
            rt[ts, 2] = 1.0
            res_trajs.append(rt)

        res_trajs = np.asarray(res_trajs, np.float32)
        res_gt = res_trajs[:, 20:].copy()

        origin = res_trajs[0, 19, :2].copy()

        rotation = np.eye(2, dtype=np.float32)
        theta = 0
        if self.align_image_with_target_x:
            pre = res_trajs[0, 19, :2] - res_trajs[0, 18, :2]
            theta = np.arctan2(pre[1], pre[0])
            rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]], np.float32)

        res_trajs[:, :, :2] = np.dot(res_trajs[:, :, :2] - origin, rotation)
        res_trajs[np.where(res_trajs[:, :, 2] == 0)] = 0

        res_fut_trajs = res_trajs[:, 20:].copy()
        res_trajs = res_trajs[:, :20].copy()

        sample = dict()
        sample["argo_id"] = argo_id
        sample["city"] = city
        sample["past_trajs"] = res_trajs
        sample["fut_trajs"] = res_fut_trajs

        sample["gt"] = res_gt[:, :, :2]

        sample["displ"], sample["centers"] = self.get_displ(sample["past_trajs"])
        sample["origin"] = origin
        # We already return the inverse transformation matrix
        sample["rotation"] = np.linalg.inv(rotation)

        return sample
