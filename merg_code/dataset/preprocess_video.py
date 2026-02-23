import os
import glob
import torch
import decord
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

VIDEO_PATH = "/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/video_v5_0"
SAVE_PATH = "/mnt/HDD_raid1/AvaMERG_jhchoi/AvaMERG/video_frame"
os.makedirs(SAVE_PATH, exist_ok=True)

NUM_FRAMES = 8
NUM_WORKERS = 6

video_files = glob.glob(os.path.join(VIDEO_PATH, "*.mp4"))


def process_video(video_path):
    try:
        name = os.path.basename(video_path).replace(".mp4", "")
        save_path = os.path.join(SAVE_PATH, name + ".pt")

        if os.path.exists(save_path):
            return

        vr = decord.VideoReader(video_path)
        if len(vr) == 0:
            return

        idxs = np.linspace(0, len(vr) - 1, NUM_FRAMES).astype(int)
        frames = vr.get_batch(idxs)

        frames = torch.from_numpy(frames.asnumpy())
        frames = frames.permute(0, 3, 1, 2).float()

        torch.save(frames, save_path, _use_new_zipfile_serialization=False)

    except Exception as e:
        print("Error:", video_path, e)


if __name__ == "__main__":
    with Pool(NUM_WORKERS) as p:
        list(tqdm(p.imap_unordered(process_video, video_files),
                  total=len(video_files)))