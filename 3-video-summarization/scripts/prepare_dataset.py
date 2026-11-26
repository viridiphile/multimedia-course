"""Prepare the official DSNet assets and validate the selected complete TVSum original."""

from dataclasses import asdict

import numpy as np

from src.dataset import annotation_scores, load_features, probe_video, tvsum_entry
from src.model import CHECKPOINT_NAME, split_role
from src.resources import prepare_dsnet, prepare_tvsum, sha256


VIDEO_ID = "EYqVtI9YWJA"  # Visually distinct motorcycle stunts in a fold-3 test video.


def prepare():
    feature_path, checkpoint = prepare_dsnet(CHECKPOINT_NAME)
    source, metadata_dir = prepare_tvsum(VIDEO_ID)
    key, catalog = tvsum_entry(metadata_dir / "info.tsv", VIDEO_ID)
    record = load_features(feature_path, key)
    video = probe_video(source)
    annotations = annotation_scores(metadata_dir / "anno.tsv", VIDEO_ID)
    if video.frame_count != record.n_frames or annotations.shape[1] != record.n_frames:
        raise ValueError("Decoded original, feature group and official annotation lengths differ")
    average = annotations[:, record.picks].mean(axis=0)
    released = record.gtscore.reshape(-1)
    if average.shape != released.shape or not np.isfinite(released).all():
        raise ValueError("Invalid HDF5 reference annotations")
    correlation = float(np.corrcoef(average, released)[0, 1])
    if not np.isfinite(correlation) or correlation < .98:
        raise ValueError(f"TVSum annotations do not corroborate the feature identity (r={correlation:.5f})")
    source_info = dict(**asdict(video), video_id=VIDEO_ID, title=catalog["title"].strip('"'),
                       category=catalog["category"], source_url=catalog["url"],
                       sha256=sha256(source), size_bytes=source.stat().st_size)
    feature_info = dict(file=feature_path.name, file_size_bytes=feature_path.stat().st_size,
                        sha256=sha256(feature_path), key=key, shape=list(record.features.shape),
                        fields=list(record.fields), n_frames=record.n_frames,
                        hdf5_shots=len(record.shots), first_pick=int(record.picks[0]),
                        last_pick=int(record.picks[-1]),
                        pick_intervals=np.unique(np.diff(record.picks)).tolist(),
                        annotation_shape=list(annotations.shape),
                        sampled_annotation_score_pearson_r=correlation,
                        fold_role=split_role(key))
    return source, video, source_info, record, feature_info, checkpoint


def main():
    _, _, source, _, features, checkpoint = prepare()
    print(f"Verified {source['video_id']} ({source['frame_count']} frames) and {features['key']} "
          f"{features['shape']}; checkpoint {checkpoint.name}; split role {features['fold_role']}.")


if __name__ == "__main__":
    main()
