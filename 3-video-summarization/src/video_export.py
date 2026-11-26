"""Extract source-frame ranges without changing source playback rate."""

import subprocess

import imageio_ffmpeg

from .segments import verify_intervals


def export(source, metadata, intervals, destination):
    expected = verify_intervals(intervals, metadata.frame_count)
    fps = metadata.fps
    expression = "+".join(f"between(n\\,{start}\\,{stop - 1})" for start, stop in intervals)
    filters = f"select={expression},settb=expr={fps.denominator}/{fps.numerator},setpts=N"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-v", "error", "-xerror", "-nostdin", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters,
        "-fps_mode", "cfr", "-r", str(fps),
        "-enc_time_base", f"{fps.denominator}/{fps.numerator}",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-crf", "16", "-movflags", "+faststart", str(destination),
    ], check=True)
    return expected
