"""
    Script for extracting audio (16-bit, mono, 22000 Hz) from video file.
"""

import os
import argparse
import subprocess


def parse_args():
    """
    Create python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Extract audio from video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--in-video",
        type=str,
        required=True,
        help="path to input video file or directory")
    parser.add_argument(
        "--out-audio",
        type=str,
        help="path to output audio file")

    args = parser.parse_args()
    return args


def extract_audio(in_video,
                  out_audio):
    """
    Real extract audio from video file.

    Parameters
    ----------
    in_video : str
        Path to input video file.
    out_audio : str
        Path to output audio file.
    """
    if not out_audio:
        file_stem, _ = os.path.splitext(in_video)
        out_audio = file_stem + ".wav"
    # command1 = "ffmpeg -i {in_video} -vn -acodec copy {aac_audio}"
    # command2 = "ffmpeg -i {aac_audio} -vn -acodec pcm_s16le -ac 1 -ar 22000 {out_audio}"
    # command = "ffmpeg -i {in_video} -vn -acodec pcm_s16le -ac 1 -ar 22000 {out_audio}"
    command = "ffmpeg -i {in_video} -vn -acodec pcm_s16le -ac 1 -ar 16000 {out_audio}"
    subprocess.call([command.format(in_video=in_video, out_audio=out_audio)], shell=True)


def main():
    """
    Main body of script.
    """
    args = parse_args()
    in_video = os.path.expanduser(args.in_video)
    if not os.path.exists(in_video):
        raise Exception("Input file/directory doesn't exist: {}".format(in_video))
    if os.path.isfile(in_video):
        extract_audio(
            in_video=in_video,
            out_audio=args.out_audio)
    else:
        video_file_paths = []
        for file_name in os.listdir(in_video):
            if not os.path.isfile(os.path.join(in_video, file_name)):
                continue
            _, file_ext = os.path.splitext(file_name)
            if file_ext.lower() in (".mp4", ".mkv", ".avi"):
                video_file_path = os.path.join(in_video, file_name)
                video_file_paths.append(video_file_path)
        video_file_paths = sorted(video_file_paths)
        for video_file_path in video_file_paths:
            extract_audio(
                in_video=video_file_path,
                out_audio="")


if __name__ == "__main__":
    main()
