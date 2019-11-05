import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', default=44100)
    parser.add_argument('--format', default='wav')
    parser.add_argument('--src-dir', default='./')
    parser.add_argument('--dst-dir', default='./')
    args = parser.parse_args()
    return args


def extract_audio(args):
    os.makedirs(os.path.join(args.dst_dir, 'audios/solo/'), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'audios/duet/'), exist_ok=True)
    src_path = os.path.join(args.src_dir, 'videos/solo')
    dst_path = os.path.join(args.dst_dir, 'audios/solo')
    for file in os.listdir(src_path):
        os.system('ffmpeg -i {} -ab 192000 -f {} -ar {} -vn {}'.format(
            os.path.join(src_path, file), args.format, args.freq,
            os.path.join(dst_path, '.'.join(file.split('.')[:-1] + [args.format]))
        ))

    src_path = os.path.join(args.src_dir, 'videos/duet')
    dst_path = os.path.join(args.dst_dir, 'audios/duet')
    for file in os.listdir(src_path):
        os.system('ffmpeg -i {} -ab 192000 -f {} -ar {} -vn {}'.format(
            os.path.join(src_path, file), args.format, args.freq,
            os.path.join(dst_path, '.'.join(file.split('.')[:-1] + [args.format]))
        ))


def main():
    args = parse_args()
    extract_audio(args)


if __name__ == '__main__':
    main()
