import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-solo', required=True, type=int)
    parser.add_argument('--dev-duet', required=True, type=int)
    parser.add_argument('--test-solo', required=True, type=int)
    parser.add_argument('--test-duet', required=True, type=int)
    parser.add_argument('--src-dir', default='./')
    parser.add_argument('--dst-dir', default='./')
    args = parser.parse_args()
    return args


def split_dataset(args):
    video_src_path = os.path.join(args.src_dir, 'videos/')
    video_dst_path = os.path.join(args.dst_dir, 'videos/')
    audio_src_path = os.path.join(args.src_dir, 'audios/')
    audio_dst_path = os.path.join(args.dst_dir, 'audios/')
    solo_cnt = 0
    duet_cnt = 0
    for file in os.listdir(os.path.join(video_src_path, 'solo/')):
        print(file)
        if file.endswith('.pt'):
            solo_cnt += 1

    for file in os.listdir(os.path.join(video_src_path, 'duet/')):
        ifsd file.endswith('.pt'):
            duet_cnt += 1

    print('solo = {} duet = {}'.format(solo_cnt, duet_cnt))

    solo_ind = [i for i in range(solo_cnt)]
    duet_ind = [i for i in range(duet_cnt)]
    random.shuffle(solo_ind)
    random.shuffle(duet_ind)

    os.makedirs(os.path.join(video_dst_path, 'train/'), exist_ok=True)
    os.makedirs(os.path.join(video_dst_path, 'dev/'), exist_ok=True)
    os.makedirs(os.path.join(video_dst_path, 'test/'), exist_ok=True)

    os.makedirs(os.path.join(audio_dst_path, 'train/'), exist_ok=True)
    os.makedirs(os.path.join(audio_dst_path, 'dev/'), exist_ok=True)
    os.makedirs(os.path.join(audio_dst_path, 'test/'), exist_ok=True)

    for i in range(args.dev_solo):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'dev/', str(i) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'dev/', str(i) + '.pt')))

    for i in range(args.dev_solo, args.test_solo):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'test/', str(i - args.dev_solo) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'test/', str(i - args.dev_solo) + '.pt')))

    for i in range(args.dev_solo + args.test_solo, len(solo_ind)):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'train/', str(i - args.dev_solo - args.test_solo) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'train/', str(i - args.dev_solo - args.test_solo) + '.pt')))

    for i in range(args.dev_duet):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'dev/', str(i) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'dev/', str(i) + '.pt')))

    for i in range(args.dev_duet, args.test_duet):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'test/', str(i - args.dev_duet) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'test/', str(i - args.dev_duet) + '.pt')))

    for i in range(args.dev_duet + args.test_duet, len(duet_ind)):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'train/', str(i - args.dev_duet - args.test_duet) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'train/', str(i - args.dev_duet - args.test_duet) + '.pt')))


def main():
    args = parse_args()
    split_dataset(args)


if __name__ == '__main__':
    main()
