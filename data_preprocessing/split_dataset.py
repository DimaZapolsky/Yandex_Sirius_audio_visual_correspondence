import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-solo', required=True, type=int)
    parser.add_argument('--dev-duet', required=True, type=int)
    parser.add_argument('--test-solo', required=True, type=int)
    parser.add_argument('--test-duet', required=True, type=int)
    parser.add_argument('--dev-silent', required=True, type=int)
    parser.add_argument('--test-silent', required=True, type=int)
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
    silent_cnt = 0
    for file in os.listdir(os.path.join(video_src_path, 'solo/')):
        if file.endswith('.pt'):
            solo_cnt += 1

    for file in os.listdir(os.path.join(video_src_path, 'duet/')):
        if file.endswith('.pt'):
           duet_cnt += 1

    for file in os.listdir(os.path.join(video_src_path, 'silent/')):
        if file.endswith('.pt'):
            silent_cnt += 1

    print('solo = {} duet = {}'.format(solo_cnt, duet_cnt))

    solo_ind = [i for i in range(solo_cnt)]
    duet_ind = [i for i in range(duet_cnt)]
    silent_ind = [i for i in range(silent_cnt)]
    random.shuffle(solo_ind)
    random.shuffle(duet_ind)
    random.shuffle(silent_ind)

    os.makedirs(os.path.join(video_dst_path, 'train/'), exist_ok=True)
    os.makedirs(os.path.join(video_dst_path, 'dev/'), exist_ok=True)
    os.makedirs(os.path.join(video_dst_path, 'test/'), exist_ok=True)

    os.makedirs(os.path.join(audio_dst_path, 'train/'), exist_ok=True)
    os.makedirs(os.path.join(audio_dst_path, 'dev/'), exist_ok=True)
    os.makedirs(os.path.join(audio_dst_path, 'test/'), exist_ok=True)

    # moving solo videos
    j = 0
    for i in range(args.dev_solo):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'dev/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'dev/', str(i + j) + '.pt')))

    j = -args.dev_solo
    for i in range(args.dev_solo, args.dev_solo + args.test_solo):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'test/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'test/', str(i + j) + '.pt')))

    j = -args.dev_solo - args.test_solo
    for i in range(args.dev_solo + args.test_solo, len(solo_ind)):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'train/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(solo_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'train/', str(i + j) + '.pt')))

    # moving duet videos
    j = args.dev_solo
    for i in range(args.dev_duet):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'dev/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'dev/', str(i + j) + '.pt')))

    j = -args.dev_duet + args.test_solo
    for i in range(args.dev_duet, args.test_duet + args.dev_duet):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'test/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'test/', str(i + j) + '.pt')))

    j = -args.dev_duet - args.test_duet + solo_cnt - args.dev_solo - args.test_solo
    for i in range(args.dev_duet + args.test_duet, len(duet_ind)):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'train/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'duet/', str(duet_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'train/', str(i + j) + '.pt')))

    # moving silent videos
    j = args.dev_solo + args.dev_duet
    for i in range(args.dev_silent):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'silent/', str(silent_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'dev/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'solo/', str(silent_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'dev/', str(i + j) + '.pt')))

    j = -args.dev_silent + args.test_solo + args.test_duet
    for i in range(args.dev_silent, args.dev_silent + args.test_silent):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'silent/', str(silent_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'test/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'silent/', str(silent_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'test/', str(i + j) + '.pt')))

    j = solo_cnt + duet_cnt - args.dev_solo - atgs.test_solo - args.dev_duet - args.test_duet - args.dev_silent - args.test_silent
    for i in range(args.dev_silent + args.test_silent, len(silent_ind)):
        os.system('mv {} {}'.format(os.path.join(video_src_path, 'silent/', str(silent_ind[i]) + '.pt'),
                                    os.path.join(video_dst_path, 'train/', str(i + j) + '.pt')))
        os.system('mv {} {}'.format(os.path.join(audio_src_path, 'silent/', str(silent_ind[i]) + '.pt'),
                                    os.path.join(audio_dst_path, 'train/', str(i + j) + '.pt')))


def main():
    args = parse_args()
    split_dataset(args)


if __name__ == '__main__':
    main()
