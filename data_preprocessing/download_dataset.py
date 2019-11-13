import os
import json


def does_file_exist(filename):
    return os.path.exists(filename)


def main():
    if not does_file_exist('MUSIC_solo_videos.json'):
        os.system('wget https://raw.githubusercontent.com/roudimit/MUSIC_dataset/master/MUSIC_solo_videos.json')

    if not does_file_exist('MUSIC_duet_videos.json'):
        os.system('wget https://raw.githubusercontent.com/roudimit/MUSIC_dataset/master/MUSIC_duet_videos.json')

    os.system('rm -rf videos/')
    os.system('mkdir videos')
    os.system('mkdir videos/solo')
    os.system('mkdir videos/duet')
    os.system('mkdir videos/silent')

    failed = 0
    success = 0

    with open('MUSIC_solo_videos.json') as file:
        cnt = 0
        videos = json.loads(file.read())
        for instrument in videos['videos']:
            for id in videos['videos'][instrument]:
                os.system('youtube-dl --retries=10 "{}" -o videos/solo/{}.{}.mp4 -f 18'.format(id, cnt, instrument))
                if does_file_exist('videos/solo/{}.{}.mp4'.format(cnt, instrument)):
                    cnt += 1
                    success += 1
                else:
                    failed += 1

    with open('MUSIC_duet_videos.json') as file:
        cnt = 0
        videos = json.loads(file.read())
        for instrument in videos['videos']:
            for id in videos['videos'][instrument]:
                instrument = instrument.replace(' ', '.')
                os.system('youtube-dl --retries=10 "{}" -o videos/duet/{}.{}.mp4 -f 18'.format(id, cnt, instrument))
                if does_file_exist('videos/duet/{}.{}.mp4'.format(cnt, instrument)):
                    cnt += 1
                    success += 1
                else:
                    failed += 1

    with open('silent_videos.txt', 'r') as file:
        cnt = 0
        for line in file.readlines():
            os.system('youtube-dl --retries=10 "{}" -o videos/silent/{}.{}.mp4 -f 18'.format(line.strip(), cnt, 'silent'))
            if does_file_exist('videos/silent/{}.{}.mp4'.format(cnt, 'silent')):
                cnt += 1
                success += 1
            else:
                failed += 1

    os.system('rm MUSIC_duet_videos.json')
    os.system('rm MUSIC_solo_videos.json')

    print('{} files successfully downloaded'.format(success))
    print('{} files failed to download'.format(failed))


if __name__ == '__main__':
    main()
