import os
import numpy as np
import pyaudio
import io
import wave
import const_map

def play_wav_binary(data):
    # 将二进制数据转换为文件对象
    f = io.BytesIO(data)
    # 用wave模块打开文件对象
    wf = wave.open(f, 'rb')
    # 创建pyaudio对象
    p = pyaudio.PyAudio()
    # 打开音频流
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    # 读取数据并播放
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    # 关闭音频流和pyaudio对象
    stream.stop_stream()
    stream.close()
    p.terminate()


# fps: frames per second
def send_frames(udp_sender, visemes, fps = 86.1328125, delay_ms = 500):
    import time
    # get current time in ms
    start_time = int(time.time() * 1000) + delay_ms
    # for ever loop
    send_count = 0
    while True:
        time_should_send = int(start_time + send_count * 1000 / fps)
        time_now = int(time.time() * 1000)
        if time_now < time_should_send:
            time.sleep((time_should_send - time_now) / 1000)
            continue
        if send_count % 100 == 0:
            print('send frame', send_count)
        udp_sender.send(visemes[send_count])
        if send_count >= len(visemes) - 1:
            print('send last frame')
            udp_sender.send(visemes[send_count])
            break
        send_count += 1

def start_send_frames(udp_sender, visemes, fps):
    # new thread
    import threading
    t = threading.Thread(target=send_frames, args=(udp_sender, visemes, fps))
    t.daemon = True
    t.start()

# 延迟n毫秒，启动新线程，执行 extra.bat
def start_extra(extra_bat, delay_ms):
    import threading
    import time
    # check if extra_bat exists
    if not os.path.exists(extra_bat):
        print('extra_bat not exists:', extra_bat)
        return
    # and it is a executable file
    if not os.access(extra_bat, os.X_OK):
        print('extra_bat not executable:', extra_bat)
        return
    time.sleep(delay_ms / 1000)
    t = threading.Thread(target=os.system, args=(extra_bat,))
    t.daemon = True
    t.start()


def play_and_send(udp_sender, bs_npy_file, wav_file, fps):
    # read bs_value_1114_3_16.npy file
    bs = np.load(bs_npy_file, allow_pickle=True)
    print('bs.shape', bs.shape, 'fps', fps)
    bs_arkit = const_map.map_arkit_values(bs)
    print('remap done')

    # read file_path to binary buffer
    print('reading wav', wav_file)
    with open(wav_file, 'rb') as f:
        data = f.read()
        start_extra('extra.bat', 500)
        start_send_frames(udp_sender, bs_arkit, fps)

        play_wav_binary(data)



import live_link
import sys
if __name__ == '__main__':
    if sys.argv.__len__() < 3:
        print('python tts2ue.py bs_npy_file wav_file fps(86.1328125)')
        exit(1)
    bs_npy_file = sys.argv[1]
    wav_file = sys.argv[2]
    fps = 86.1328125
    if sys.argv.__len__() > 3:
        fps = sys.argv[3]
    # fps to float
    fps = float(fps)
    if not os.path.exists(bs_npy_file):
        print('bs_npy_file not exists')
        exit(1)
    if not os.path.exists(wav_file):
        print('wav_file not exists')
        exit(1)
    print('input npy_file', bs_npy_file, 'wav_file', wav_file)

    udp_sender = live_link.UdpSender([[
				"192.168.17.141",
				11111
			]])
    play_and_send(udp_sender, bs_npy_file, wav_file, fps)

