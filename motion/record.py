# 录音的同时开启udp监听，获取arkit发送过来的数据，然后分别保存到文件里
import wave
import threading
import pyaudio
import datetime
import live_link

class AudioRecorder():
    '''
    用pyaudio录音，并保存到 wav 文件
    '''
    def __init__(self, filename=None, save_path= './records', channels=1, rate=44100, chunk=1024):
        if filename is None:
            # 时间戳 yyyy-mm-dd-hh-mm-ss.wav
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.wav'
        else:
            filename += '.wav'
        self.isrecording = False
        self.save_path = save_path
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
    
    def close(self):
        # self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def save(self):
        wf = wave.open(self.save_path + '/' + self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
    
    def __recording(self):
        while self.isrecording:
            self.frames.append(self.stream.read(self.chunk))

    def start(self):
        print('Start recording')
        self.frames = []
        self.isrecording = True
        self.t = threading.Thread(target=self.__recording)
        self.t.daemon = True
        self.t.start()


if __name__ == '__main__':
    filename_prefix = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    recorder = AudioRecorder(filename=filename_prefix)
    # arkit 127.0.0.1 11111
    arkit_recorder = live_link.UdpRecvHandlerForArkit(
        (
				"0.0.0.0",
				11111
        ),
        filename_prefix=filename_prefix
    )
    arkit_recorder.start()
    recorder.start()
    print('Recording audio...')
    # wait 10 seconds
    import time
    time.sleep(20)
    recorder.save()
    arkit_recorder.save()
    print('End audio recording')
    recorder.close()
    arkit_recorder.close()
    print('End arkit recording')
