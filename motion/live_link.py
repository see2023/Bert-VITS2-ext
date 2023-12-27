import os
import sys
import numpy as np

import threading
import socket
import time
import struct
import datetime

BUFF_SIZE = 1024
class UdpRecvHandlerForArkit:
    def __init__(self, addr_bind, save_path='./records', filename_prefix=None):
        self.addr_bind = addr_bind
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setblocking(1)
        self.udp.bind(self.addr_bind)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # set udp to noblocking
        self.udp.setblocking(0)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if filename_prefix is None:
            # 时间戳 yyyy-mm-dd-hh-mm-ss.npy
            filename_prefix = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = save_path + '/' + filename_prefix + '.npy'
        self.frames = []

    def start(self):
        self.recving = True
        self.thread = threading.Thread(None, target=self.recv_handler)
        self.thread.daemon = True
        self.thread.start()
        print('UdpRecvHandlerForArkit listening on', self.addr_bind)

    def recv_handler(self):
        while self.recving:
            time.sleep(0.01)
            msg = None
            addr = None
            try:
                msg, addr = self.udp.recvfrom(BUFF_SIZE)
            except:
                # get last error
                err = sys.exc_info()[1]
                # check if Errno is 11: [Errno 11] Resource temporarily unavailable
                if err.args[0] != 11 and err.args[0] != 35 and err.args[0] != 10035:
                    print('recv error:', err)
                continue
            # print('recv msg from', addr, 'len', len(msg))
            # unpack msg head like get_livelink_head
            # unpack packer_ver
            packer_ver = np.frombuffer(msg[0:1], dtype='>u1')[0]
            # unpack device_id
            device_id_len = np.frombuffer(msg[1:5], dtype='>i4')[0]
            device_id = msg[5:5 + device_id_len]
            # unpack subject_name
            subject_name_len = np.frombuffer(msg[5 + device_id_len:9 + device_id_len], dtype='>i4')[0]
            subject_name = msg[9 + device_id_len:9 + device_id_len + subject_name_len]
            # print('device_id_len', device_id_len, 'device_id', device_id, 'subject_name_len', subject_name_len, 'subject_name', subject_name)
            # unpack frame_number
            frame_number = np.frombuffer(msg[9 + device_id_len + subject_name_len:13 + device_id_len + subject_name_len], dtype='>u4')[0]
            # unpack sub_frame
            sub_frame = np.frombuffer(msg[13 + device_id_len + subject_name_len:17 + device_id_len + subject_name_len], dtype='>f4')[0]
            # print('sub_frame', sub_frame)
            # unpack numerator
            numerator = np.frombuffer(msg[17 + device_id_len + subject_name_len:21 + device_id_len + subject_name_len], dtype='>u4')[0]
            # unpack denominator
            denominator = np.frombuffer(msg[21 + device_id_len + subject_name_len:25 + device_id_len + subject_name_len], dtype='>u4')[0]
            # print('frame_number', frame_number, 'sub_frame', sub_frame, 'numerator', numerator, 'denominator', denominator)
            # unpack blendshape_count
            if len(msg) <= 25 + device_id_len + subject_name_len:
                # print('msg len error from', addr)
                continue
            blendshape_count = np.frombuffer(msg[25 + device_id_len + subject_name_len:26 + device_id_len + subject_name_len], dtype='>u1')[0]
            if blendshape_count != 61:
                print('blendshape_count error:', blendshape_count, 'from', addr)
                continue
            # unpack blendshape_data
            blendshape_data = np.frombuffer(msg[26 + device_id_len + subject_name_len:], dtype='>f4')
            # check msg len
            if len(msg) != 26 + device_id_len + subject_name_len + 4 * blendshape_count:
                print('msg len error from', addr)
                continue

            self.frames.append(blendshape_data)
            # print blendshape_data every 60 frames
            if frame_number % 60 == 0:
                # print device_id, subject_name, frame_number, sub_frame, numerator, denominator, blendshape_count
                print('recv msg from', addr, ', device_id:', device_id, 'subject_name:', subject_name,
                      'numerator:', numerator, 'denominator:', denominator, 'blendshape_count:', blendshape_count)
                print('blendshape_data:', blendshape_data)

    def save(self):
        # save to file as npy
        np.save(self.filename, np.array(self.frames))
        print('save to', self.filename, 'shape', np.array(self.frames).shape)


    def close(self):
        self.recving = False
        self.udp.close()


class UdpSender:
    def __init__(self, addr_send):
        self.addr_send = addr_send
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setblocking(1)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.common_arkit_head = self.get_livelink_head()

    def get_livelink_head(self):
        packer_ver = np.int8(6)

        # read device_id and subject_name from user env
        device_id = os.environ.get('ARKIT_DEVICE_ID').encode('utf-8')
        subject_name = os.environ.get('ARKIT_SUBJECT_NAME').encode('utf-8')

        frame_number = np.int32(1)
        sub_frame = float(1.000)
        numerator = np.int32(30)
        denominator = np.int32(1)
        blendshape_count = np.int8(61)
        data = bytes()
        data += packer_ver.tobytes()
        data += np.int32(len(device_id)).tobytes()[::-1]
        data += device_id
        data += np.int32(len(subject_name)).tobytes()[::-1]
        data += subject_name
        data += frame_number.tobytes()[::-1]
        data += struct.pack("<f", sub_frame)[::-1]
        data += numerator.tobytes()[::-1]
        data += denominator.tobytes()[::-1]
        data += blendshape_count.tobytes()[::-1]
        return data

    def send(self, data):
        if len(data) < 61:
            data = np.pad(data, (0, 61 - len(data)), "constant")
        dataBuffer = bytes()
        shape_count = len(data)
        if shape_count > 61:
            shape_count = 61
        for i in range(shape_count):
            dataBuffer += struct.pack("<f", data[i])[::-1]
        send_data = self.common_arkit_head + dataBuffer  # + b"\x00\x00\x00\x00"
        for a in self.addr_send:
            self.udp.sendto(send_data, tuple(a))
