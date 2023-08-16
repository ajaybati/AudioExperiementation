from nemo.collections.asr.models import EncDecSpeakerLabelModel
from IPython.display import Audio, display
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import math
import os
import requests
import boto3
import wave
import sys
import contextlib
import webrtcvad
import collections
from tqdm import tqdm
# from tqdm.notebook import tqdm

def play_audio(waveform, sample_rate, channel=-1):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        if channel != -1:
            display(Audio((waveform[channel]), rate=sample_rate))
        else:
            display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")

speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name="speakerverification_speakernet")

speaker_encoder.eval()

def get_emb(wav_path):
    return speaker_encoder.get_embedding(wav_path)

client = boto3.client(
    's3',
    aws_access_key_id = 'ENTER ACCESS KEY',
    aws_secret_access_key = 'ENTER SECRET ACCESS KEY',
    region_name = 'REGION NAME'
)
    
# Creating the high level object oriented interface
resource = boto3.resource(
    's3',
    aws_access_key_id = 'ENTER ACCESS KEY',
    aws_secret_access_key = 'ENTER SECRET ACCESS KEY',
    region_name = 'REGION NAME'
)

bucket_name = "bucket name"
bucket_name

resp = client.list_objects(Bucket=bucket_name, Prefix='PREFIX', Delimiter="/")
speakers = [r['Prefix'] for r in resp["CommonPrefixes"]]

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2) #*2 probably for 2 bytes for 16 bits
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    time_stamps = []

    voiced_frames = []
    tup = [None,None]
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                tup[0] = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write(f'-{frame.timestamp + frame.duration}')
                tup[1] = frame.timestamp + frame.duration
                time_stamps.append(tup)
                tup = [None, None]
                triggered = False
                # yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    # if voiced_frames:
    #     yield b''.join([f.bytes for f in voiced_frames])
    return time_stamps

RESAMPLE_RATE = 16000
import numpy as np
def getAudioSegs(key):
    obj = resource.Object(bucket_name, key)
    
    waveform, sample_rate = torchaudio.load(obj.get()["Body"])
    waveform = F.resample(waveform, sample_rate, RESAMPLE_RATE)[1].reshape(1,-1)
    
    torchaudio.save("test.wav", waveform, RESAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)
    
    spf = wave.open("test.wav", 'rb')
    data = spf.readframes(spf.getnframes())
    
    vad = webrtcvad.Vad(3)
    frames = frame_generator(30, data, 16000)
    frames = list(frames)
    
    segments = vad_collector(16000, 30, 300, vad, frames)
    

    segments = np.array(segments).reshape(-1,2)

    frame_segments = segments * RESAMPLE_RATE
    
    speaker_segs = frame_segments[np.where((segments[:,1]-segments[:,0]) > 2)[0]]

    os.remove("test.wav")
    return [waveform[:,int(speaker_segs[i][0]):int(speaker_segs[i][1])] for i in range(len(speaker_segs))]

all_speaker_segs = [] #torch.load('all_speaker_segs0-5.pkl')

#already configured, just run this cell
import time
for sprefix in tqdm(range(30,len(speakers)), desc=" outer", position=0):
    if sprefix % 5 == 0 and sprefix != 30:
        torch.save(all_speaker_segs[-5:], f'all_speaker_segs{sprefix-5}-{sprefix}.pkl')
        all_speaker_segs = []
        time.sleep(10)
    speaker_segs = []
    files = client.list_objects(Bucket=bucket_name, Prefix=speakers[sprefix])
    for audio_file in tqdm(range(len(files["Contents"])), desc=" inner loop", position=1, leave=True):
        try:
            key = files["Contents"][audio_file]["Key"]
            speaker_segs.append(getAudioSegs(key))
        except Exception as e:
            speaker_segs.append([])
        if audio_file > 14:
            break
    all_speaker_segs.append(speaker_segs)