import pysndfile
from pysndfile import PySndfile
# print(pysndfile.get_sndfile_formats())
import math
import numpy as np
from six.moves import range

def framesig(sig,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

    padlen = int((numframes-1)*frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig,zeros))

    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len),(numframes,1))
    return frames*win

def enframe(audio, sampling_rate, frame_len, frame_jump):
    frames = framesig(audio, frame_len*sampling_rate/1000, frame_jump*sampling_rate/1000)
    num_frames, features = frames.shape
    return frames, num_frames, features

def get_libri_labels(audio_name, feature_length):
    root = 'all_labels_target/'
    label = np.zeros(1)
    file_name = root+audio_name+'.txt'
    file = open(file_name,'r')
    for line in file:
    # i += 1
        line = line[0:-1]
        label = np.vstack((label, line))
    # print(line)
    # print(i)
    label = label[1:-2]
    label = label.astype(np.int64)
    a,b = label.shape
    # print('shape %d'%(a))
    # print('feature length %d'%(feature_length))
    if (feature_length - a) == 2:
        # print('aa')
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    elif (feature_length - a ) == 1:
        # print('bb')
        label = np.vstack((label, label[-1]))
    elif (feature_length - a) == 3:
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    file.close()
    return label

def get_ted_labels(audio_name, feature_length):
    root = 'all_labels_ted/'
    label = np.zeros(1)
    file_name = root+audio_name+'.txt'
    file = open(file_name,'r')
    for line in file:
    # i += 1
        line = line[0:-1]
        label = np.vstack((label, line))
    # print(line)
    # print(i)
    label = label[1:-2]
    label = label.astype(np.int64)
    a,b = label.shape
    if (feature_length - a) == 2:
        # print('aa')
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    elif (feature_length - a ) == 1:
        # print('bb')
        label = npy.vstack((label, label[-1]))
    elif (feature_length - a) == 3:
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    file.close()
    return label

def get_vox_labels(audio_path, feature_length):
    label = np.zeros(1)
    file = open(audio_path,'rb')
    for line in file:
    # i += 1
        line = line[0:-1]
        label = np.vstack((label, line))
    # print(line)
    # print(i)
    label = label[1:-2]
    label = label.astype(np.int64)
    a,b = label.shape
    if (feature_length - a) == 2:
        # print('aa')
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    elif (feature_length - a ) == 1:
        # print('bb')
        label = npy.vstack((label, label[-1]))
    elif (feature_length - a) == 3:
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    elif (feature_length - a) == 4:
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    elif (feature_length -a) == 5:
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
        label = np.vstack((label, label[-1]))
    file.close()
    return label
def audioToInputVector(audio, fs, numsamples, numcontext):
    orig_inputs = enframe(audio, fs, 10, 10)
    orig_inputs = orig_inputs[0]
	# print(orig_inputs)
    train_inputs = np.array([],np.float32)
    train_inputs.resize((orig_inputs.shape[0], numsamples+2*numsamples*numcontext))
	# print(train_inputs.shape)
    empty_raw = np.array([])
    empty_raw.resize((numsamples))

    time_slices = list(range(train_inputs.shape[0]))
	# print(time_slices)
    context_past_min = time_slices[0] +numcontext
	# print(context_past_min)
    context_future_max = time_slices[-1] - numcontext
    # print(context_future_max)
    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_raw for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past)+len(data_source_past)==numcontext)

        need_empty_future = max(0, (time_slice- context_future_max))
        empty_source_future = list(empty_raw for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice+ 1:time_slice+ numcontext+1]
        assert(len(empty_source_future)+len(data_source_future)==numcontext)
        if need_empty_past:
			# print(empty_source_past.shape)
			# print(data_source_past.shape)
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future
        past = np.reshape(past, numcontext*numsamples)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext*numsamples)
        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice])== numsamples+ 2*numcontext*numsamples)
    train_inputs = (train_inputs-np.mean(train_inputs))/np.std(train_inputs)  
    return train_inputs
def get_utterance_ted(file_name):
	fileId = open(file_name, 'r')
	i = 0
	for line in fileId:
		words = line.split(' ')
		audio_name = words[0]
		end_time = words[-1]
		start_time = words[-2]
		yield audio_name, start_time, end_time

def read_audio(name_audio):
    soundIO = PySndfile(name_audio)

    frames = soundIO.read_frames()
    s_rate = soundIO.samplerate()
    return frames, s_rate

def audiofile_to_input_vector_ted(audio_filename, numsamples=160, numcontext=15):
	# fileId = open(file_name, 'r')
    words = audio_filename.split(' ')
    audio_name = words[0]
    end_time = words[-1]
    start_time = words[-2]
    ss = start_time.split()
    if ss[-1] == '\n':
        start_time = float(start_time[0:-1])
    else:
        start_time = float(start_time)

    ee = end_time.split()
    if ee[-1]== '\n':
        end_time = float(end_time[0:-1])
    else:
        end_time = float(end_time)
    label_file_name = audio_name
    audio_split = audio_name.split('-')
    audio_name = audio_split[0]
    audio = 'TEDLIUM_release1/ted_all/'+audio_name+'.sph.wav'
    r_audio, samplerate = read_audio(audio)
    framed_audio = r_audio[int(start_time*samplerate): int(end_time*samplerate)]
    train_input = audioToInputVector(framed_audio, samplerate, 160, 15)
    train_label = get_ted_labels(label_file_name, train_input.shape[0])
    return (train_input, train_label)

def audiofile_to_input_vector_libri(audio_filename, numsamples=160, numcontext=15):
    audio_name = audio_filename.split(' ')[0]
    audio = 'LibriSpeech/libri_all/'+audio_name+'.flac'
    r_audio, sampling_rate = read_audio(audio)
    train_input = audioToInputVector(r_audio, sampling_rate, 160, 15)
    train_label = get_libri_labels(audio_name, train_input.shape[0])
    return (train_input, train_label)

def audiofile_to_input_vector_vox(audio_path, label_path, numsamples=160, numcontext=15):
    total_path = audio_path.split('/')
    t_path = total_path[-1].split('.')
    audio = total_path[-4]+'/'+total_path[-3]+'/'+total_path[-2]+'/'+t_path[0]+'.wav'
    r_audio, sampling_rate = read_audio(audio)
    train_input = audioToInputVector(r_audio, sampling_rate, 160, 15)
    train_label = get_vox_labels(label_path, train_input.shape[0])
    return (train_input, train_label)




