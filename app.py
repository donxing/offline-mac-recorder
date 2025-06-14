import os
import threading
import queue
import configparser
from datetime import timedelta, datetime
from pydub import AudioSegment
import ffmpeg
from funasr import AutoModel

spk_txt_queue = queue.Queue()
result_queue = queue.Queue()
audio_concat_queue = queue.Queue()

# Define model paths
home_directory = os.path.expanduser("~")
asr_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
asr_model_revision = "v2.0.4"
vad_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
vad_model_revision = "v2.0.4"
punc_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
punc_model_revision = "v2.0.4"
spk_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_campplus_sv_zh-cn_16k-common")
spk_model_revision = "v2.0.4"
ngpu = 1
device = "cuda"
ncpu = 4

# ASR model
model = AutoModel(model=asr_model_path,
                  model_revision=asr_model_revision,
                  vad_model=vad_model_path,
                  vad_model_revision=vad_model_revision,
                  punc_model=punc_model_path,
                  punc_model_revision=punc_model_revision,
                  spk_model=spk_model_path,
                  spk_model_revision=spk_model_revision,
                  ngpu=ngpu,
                  ncpu=ncpu,
                  device=device,
                  disable_pbar=True,
                  disable_log=True,
                  disable_update=True)

# Supported audio/video formats
support_audio_format = ['.mp3', '.m4a', '.aac', '.ogg', '.wav', '.flac', '.wma', '.aif']
support_video_format = ['.mp4', '.avi', '.mov', '.mkv']

def to_date(milliseconds):
    """Convert timestamp to SRT format time"""
    time_obj = timedelta(milliseconds=milliseconds)
    return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"

def to_milliseconds(time_str):
    """Convert time string to milliseconds"""
    time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    time_delta = time_obj - datetime(1900, 1, 1)
    milliseconds = int(time_delta.total_seconds() * 1000)
    return milliseconds

def trans(selected_file_list, save_path, split_number):
    """
    Perform speaker diarization and transcription for audio/video files.

    Args:
        selected_file_list (list): List of audio or video file paths to process.
        save_path (str): Directory to save separated files and transcribed text.
        split_number (int): Maximum characters for merging speaker text.
    """
    if not selected_file_list:
        print("Error: No input files selected.")
        return
    if not save_path:
        print("Error: No save path specified.")
        return

    for audio_path in selected_file_list:
        if not os.path.exists(audio_path):
            print(f"Error: Input file does not exist: {audio_path}")
            continue

        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        _, audio_extension = os.path.splitext(audio_path)
        print(f'Processing: {audio_path}')

        speaker_audios = {}

        try:
            audio_bytes, _ = (
                ffmpeg.input(audio_path, threads=0, hwaccel='cuda')
                .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
            res = model.generate(input=audio_bytes, batch_size_s=300, is_final=True, sentence_timestamp=True)
            rec_result = res[0]
            asr_result_text = rec_result['text']

            if asr_result_text:
                sentences = []
                for sentence in rec_result["sentence_info"]:
                    start = to_date(sentence["start"])
                    end = to_date(sentence["end"])
                    if sentences and sentence["spk"] == sentences[-1]["spk"] and len(sentences[-1]["text"]) < int(split_number):
                        sentences[-1]["text"] += "" + sentence["text"]
                        sentences[-1]["end"] = end
                    else:
                        sentences.append(
                            {"text": sentence["text"], "start": start, "end": end, "spk": sentence["spk"]}
                        )

                i = 0
                for stn in sentences:
                    stn_txt = stn['text']
                    start = stn['start']
                    end = stn['end']
                    spk = stn['spk']

                    date = datetime.now().strftime("%Y-%m-%d")
                    final_save_path = os.path.join(save_path, date, audio_name, str(spk))
                    os.makedirs(final_save_path, exist_ok=True)

                    file_ext = os.path.splitext(audio_path)[-1]
                    final_cut_file_name = f"{i}{file_ext}"
                    if file_ext in support_video_format:
                        final_cut_file_name = f"{i}.mp4"

                    final_save_file = os.path.join(final_save_path, final_cut_file_name)

                    spk_txt_dir = os.path.join(save_path, date, audio_name)
                    spk_txt_file = os.path.join(spk_txt_dir, f'spk{spk}.txt')
                    spk_txt_queue.put({'spk_txt_file': spk_txt_file, 'spk_txt': stn_txt, 'start': start, 'end': end})
                    i += 1

                    try:
                        if file_ext in support_audio_format:
                            (
                                ffmpeg.input(audio_path, threads=0, ss=start, to=end, hwaccel='cuda')
                                .output(final_save_file)
                                .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True,
                                     capture_stderr=True)
                            )
                        elif file_ext in support_video_format:
                            (
                                ffmpeg.input(audio_path, threads=0, ss=start, to=end, hwaccel='cuda')
                                .output(final_save_file, vcodec='libx264', crf=23, acodec='aac', ab='128k')
                                .run(cmd=["ffmpeg", "-nostdin"], overwrite_output=True, capture_stdout=True,
                                     capture_stderr=True)
                            )
                        else:
                            print(f'{audio_path} does not support segmentation.')
                    except ffmpeg.Error as e:
                        print(f"Error segmenting audio/video: {e}")

                    if spk not in speaker_audios:
                        speaker_audios[spk] = []
                    speaker_audios[spk].append({'file': final_save_file, 'audio_name': audio_name})

                print(f'{audio_path} segmentation completed')
                result_queue.put(f'{audio_path} segmentation completed')
                print(f'Transcription result: {rec_result["text"]}')
                audio_concat_queue.put(speaker_audios)
            else:
                print(f"{audio_path} has no transcription result")
        except Exception as e:
            print(f"Transcription error for {audio_path}: {e}")

def write_txt():
    """Write speaker text information from the queue to corresponding text files"""
    while True:
        item = spk_txt_queue.get()
        spk_txt_file = item['spk_txt_file']
        spk_txt = item['spk_txt']
        spk_start = item['start']
        spk_end = item['end']
        dir_path = os.path.dirname(spk_txt_file)
        os.makedirs(dir_path, exist_ok=True)
        with open(spk_txt_file, 'a', encoding='utf-8') as f:
            f.write(f"{spk_start} --> {spk_end}\n{spk_txt}\n\n")
        spk_txt_queue.task_done()

def audio_concat_worker(save_path_base):
    """
    Concatenate audio segments for each speaker into a single MP3 file.

    Args:
        save_path_base (str): Base directory to save concatenated audio files.
    """
    while True:
        speaker_audios_tmp = audio_concat_queue.get()
        for spk, audio_segments in speaker_audios_tmp.items():
            if not audio_segments:
                continue

            audio_name = audio_segments[0]['audio_name']
            output_file = os.path.join(save_path_base, datetime.now().strftime("%Y-%m-%d"), audio_name, f"{spk}.mp3")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            valid_inputs = [seg['file'] for seg in audio_segments if os.path.exists(seg['file'])]
            if not valid_inputs:
                print(f"Warning: No valid audio segments found for speaker {spk} in {audio_name}.")
                continue

            try:
                concat_audio = AudioSegment.from_file(valid_inputs[0])
                for i in range(1, len(valid_inputs)):
                    concat_audio = concat_audio + AudioSegment.from_file(valid_inputs[i])
                concat_audio.export(output_file, format="mp3")
                print(f"Concatenated audio for speaker {spk} to {output_file}")
            except Exception as e:
                print(f"Error concatenating audio for speaker {spk} in {audio_name}: {e}")
        audio_concat_queue.task_done()

if __name__ == '__main__':
    print("Starting project...")

    # Load configuration
    config = configparser.ConfigParser()
    config_file_path = "config.ini"
    output_directory = os.path.join(os.path.expanduser("~"), "output")
    if os.path.exists(config_file_path):
        try:
            config.read(config_file_path, encoding='utf-8')
            if 'Paths' in config and 'output_directory' in config['Paths']:
                output_directory = config['Paths']['output_directory']
                print(f"Read output directory from config: {output_directory}")
        except Exception as e:
            print(f"Error reading config file '{config_file_path}': {e}")

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Start worker threads
    threading.Thread(target=write_txt, daemon=True).start()
    threading.Thread(target=audio_concat_worker, args=(output_directory,), daemon=True).start()

    print("Worker threads started. Ready for transcription tasks.")