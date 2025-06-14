import sounddevice as sd
import numpy as np
import torch
import srt
import datetime
import customtkinter as ctk
import threading
import os
from queue import Queue
from funasr import AutoModel
import ffmpeg
from pydub import AudioSegment
import io
import scipy.io.wavfile as wavfile

# ==== 参数 ====
OUTPUT_WAV = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/recorded.wav"
OUTPUT_TXT = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/transcript.txt"
OUTPUT_SRT = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/transcript.srt"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 20  # Seconds per segment
SPLIT_NUMBER = 1000  # Maximum characters for merging speaker text

# ==== 加载 FunASR 模型 ====
home_directory = os.path.expanduser("~")
asr_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
vad_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
punc_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
spk_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "iic", "speech_campplus_sv_zh-cn_16k-common")
model = AutoModel(
    model=asr_model_path,
    model_revision="v2.0.4",
    vad_model=vad_model_path,
    vad_model_revision="v2.0.4",
    punc_model=punc_model_path,
    punc_model_revision="v2.0.4",
    spk_model=spk_model_path,
    spk_model_revision="v2.0.4",
    ngpu=1,
    ncpu=4,
    device="cuda",
    disable_pbar=True,
    disable_log=True,
    disable_update=True
)

# ==== GUI 应用程序 ====
class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FunASR Real-Time Transcriber")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Configure customtkinter appearance
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # State variables
        self.is_recording = False
        self.transcription_thread = None
        self.transcription_queue = Queue()
        self.segments = []
        self.srt_entries = []
        self.start_time = datetime.timedelta(seconds=0)
        self.segment_idx = 1

        # Build UI
        self.setup_ui()

        # Start queue processing
        self.process_queue()

    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Real-Time Audio Transcriber with FunASR",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Status: Idle",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=5)

        self.transcription_text = ctk.CTkTextbox(
            self.main_frame,
            height=300,
            font=ctk.CTkFont(size=14),
            wrap="word",
            corner_radius=10
        )
        self.transcription_text.pack(pady=10, padx=10, fill="both", expand=True)
        self.transcription_text.insert("end", "Transcriptions will appear here...\n")
        self.transcription_text.configure(state="disabled")

        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(pady=10)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Recording",
            command=self.start_recording,
            corner_radius=8,
            width=150
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="Stop Recording",
            command=self.stop_recording,
            corner_radius=8,
            width=150,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)

        self.appearance_switch = ctk.CTkSwitch(
            self.main_frame,
            text="Dark Mode",
            command=self.toggle_appearance,
            onvalue="Dark",
            offvalue="Light"
        )
        self.appearance_switch.pack(pady=10)

    def toggle_appearance(self):
        mode = self.appearance_switch.get()
        ctk.set_appearance_mode(mode)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text="Status: Recording...")
            self.segments = []
            self.srt_entries = []
            self.start_time = datetime.timedelta(seconds=0)
            self.segment_idx = 1
            self.transcription_text.configure(state="normal")
            self.transcription_text.delete("1.0", "end")
            self.transcription_text.insert("end", "Recording started...\n")
            self.transcription_text.configure(state="disabled")
            self.transcription_thread = threading.Thread(target=self.record_and_transcribe)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Status: Saving...")
            self.save_transcriptions()
            self.status_label.configure(text="Status: Idle")

    def to_date(self, milliseconds):
        """Convert timestamp to SRT format time"""
        time_obj = datetime.timedelta(milliseconds=milliseconds)
        return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"

    def record_and_transcribe(self):
        while self.is_recording:
            # Record audio segment
            recording = sd.rec(int(SEGMENT_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            waveform = np.array(recording, dtype=np.int16)

            # Save audio segment to temporary WAV file
            temp_wav = "temp_segment.wav"
            wavfile.write(temp_wav, SAMPLE_RATE, waveform)

            # Convert to bytes for FunASR
            audio_bytes, _ = (
                ffmpeg.input(temp_wav, threads=0)
                .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )

            # Process audio with FunASR
            res = model.generate(input=audio_bytes, batch_size_s=300, is_final=True, sentence_timestamp=True)
            if not res or not res[0].get('sentence_info'):
                self.transcription_queue.put((f"Segment {self.segment_idx}: No transcription result", self.segment_idx))
                self.start_time += datetime.timedelta(seconds=SEGMENT_DURATION)
                self.segment_idx += 1
                os.remove(temp_wav)
                continue

            rec_result = res[0]
            sentences = []
            for sentence in rec_result["sentence_info"]:
                start_ms = sentence["start"]
                end_ms = sentence["end"]
                start = self.to_date(start_ms)
                end = self.to_date(end_ms)
                text = sentence["text"]
                spk = sentence["spk"]
                
                if sentences and spk == sentences[-1]["spk"] and len(sentences[-1]["text"]) < SPLIT_NUMBER:
                    sentences[-1]["text"] += " " + text
                    sentences[-1]["end"] = end
                    sentences[-1]["end_ms"] = end_ms
                else:
                    sentences.append({"text": text, "start": start, "end": end, "spk": spk, "start_ms": start_ms, "end_ms": end_ms})

            # Process sentences for transcription and SRT
            for sentence in sentences:
                text = sentence["text"]
                spk = sentence["spk"]
                start = sentence["start"]
                end = sentence["end"]
                start_ms = sentence["start_ms"]
                end_ms = sentence["end_ms"]

                # Add to segments for text output
                self.segments.append(f"Speaker {spk}: {text}")

                # Create SRT entry
                srt_entry = srt.Subtitle(
                    index=self.segment_idx,
                    start=datetime.timedelta(milliseconds=start_ms),
                    end=datetime.timedelta(milliseconds=end_ms),
                    content=f"Speaker {spk}: {text}"
                )
                self.srt_entries.append(srt_entry)

                # Queue for UI display
                self.transcription_queue.put((f"Segment {self.segment_idx} (Speaker {spk}): {text}", self.segment_idx))
                self.segment_idx += 1

            self.start_time += datetime.timedelta(seconds=SEGMENT_DURATION)
            os.remove(temp_wav)

    def process_queue(self):
        try:
            while True:
                result, idx = self.transcription_queue.get_nowait()
                self.transcription_text.configure(state="normal")
                self.transcription_text.insert("end", f"{result}\n")
                self.transcription_text.see("end")
                self.transcription_text.configure(state="disabled")
                self.transcription_queue.task_done()
        except:
            self.root.after(100, self.process_queue)

    def save_transcriptions(self):
        # Ensure output directories exist
        os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_SRT), exist_ok=True)

        # Save text file
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(self.segments))

        # Save SRT file
        with open(OUTPUT_SRT, "w", encoding="utf-8") as f_srt:
            f_srt.write(srt.compose(self.srt_entries))

        self.transcription_text.configure(state="normal")
        self.transcription_text.insert("end", f"\nSaved transcript to {OUTPUT_TXT}\nSaved subtitles to {OUTPUT_SRT}\n")
        self.transcription_text.see("end")
        self.transcription_text.configure(state="disabled")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()
    app = TranscriberApp(root)
    app.run()