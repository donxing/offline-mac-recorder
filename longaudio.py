import numpy as np
import srt
import datetime
import customtkinter as ctk
import threading
import os
import logging
from queue import Queue
from funasr import AutoModel
import ffmpeg
import scipy.io.wavfile as wavfile
from ollama_refiner import OllamaRefiner
import tkinter.filedialog as filedialog
import soundfile as sf
import io

# ==== Logging Setup ====
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==== Parameters ====
OUTPUT_WAV = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/longaudio_recorded.wav"
RAW_OUTPUT_TXT = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/longaudio_raw_transcript.txt"
RAW_OUTPUT_SRT = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/longaudio_raw_transcript.srt"
REFINED_OUTPUT_TXT = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/longaudio_refined_transcript.txt"
REFINED_OUTPUT_SRT = "/Users/heelgoed/Documents/realtime_meeting_assistant/record/longaudio_refined_transcript.srt"
SAMPLE_RATE = 16000
PAUSE_THRESHOLD = 0.5  # Seconds of silence to start a new segment
SPLIT_NUMBER = 1000  # Maximum characters for merging speaker text
CHUNK_DURATION = 300  # Seconds per chunk for large files (5 minutes)

# ==== Load FunASR Model ====
home_directory = os.path.expanduser("~")
asr_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "asr", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
vad_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "asr", "speech_fsmn_vad_zh-cn-16k-common-pytorch")
punc_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "asr", "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
spk_model_path = os.path.join(home_directory, ".cache", "modelscope", "hub", "models", "asr", "speech_campplus_sv_zh-cn_16k-common")
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
    disable_pbar=False,  # Enable progress bar for debugging
    disable_log=False,
    disable_update=True
)

# ==== GUI Application ====
class LongAudioTranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FunASR Long Audio Transcriber")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Configure customtkinter appearance
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Create refined transcription window
        self.refined_window = ctk.CTkToplevel(self.root)
        self.refined_window.geometry("800x700")
        self.refined_window.resizable(True, True)
        self.refined_window.protocol("WM_DELETE_WINDOW", self.on_refined_window_close)

        # State variables
        self.is_processing = False
        self.transcription_thread = None
        self.raw_transcription_queue = Queue()
        self.refined_transcription_queue = Queue()
        self.raw_segments = []
        self.raw_srt_entries = []
        self.refined_segments = []
        self.refined_srt_entries = []
        self.segment_idx = 1
        self.refiner = OllamaRefiner(self.refined_transcription_queue)
        self.current_chunk = 0
        self.total_chunks = 0

        # Build UI
        self.setup_ui()

        # Start queue processing
        self.process_queues()

    def setup_ui(self):
        logger.info("Setting up UI components")
        # Main window UI (Raw Transcription)
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Raw FunASR Transcription",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Status: Idle",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=5)

        self.progress_label = ctk.CTkLabel(
            self.main_frame,
            text="Progress: 0/0 chunks processed",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(pady=5)

        self.raw_transcription_text = ctk.CTkTextbox(
            self.main_frame,
            height=400,
            font=ctk.CTkFont(size=14),
            wrap="word",
            corner_radius=10
        )
        self.raw_transcription_text.pack(pady=10, padx=10, fill="both", expand=True)
        self.raw_transcription_text.insert("end", "Raw transcriptions will appear here...\n")
        self.raw_transcription_text.configure(state="disabled")

        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(pady=10)

        # Button for long audio processing
        self.process_long_audio_button = ctk.CTkButton(
            self.button_frame,
            text="Process Long Audio",
            command=self.start_process_long_audio,
            corner_radius=8,
            width=150,
            fg_color="#2196F3",
            hover_color="#1976D2"
        )
        self.process_long_audio_button.pack(side="left", padx=5)
        logger.info("Process Long Audio button created")

        self.appearance_switch = ctk.CTkSwitch(
            self.main_frame,
            text="Dark Mode",
            command=self.toggle_appearance,
            onvalue="Dark",
            offvalue="Light"
        )
        self.appearance_switch.pack(pady=10)

        # Refined transcription window UI
        self.refined_frame = ctk.CTkFrame(self.refined_window, corner_radius=10)
        self.refined_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.refined_title_label = ctk.CTkLabel(
            self.refined_frame,
            text="Ollama Refined Transcription",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.refined_title_label.pack(pady=10)

        self.refined_status_label = ctk.CTkLabel(
            self.refined_frame,
            text="Status: Idle",
            font=ctk.CTkFont(size=14)
        )
        self.refined_status_label.pack(pady=5)

        self.refined_transcription_text = ctk.CTkTextbox(
            self.refined_frame,
            height=400,
            font=ctk.CTkFont(size=14),
            wrap="word",
            corner_radius=10
        )
        self.refined_transcription_text.pack(pady=10, padx=10, fill="both", expand=True)
        self.refined_transcription_text.insert("end", "Refined transcriptions will appear here...\n")
        self.refined_transcription_text.configure(state="disabled")

    def toggle_appearance(self):
        mode = self.appearance_switch.get()
        logger.info(f"Switching appearance to {mode} mode")
        ctk.set_appearance_mode(mode)

    def on_refined_window_close(self):
        logger.info("Attempting to close refined window, preventing closure")
        self.refined_window.withdraw()
        self.root.after(100, self.refined_window.deiconify)

    def start_process_long_audio(self):
        if not self.is_processing:
            logger.info("Opening file dialog for long audio file selection")
            file_path = filedialog.askopenfilename(
                title="Select Long Audio or Video File",
                filetypes=[("Audio/Video files", "*.wav *.mp4 *.mp3"), ("All files", "*.*")]
            )
            if file_path:
                logger.info(f"Processing long audio file: {file_path}")
                self.is_processing = True
                self.process_long_audio_button.configure(state="disabled")
                self.status_label.configure(text="Status: Processing Long Audio...")
                self.refined_status_label.configure(text="Status: Processing...")
                self.progress_label.configure(text="Progress: 0/0 chunks processed")
                self.clear_transcriptions()
                self.raw_transcription_text.configure(state="normal")
                self.raw_transcription_text.delete("1.0", "end")
                self.raw_transcription_text.insert("end", f"Processing {os.path.basename(file_path)}...\n")
                self.raw_transcription_text.configure(state="disabled")
                self.refined_transcription_text.configure(state="normal")
                self.refined_transcription_text.delete("1.0", "end")
                self.refined_transcription_text.insert("end", "Refinement started...\n")
                self.refined_transcription_text.configure(state="disabled")
                self.transcription_thread = threading.Thread(target=lambda: self.process_long_audio_file(file_path))
                self.transcription_thread.daemon = True
                self.transcription_thread.start()

    def clear_transcriptions(self):
        logger.info("Clearing transcriptions")
        self.raw_segments = []
        self.raw_srt_entries = []
        self.refined_segments = []
        self.refined_srt_entries = []
        self.segment_idx = 1
        self.current_chunk = 0
        self.total_chunks = 0
        self.refiner.reset_conversation()

    def to_date(self, milliseconds):
        """Convert timestamp to SRT format time"""
        time_obj = datetime.timedelta(milliseconds=milliseconds)
        return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"

    def process_long_audio_file(self, file_path):
        try:
            logger.info(f"Processing audio file: {file_path}")
            # Determine file type
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".wav":
                # Read WAV file directly
                audio_data, file_sample_rate = sf.read(file_path)
                logger.info(f"WAV file sample rate: {file_sample_rate}Hz")
                if file_sample_rate != SAMPLE_RATE:
                    logger.info(f"Resampling WAV from {file_sample_rate}Hz to {SAMPLE_RATE}Hz")
                    stream = ffmpeg.input(file_path).audio.filter("aresample", SAMPLE_RATE)
                    audio_bytes = stream.output("-", format="wav", acodec="pcm_s16le", ac=1).run(capture_stdout=True)[0]
                    temp_wav = "temp_input.wav"
                    with open(temp_wav, "wb") as f:
                        f.write(audio_bytes)
                    audio_data, _ = sf.read(temp_wav)
                    os.remove(temp_wav)
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]  # Take first channel for mono

            elif file_extension in [".mp4", ".mp3"]:
                # Extract audio from MP4 or MP3
                logger.info(f"Extracting audio from {file_extension} and resampling to {SAMPLE_RATE}Hz")
                stream = ffmpeg.input(file_path).audio.filter("aresample", SAMPLE_RATE)
                audio_bytes = stream.output("-", format="wav", acodec="pcm_s16le", ac=1).run(capture_stdout=True)[0]
                temp_wav = "temp_input.wav"
                with open(temp_wav, "wb") as f:
                    f.write(audio_bytes)
                audio_data, _ = sf.read(temp_wav)
                os.remove(temp_wav)
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]  # Take first channel for mono

            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: WAV, MP4, MP3")

            # Process audio in chunks to manage memory
            chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
            total_samples = len(audio_data)
            self.total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            logger.info(f"Processing {total_samples} samples in {self.total_chunks} chunks of {chunk_samples} samples")

            for i in range(0, total_samples, chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                chunk_duration = len(chunk) / SAMPLE_RATE
                if len(chunk) > 0:
                    self.current_chunk += 1
                    self.root.after(0, lambda: self.progress_label.configure(
                        text=f"Progress: {self.current_chunk}/{self.total_chunks} chunks processed"
                    ))
                    self.process_audio_buffer(chunk, chunk_duration)

            # Save transcriptions
            self.root.after(0, self.finalize_file_processing)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing file: {error_msg}")
            self.root.after(0, lambda: self.handle_processing_error(error_msg))
        finally:
            self.is_processing = False
            logger.info("Finished processing file")

    def process_audio_buffer(self, audio_buffer, buffer_duration):
        # Save audio buffer to temporary WAV file
        temp_wav = "temp_segment.wav"
        wavfile.write(temp_wav, SAMPLE_RATE, audio_buffer)

        # Convert to bytes for FunASR
        try:
            audio_bytes, _ = (
                ffmpeg.input(temp_wav, threads=0)
                .output("-", format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error in processing buffer: {str(e)}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        # Process audio with FunASR
        try:
            res = model.generate(input=audio_bytes, batch_size_s=300, is_final=True, sentence_timestamp=True)
            logger.info(f"FunASR processing completed for chunk {self.current_chunk}")
        except Exception as e:
            logger.error(f"FunASR error in processing chunk {self.current_chunk}: {str(e)}")
            return

        if not res or not res[0].get('sentence_info'):
            logger.warning(f"No transcription result for chunk {self.current_chunk}, segment {self.segment_idx}")
            self.raw_transcription_queue.put((f"Segment {self.segment_idx}: No transcription result", self.segment_idx))
            self.segment_idx += 1
            return

        # Group sentences by pauses
        rec_result = res[0]
        segments = []
        current_segment = []
        last_end_ms = None

        for sentence in rec_result["sentence_info"]:
            start_ms = sentence["start"]
            end_ms = sentence["end"]
            text = sentence["text"]
            spk = sentence["spk"]

            # Check for pause to start a new segment
            if last_end_ms is not None and (start_ms - last_end_ms) / 1000.0 >= PAUSE_THRESHOLD:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            current_segment.append({"text": text, "start_ms": start_ms, "end_ms": end_ms, "spk": spk})
            last_end_ms = end_ms

        # Append the last segment if it exists
        if current_segment:
            segments.append(current_segment)

        # Process each segment
        for segment in segments:
            if not segment:
                continue

            start_ms = segment[0]["start_ms"]
            end_ms = segment[-1]["end_ms"]
            start = self.to_date(start_ms)
            end = self.to_date(end_ms)

            # Combine sentences in the segment
            segment_text = ""
            current_spk = segment[0]["spk"]
            combined_text = []
            for sentence in segment:
                if sentence["spk"] == current_spk and len(segment_text) + len(sentence["text"]) < SPLIT_NUMBER:
                    combined_text.append(sentence["text"])
                else:
                    if combined_text:
                        segment_text += f"Speaker {current_spk}: {' '.join(combined_text)} "
                    current_spk = sentence["spk"]
                    combined_text = [sentence["text"]]
            if combined_text:
                segment_text += f"Speaker {current_spk}: {' '.join(combined_text)} "

            # Raw transcription
            raw_text = segment_text.strip()
            self.raw_segments.append(raw_text)
            raw_srt_entry = srt.Subtitle(
                index=self.segment_idx,
                start=datetime.timedelta(milliseconds=start_ms),
                end=datetime.timedelta(milliseconds=end_ms),
                content=raw_text
            )
            self.raw_srt_entries.append(raw_srt_entry)
            self.raw_transcription_queue.put((f"Segment {self.segment_idx}: {raw_text}", self.segment_idx))

            # Send to Ollama refiner for language refinement
            try:
                self.refiner.summarize_text(raw_text, current_spk, self.segment_idx, start_ms, end_ms)
                logger.info(f"Sent segment {self.segment_idx} to Ollama for refinement")
            except Exception as e:
                logger.error(f"Ollama refinement error for segment {self.segment_idx}: {str(e)}")
                self.refined_transcription_queue.put((f"Segment {self.segment_idx}: Refinement failed - {str(e)}", self.segment_idx, start_ms, end_ms))

            self.segment_idx += 1

    def finalize_file_processing(self):
        logger.info("Finalizing file processing and saving transcriptions")
        self.status_label.configure(text="Status: Saving...")
        self.refined_status_label.configure(text="Status: Saving...")
        self.save_transcriptions()
        self.status_label.configure(text="Status: Idle")
        self.refined_status_label.configure(text="Status: Idle")
        self.process_long_audio_button.configure(state="normal")
        self.progress_label.configure(text="Progress: Completed")

    def handle_processing_error(self, error):
        logger.error(f"Handling error: {error}")
        self.status_label.configure(text=f"Status: Error - {error}")
        self.refined_status_label.configure(text="Status: Idle")
        self.process_long_audio_button.configure(state="normal")
        self.progress_label.configure(text="Progress: Error")
        self.raw_transcription_text.configure(state="normal")
        self.raw_transcription_text.insert("end", f"\nError: {error}\n")
        self.raw_transcription_text.see("end")
        self.raw_transcription_text.configure(state="disabled")

    def save_transcriptions(self):
        logger.info("Saving transcriptions")
        # Ensure output directories exist
        os.makedirs(os.path.dirname(RAW_OUTPUT_TXT), exist_ok=True)
        os.makedirs(os.path.dirname(REFINED_OUTPUT_TXT), exist_ok=True)

        # Save raw text file
        with open(RAW_OUTPUT_TXT, "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(self.raw_segments))

        # Save raw SRT file
        with open(RAW_OUTPUT_SRT, "w", encoding="utf-8") as f_srt:
            f_srt.write(srt.compose(self.raw_srt_entries))

        # Save refined text file
        with open(REFINED_OUTPUT_TXT, "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(self.refined_segments))

        # Save refined SRT file
        with open(REFINED_OUTPUT_SRT, "w", encoding="utf-8") as f_srt:
            f_srt.write(srt.compose(self.refined_srt_entries))

        self.raw_transcription_text.configure(state="normal")
        self.raw_transcription_text.insert("end", f"\nSaved raw transcript to {RAW_OUTPUT_TXT}\nSaved raw subtitles to {RAW_OUTPUT_SRT}\n")
        self.raw_transcription_text.see("end")
        self.raw_transcription_text.configure(state="disabled")
        self.refined_transcription_text.configure(state="normal")
        self.refined_transcription_text.insert("end", f"\nSaved refined transcript to {REFINED_OUTPUT_TXT}\nSaved refined subtitles to {REFINED_OUTPUT_SRT}\n")
        self.refined_transcription_text.see("end")
        self.refined_transcription_text.configure(state="disabled")

    def process_queues(self):
        try:
            # Process raw transcription queue
            while True:
                result, idx = self.raw_transcription_queue.get_nowait()
                self.raw_transcription_text.configure(state="normal")
                self.raw_transcription_text.insert("end", f"{result}\n")
                self.raw_transcription_text.see("end")
                self.raw_transcription_text.configure(state="disabled")
                self.raw_transcription_queue.task_done()
        except:
            pass

        try:
            # Process refined transcription queue
            while True:
                result, idx, start_ms, end_ms = self.refined_transcription_queue.get_nowait()
                self.refined_transcription_text.configure(state="normal")
                self.refined_transcription_text.insert("end", f"Segment {idx}: {result}\n")
                self.refined_transcription_text.see("end")
                self.refined_transcription_text.configure(state="disabled")
                self.refined_segments.append(result)
                srt_entry = srt.Subtitle(
                    index=idx,
                    start=datetime.timedelta(milliseconds=start_ms),
                    end=datetime.timedelta(milliseconds=end_ms),
                    content=result
                )
                self.refined_srt_entries.append(srt_entry)
                self.refined_transcription_queue.task_done()
        except:
            pass

        self.root.after(100, self.process_queues)

    def run(self):
        logger.info("Starting application mainloop")
        self.root.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()
    app = LongAudioTranscriberApp(root)
    app.run()