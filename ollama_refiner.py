import requests
import json
from queue import Queue
import time

class OllamaRefiner:
    def __init__(self, summarized_queue):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "qwen2.5:latest"
        self.conversation_history = []
        self.summarized_queue = summarized_queue
        self.max_retries = 3
        self.timeout = 30  # Increased timeout to 30 seconds

    def reset_conversation(self):
        """Reset conversation history for a new session."""
        self.conversation_history = []

    def summarize_text(self, text, speaker, segment_idx, start_ms, end_ms):
        """Send text to Ollama for summarization and queue the result."""
        prompt = (
            f"Summarize the following transcribed text from Speaker {speaker} concisely, preserving key information and meaning. "
            f"Include the speaker label 'Speaker {speaker}:' at the start of the summarized text. "
            f"Text: {text}"
        )
        self.conversation_history.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": True
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.ollama_url, json=payload, stream=True, timeout=self.timeout)
                response.raise_for_status()
                summarized_text = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if not data.get("done", False):
                            summarized_text += data["message"]["content"]
                if summarized_text:
                    self.conversation_history.append({"role": "assistant", "content": summarized_text})
                    self.summarized_queue.put((summarized_text, segment_idx, start_ms, end_ms))
                    return
                else:
                    print(f"Ollama returned empty summary on attempt {attempt + 1} for segment {segment_idx}")
            except Exception as e:
                print(f"Ollama summarization error on attempt {attempt + 1} for segment {segment_idx}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)  # Wait 2 seconds before retrying
                    continue
                else:
                    # Fallback to raw text after max retries
                    summarized_text = f"Speaker {speaker}: {text}"
                    self.conversation_history.append({"role": "assistant", "content": summarized_text})
                    self.summarized_queue.put((summarized_text, segment_idx, start_ms, end_ms))
                    return
        # Ensure fallback is queued even if loop exits without success
        summarized_text = f"Speaker {speaker}: {text}"
        self.conversation_history.append({"role": "assistant", "content": summarized_text})
        self.summarized_queue.put((summarized_text, segment_idx, start_ms, end_ms))