import os
import json
import uuid
import threading
import queue
import configparser
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from markdown import markdown
import app  # Import the provided app.py for transcription functions

api = FastAPI(title="Meeting Minutes Assistant API")

# Configuration
CONFIG_FILE = "config.ini"
UPLOAD_DIR = os.path.join(os.path.expanduser("~"), "uploads")
METADATA_FILE = os.path.join(UPLOAD_DIR, "files.json")
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "output")

# Load config.ini
config = configparser.ConfigParser()
if os.path.exists(CONFIG_FILE):
    config.read(CONFIG_FILE, encoding='utf-8')
    if 'Paths' in config and 'output_directory' in config['Paths']:
        OUTPUT_DIR = config['Paths']['output_directory']
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize queues and threads from app.py
spk_txt_queue = queue.Queue()
audio_concat_queue = queue.Queue()
threading.Thread(target=app.write_txt, daemon=True).start()
threading.Thread(target=app.audio_concat_worker, args=(OUTPUT_DIR,), daemon=True).start()

class FileMetadata(BaseModel):
    id: str
    name: str
    date: str

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

@api.post("/upload", response_model=FileMetadata)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in app.support_audio_format:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Update metadata
    metadata = load_metadata()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata.append({"id": file_id, "name": file.filename, "date": current_date})
    save_metadata(metadata)
    
    # Trigger transcription in a separate thread
    threading.Thread(
        target=app.trans,
        args=([file_path], OUTPUT_DIR, 10)  # Using default split_chars=10
    ).start()
    
    return {"id": file_id, "name": file.filename, "date": current_date}

@api.get("/files", response_model=List[FileMetadata])
async def list_files():
    return load_metadata()

@api.get("/meeting-minutes/{file_id}")
async def get_meeting_minutes(file_id: str):
    metadata = load_metadata()
    file_entry = next((f for f in metadata if f["id"] == file_id), None)
    if not file_entry:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Find the transcribed text file
    audio_name = os.path.splitext(file_entry["name"])[0]
    date = file_entry["date"].split()[0]
    spk_txt_dir = os.path.join(OUTPUT_DIR, date, audio_name)
    txt_files = []
    for spk in range(10):  # Assuming a reasonable number of speakers
        spk_file = os.path.join(spk_txt_dir, f"spk{spk}.txt")
        if os.path.exists(spk_file):
            with open(spk_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Preprocess content to replace '-->' and reduce newlines
            processed_content = content.replace('-->', ':').replace('\n\n', '\n')
            # Convert to Markdown
            markdown_content = f"## Speaker {spk}\n\n{processed_content}"
            txt_files.append(markdown_content)
    
    if not txt_files:
        raise HTTPException(status_code=404, detail="Meeting minutes not found")
    
    return {"content": "\n\n".join(txt_files)}

@api.get("/meeting-minutes/{file_id}/pdf")
async def get_meeting_minutes_pdf(file_id: str):
    metadata = load_metadata()
    file_entry = next((f for f in metadata if f["id"] == file_id), None)
    if not file_entry:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Generate PDF from Markdown
    audio_name = os.path.splitext(file_entry["name"])[0]
    date = file_entry["date"].split()[0]
    spk_txt_dir = os.path.join(OUTPUT_DIR, date, audio_name)
    markdown_content = []
    for spk in range(10):
        spk_file = os.path.join(spk_txt_dir, f"spk{spk}.txt")
        if os.path.exists(spk_file):
            with open(spk_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # Preprocess content to replace '-->' and reduce newlines
            processed_content = content.replace('-->', ':').replace('\n\n', '\n')
            markdown_content.append(f"## Speaker {spk}\n\n{processed_content}")
    
    if not markdown_content:
        raise HTTPException(status_code=404, detail="Meeting minutes not found")
    
    # Convert Markdown to HTML, then to PDF
    html_content = markdown("\n\n".join(markdown_content))
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(html_content, styles["Normal"])]
    doc.build(story)
    
    return FileResponse(pdf_path, filename=f"{audio_name}_minutes.pdf")

@api.delete("/files/{file_id}")
async def delete_file(file_id: str):
    metadata = load_metadata()
    file_entry = next((f for f in metadata if f["id"] == file_id), None)
    if not file_entry:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Remove file and associated data
    file_ext = os.path.splitext(file_entry["name"])[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove output directory
    audio_name = os.path.splitext(file_entry["name"])[0]
    date = file_entry["date"].split()[0]
    output_dir = os.path.join(OUTPUT_DIR, date, audio_name)
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    
    # Remove PDF if exists
    pdf_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    # Update metadata
    metadata = [f for f in metadata if f["id"] != file_id]
    save_metadata(metadata)
    
    return {"message": "File deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8099)