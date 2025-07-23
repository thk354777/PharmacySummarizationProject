import boto3
import io
import csv
import time
from together import Together  # สมมติใช้ไลบรารีนี้สำหรับ Whisper API
from botocore.exceptions import ClientError
from fastapi import FastAPI, APIRouter, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import datetime
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # หรือ ["*"] ถ้ายังไม่ deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter(prefix="/api", tags=["API"])

# ตั้งค่า S3 และ Together API
S3_BUCKET = "asr-bu-intern"
PREFIX = "meeting_chunks/"
s3 = boto3.client("s3")
client = Together(api_key=api_key)

def download_s3_file(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read()  # bytes

def transcribe_chunk(audio_bytes):
    # ส่งไฟล์ audio bytes ไป Whisper model (ในรูปแบบ file-like object)
    file_like = io.BytesIO(audio_bytes)
    file_like.name = "chunk.wav"  # บาง API ต้องมีชื่อไฟล์
    response = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=file_like,
        language="th",
        response_format="json",
        timestamp_granularities=["segment"]
    )
    return response.text.strip()

def save_transcripts_to_csv(results, csv_key):
    # สร้าง CSV ใน memory
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["time", "text"])
    writer.writeheader()
    writer.writerows(results)
    csv_buffer.seek(0)
    
    # อัปโหลด CSV ขึ้น S3
    s3.put_object(Bucket=S3_BUCKET, Key=csv_key, Body=csv_buffer.getvalue().encode("utf-8-sig"))
    print(f"Uploaded CSV to s3://{S3_BUCKET}/{csv_key}")

def process_chunks_and_transcribe(s3_prefix):
    # ดึงรายการไฟล์ chunk จาก S3 โฟลเดอร์ s3://bucket/s3_prefix/
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix)

    results = []
    start_time = time.time()

    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            print(f"Processing chunk: {key}")

            audio_bytes = download_s3_file(S3_BUCKET, key)
            text = transcribe_chunk(audio_bytes)

            # time_range เอาจากชื่อไฟล์ (สมมติ format ชื่อไฟล์ตามที่เคยกำหนด)
            time_range = key.split("/")[-1].split("_", 1)[-1].replace(".wav", "")
            results.append({
                "time": time_range,
                "text": text
            })

    end_time = time.time()
    print(f"Total transcription time: {end_time - start_time:.2f} sec")

    # อัปโหลดไฟล์ CSV
    csv_key = f"{s3_prefix}chunk_transcript.csv"
    save_transcripts_to_csv(results, csv_key)

    return csv_key


# ตัวอย่างเรียกใช้งาน
# if __name__ == "__main__":
#     folder_prefix = "meeting_chunks/20250719_192542/"  # ปรับตาม folder จริง
#     csv_path = process_chunks_and_transcribe(folder_prefix)
#     print("CSV saved at:", csv_path)


@router.get("/folders", response_model=List[str])
def list_folders():
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=PREFIX, Delimiter="/")
    folders = [cp["Prefix"].split("/")[-2] for cp in response.get("CommonPrefixes", [])]
    return folders



def timedelta_str(t: datetime.timedelta) -> str:
    total_seconds = int(t.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}_{minutes:02}_{seconds:02}"

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    # === 1. อ่านไฟล์ MP3 เป็น bytes (in-memory)
    mp3_bytes = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")

    # === 2. แบ่ง chunk ขนาด 15 วิ
    segment_ms = 15 * 1000
    now_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_prefix = f"meeting_chunks/{now_folder}/"
    uploaded_chunks = []

    for idx in range(0, len(audio), segment_ms):
        segment = audio[idx:idx + segment_ms]

        start = datetime.timedelta(milliseconds=idx)
        end = datetime.timedelta(milliseconds=min(idx + segment_ms, len(audio)))

        timerange = f"{timedelta_str(start)}-{timedelta_str(end)}"
        filename = f"{now_folder}_{timerange}.wav"
        s3_key = f"{s3_prefix}{filename}"

        # === 3. แปลง segment เป็น WAV (in-memory)
        wav_buffer = io.BytesIO()
        segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # === 4. อัปโหลดขึ้น S3
        s3.upload_fileobj(wav_buffer, S3_BUCKET, s3_key)
        uploaded_chunks.append(s3_key)
    csv_path = process_chunks_and_transcribe(s3_prefix)
    return {
        "message": "Uploaded & chunked directly to S3",
        "chunks": uploaded_chunks,
        "folder": s3_prefix,
        "csv_path": csv_path
    }

def preprocess_text(text, threshold=0.05):
    if not isinstance(text, str):
        return text
    total_length = len(text)
    if total_length == 0:
        return text
    space_count = text.count(' ')
    if space_count / total_length > threshold:
        return ""
    return text

@router.get("/record/{folder}/maincontent")
def get_maincontent(folder: str):
    csv_key = f"{PREFIX}{folder.rstrip('/')}/chunk_transcript.csv"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=csv_key)
        csv_content = obj["Body"].read().decode("utf-8-sig")
        results = []
        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            processed_text = preprocess_text(row["text"])
            if processed_text:
                results.append({"time": row["time"], "text": processed_text})
        return {"transcript": results}
    except ClientError as e:
        return {"error": "CSV file not found", "details": str(e)}

app.include_router(router)
# with open("chunks/2025-07-15_13-06/audio/chunk_000.mp3", "rb") as f:
#     file_like = io.BytesIO(f.read())
#     file_like.name = "chunk.wav"  # บาง API ต้องการชื่อไฟล์

# response = client.audio.transcriptions.create(
#     model="openai/whisper-large-v3",
#     file=file_like,
#     language="th",
#     response_format="json",
#     timestamp_granularities=["segment"]
# )

# print(response.text.strip())



