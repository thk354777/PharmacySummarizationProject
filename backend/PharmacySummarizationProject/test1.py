# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torchaudio
# import torch


# # device = 0 if torch.cuda.is_available() else "cpu"
# # print(device)
# # print(torch.cuda.get_device_name(0))

# model_path = "scb10x/monsoon-whisper-medium-gigaspeech2"
# device = "cuda"
# filepath = 'ขนตอนการทกทายทประชมอยางงาย #การพดในทสาธารณะ #เทคนคนำเสนอ #เทคนคการพด #การสอสาร.mp3'

# processor = WhisperProcessor.from_pretrained(model_path)
# model = WhisperForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype=torch.bfloat16
# )
# model.to(device)
# model.eval()

# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
#     language="th", task="transcribe"
# )
# array, sr = torchaudio.load(filepath)

# # แปลงเป็น mono หากไม่ใช่
# if array.shape[0] > 1:
#     array = array.mean(dim=0, keepdim=True)  # shape: [1, time]

# import torchaudio.transforms as T

# if sr != 16000:
#     resampler = T.Resample(sr, 16000)
#     array = resampler(array)
#     sr = 16000


# # ส่งเข้า processor (แปลงเป็น input_features)
# input_features = processor(array.squeeze(), sampling_rate=sr, return_tensors="pt").input_features
# input_features = input_features.to(device).to(torch.bfloat16)


# predicted_ids = model.generate(input_features)
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
# print(transcription)

###################################


# import torch
# from transformers import pipeline

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# lang = "th"
# task = "transcribe"

# pipe = pipeline(
#     task="automatic-speech-recognition",
#     model="nectec/Pathumma-whisper-th-large-v3",
#     torch_dtype=torch_dtype,
#     device=device,
# )
# pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task=task)

# text = pipe("ขนตอนการทกทายทประชมอยางงาย #การพดในทสาธารณะ #เทคนคนำเสนอ #เทคนคการพด #การสอสาร.mp3", return_timestamps=True)["text"]
# print(text)

######################################


# import torch
# from transformers import pipeline
# import datetime
# import csv

# # Setup
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# # Load Whisper pipeline
# pipe = pipeline(
#     task="automatic-speech-recognition",
#     model="nectec/Pathumma-whisper-th-large-v3",
#     torch_dtype=torch_dtype,
#     device=device,
# )

# lang = "th"
# task = "transcribe"
# pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task=task)

# # Transcribe with timestamps
# result = pipe("ขนตอนการทกทายทประชมอยางงาย #การพดในทสาธารณะ #เทคนคนำเสนอ #เทคนคการพด #การสอสาร.mp3", return_timestamps=True)

# segments = result["chunks"]  # มี start, end, text
# print(segments)
# # สร้างฟังก์ชันแปลงวินาที -> HH:MM:SS
# def to_hhmmss(seconds):
#     return str(datetime.timedelta(seconds=int(seconds))).zfill(8)

# # จัด transcript เป็น chunk 15 วิ
# chunk_size = 15
# output_chunks = []
# current_chunk_start = 0
# current_chunk_text = []

# for seg in segments:
#     start = seg["timestamp"][0]
#     end = seg["timestamp"][1]
#     text = seg["text"]

#     # ถ้า segment นี้เลยช่วง chunk ปัจจุบัน
#     while start >= current_chunk_start + chunk_size:
#         if current_chunk_text:
#             output_chunks.append({
#                 "time": f"{to_hhmmss(current_chunk_start)}-{to_hhmmss(current_chunk_start + chunk_size)}",
#                 "text": " ".join(current_chunk_text)
#             })
#             current_chunk_text = []
#         current_chunk_start += chunk_size

#     current_chunk_text.append(text)

# # บันทึก chunk สุดท้าย
# if current_chunk_text:
#     output_chunks.append({
#         "time": f"{to_hhmmss(current_chunk_start)}-{to_hhmmss(current_chunk_start + chunk_size)}",
#         "text": " ".join(current_chunk_text)
#     })

# # บันทึกลง CSV
# with open("transcript_chunks.csv", "w", newline="", encoding="utf-8-sig") as f:
#     writer = csv.DictWriter(f, fieldnames=["time", "text"])
#     writer.writeheader()
#     writer.writerows(output_chunks)

# print("✅ Transcript แบ่ง chunk เสร็จแล้ว บันทึกในไฟล์: transcript_chunks.csv")


#################

import os
from pydub import AudioSegment
from transformers import pipeline
import csv
import datetime
import torch
import time

# # === Step 1: ตัดเสียง ===
audio = AudioSegment.from_file("audio1805836484.m4a", format="mp4")
chunk_length_ms = 15 * 1000  # 15 วินาที
chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# สร้างโฟลเดอร์
os.makedirs("chunks", exist_ok=True)

for idx, chunk in enumerate(chunks):
    chunk.export(f"chunks/chunk_{idx:03}.wav", format="wav")

# === Step 2: โหลดโมเดล ===
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

pipe = pipeline(
    task="automatic-speech-recognition",
    model="nectec/Pathumma-whisper-th-large-v3",
    torch_dtype=torch_dtype,
    device=device,
)
lang = "th"
task = "transcribe"
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task=task)


start_time = time.time()

# === Step 3: ถอดเสียงแต่ละ chunk ===
results = []
for idx in range(len(chunks)):
    start_sec = idx * 15
    end_sec = min((idx + 1) * 15, len(audio) / 1000)
    time_range = f"{str(datetime.timedelta(seconds=int(start_sec))).zfill(8)}-{str(datetime.timedelta(seconds=int(end_sec))).zfill(8)}"
    
    filename = f"chunks/chunk_{idx:03}.wav"
    result = pipe(filename)
    text = result["text"].strip()
    
    results.append({
        "time": time_range,
        "text": text
    })

end_time = time.time()
elapsed_time = end_time - start_time
print(f"ใช้เวลาทั้งหมด: {elapsed_time:.2f} วินาที")


# === Step 4: บันทึกลง CSV ===
with open("chunk_transcript.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["time", "text"])
    writer.writeheader()
    writer.writerows(results)

print("Save in chunk_transcript.csv")