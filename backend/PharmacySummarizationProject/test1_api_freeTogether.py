from together import Together





import os
from pydub import AudioSegment
from transformers import pipeline
import csv
import torch
import time
from datetime import datetime, timedelta
from typing import Union
from fastapi import FastAPI
from fastapi import Query, UploadFile, File
import io
from pathlib import Path


import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import os

together_api_key = os.getenv("TOGETHER_API_KEY")
typhoon_api_key = os.getenv("TYPHOON_API_KEY")

client = Together(api_key=together_api_key)  

app = FastAPI()


@app.post("/transcribe")
# def transcribe(audio: str = Query(...)):
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    # audio = AudioSegment.from_file(audio, format="mp3")
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    chunk_length_ms = 15 * 1000  # 15 seconds
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_dir = os.path.join("chunks", date_time_str, "audio")
    os.makedirs(base_dir, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        filename = f"chunk_{idx:03}.mp3" #wav but longer
        filepath = os.path.join(base_dir, filename)
        chunk.export(filepath, format="mp3")
    results = []
    start_time = time.time()
    for idx in range(len(chunks)):
        start_sec = idx * 15
        end_sec = min((idx + 1) * 15, len(audio) / 1000)
        time_range = f"{str(timedelta(seconds=int(start_sec))).zfill(8)}-{str(timedelta(seconds=int(end_sec))).zfill(8)}"
        filename = os.path.join(base_dir, f"chunk_{idx:03}.mp3")
        with open(filename, "rb") as f:
            response = client.audio.transcriptions.create(
                model="openai/whisper-large-v3",
                file=f,
                language="th", 
                response_format="json",
                timestamp_granularities=["segment"]
            )
        result = response.text
        print(f"Transcription: {response.text}")
        # result = pipe(filename)
        text = response.text.strip()
        
        results.append({
            "time": time_range,
            "text": text
        })
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Use time: {elapsed_time:.2f} seconds")
    fileCSV = os.path.join(os.path.dirname(base_dir), "chunk_transcript.csv")
    with open(fileCSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "text"])
        writer.writeheader()
        writer.writerows(results)

    # Also save to frontend public directory
    target_base_dir = os.path.abspath(os.path.join(os.getcwd(), "../../frontend/public/chunk"))
    os.makedirs(target_base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_filename = f"chunk_transcript_{timestamp}.csv"
    fileCSV = os.path.join(target_base_dir, csv_filename)
    with open(fileCSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "text"])
        writer.writeheader()
        writer.writerows(results)


    llm = ChatOpenAI(
        model_name="typhoon-v2-70b-instruct", #"typhoon-v2.1-12b-instruct", "typhoon-v2-70b-instruct"
        temperature=0.0,
        max_tokens=3000, # Total Context Window 8k เป็น token input + token output แล้ว
        openai_api_key=typhoon_api_key,
        openai_api_base="https://api.opentyphoon.ai/v1",
    )
    df = pd.read_csv(fileCSV, encoding="utf-8-sig")
    prompt_template = ChatPromptTemplate.from_template(
        """โปรดแก้ไขคำผิดหรือคำที่สะกดไม่เหมาะสมในประโยคต่อไปนี้ให้ถูกต้อง โดยเฉพาะคำที่มักสะกดผิด เช่น "เภสัชกรรม", "กรรมการสภา" หรือคำที่ใช้ในบริบทไม่เหมาะสม  
    **ตอบกลับเฉพาะประโยคที่แก้ไขแล้วเท่านั้น และห้ามใส่เครื่องหมายคำพูด (" ")**

    ตัวอย่าง:
    ผิด: เขาเป็นกำมะการสภาของวิทยาลัย
    ถูก: เขาเป็นกรรมการสภาของวิทยาลัย

    ประโยค:
    "{sentence}" """
    )
    corrected_texts = []
    for text in df["text"].fillna(""):
        prompt = prompt_template.format_messages(sentence=text)
        response = llm(prompt)
        corrected_texts.append(response.content.strip())

    df["text_corrected"] = corrected_texts

    df.to_csv("chunk_transcript2_corrected2.csv", index=False)

    llm = ChatOpenAI(
        model_name="typhoon-v2-70b-instruct", #"typhoon-v2.1-12b-instruct", "typhoon-v2-70b-instruct"
        temperature=0.5,
        max_tokens=3000, # Total Context Window 8k เป็น token input + token output แล้ว
        openai_api_key=typhoon_api_key,
        openai_api_base="https://api.opentyphoon.ai/v1",
    )


    all_text = "\n".join(df["text_corrected"].dropna().astype(str))
    docs = [Document(page_content=all_text)]
    print(docs)



    map_prompt = PromptTemplate.from_template("""
    ช่วยสรุปข้อความถอดเสียงในที่ประชุมนี้ให้กระชับและชัดเจนเป็นภาษาไทย  
    และถ้ามีคำที่เป็นภาษาอังกฤษในประโยค ให้ใช้คำภาษาอังกฤษนั้นแทน  
    เช่น โปรเจค -> project, เดดไล -> deadline เป็นต้น:
    "{text}"

    สรุป:
    """)

    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    summaries = []
    for i, doc in enumerate(split_docs):
        summary = map_chain.run({"text": doc.page_content})
        summaries.append(summary)
        print(f"💡 Chunk {i+1} summary:\n{summary}\n")

    combined_text = "\n\n".join(summaries)

    combine_prompt = PromptTemplate.from_template("""
    ข้อความสรุปย่อยต่อไปนี้มาจากเนื้อหาที่ยาว:

    {text}

    กรุณาสรุปรวมทั้งหมด เป็นภาษาไทย:
    """)

    combine_chain = LLMChain(llm=llm, prompt=combine_prompt, verbose=True)
    final_summary = combine_chain.run({"text": combined_text})
    print(f"\n🔖 Final Summary:\n{final_summary}")
    # ใช้ id จาก date_time_str เดิม
    id = date_time_str

    # กำหนด path ให้ไฟล์ summary ไปยัง frontend/public/chunk/{id}_summary.txt
    frontend_chunk_dir = os.path.join("..", "..", "frontend", "public", "chunk")
    os.makedirs(frontend_chunk_dir, exist_ok=True)

    summary_path = os.path.join(frontend_chunk_dir, f"{id}_summary.txt")

    # บันทึกไฟล์ summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    return {"elapsed_time": elapsed_time, "transcript": results, "final_summary": final_summary}

# transcribe("ขนตอนการทกทายทประชมอยางงาย #การพดในทสาธารณะ #เทคนคนำเสนอ #เทคนคการพด #การสอสาร.mp3")

