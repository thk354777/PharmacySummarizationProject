import torch

device = 0 if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.get_device_name(0))

# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # สมมุติ text คือข้อความที่มีอยู่แล้ว
# text = "ขั้นตอนการทักทายที่ประชุมมีหลากหลายรูปแบบขึ้นอยู่กับว่าคุณไปพูดในงานใดและคุณไปพูดในฐานะอะไร ในวันนี้ค่ะสิ่งที่พิมพ์จะบอกคุณนะคะคือ ขั้นตอนการทักทายที่ประชุมอย่างง่ายอย่างเบสิกที่สุดที่ใคร ๆ ค่ะก็สามารถหยิบไปแล้วทักทายที่ประชุมได้เหมือนกันไม่ว่าจะเป็นการที่คุณนะคะพูดในที่สาธารณะเป็นการที่คุณพูดในที่ชุมชน ไปจนกระทั่งการพูดนำเสนอ คุณก็สามารถทักถ่ายที่ประชุมตามขั้นตอนนี้ได้ ขั้นตอนแรกคืออะไร ขั้นตอนแรกค่ะ คือการที่คุณเปิดฉากการพูดด้วยการกล่าวว่า สวัสดีครับ และสวัสดีค่ะ จากนั้นค่ะในขั้นตอนที่สองคือ ให้คุณบอกที่ประชุมนะฮะว่าคุณเป็นใคร นั่นคือบอกชื่อและบอกนามสกุลของคุณเป็นขั้นตอนที่สองจากนั้นค่ะให้นำผู้ฟังเท่าสู่ขั้นตอนที่สามนั่นคือบอกผู้ฟังว่าคุณมาทำอะไรในวันนี้นี่คือสามขั้นตอนการทักทายที่ประชุมอย่างง่ายซึ่งจะเป็นประโยชน์กับคุณเมื่อคุณต้องเปิดฉาการพูดเผ็ดสปีกสปาคเราขอเป็นกำลังใจให้ผู้ฝึกฝนการพูดในที่สาธารณะถูกทันค่ะ"

# # แปลงให้กลายเป็น Document
# doc = [Document(page_content=text)]

# # แยกข้อความ
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# split_docs = text_splitter.split_documents(doc)

# # เรียก chain สรุปผล
# from langchain.chains.summarize import load_summarize_chain
# chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
# summary = chain.run(split_docs)

# from openai import OpenAI

# client = OpenAI(
#     base_url="http://localhost:1234/v1",
#     api_key="lm-studio"  # default key สำหรับ LM Studio
# )


# response = client.chat.completions.create(
#     model="typhoon-7b",  # หรือชื่อจริงของโมเดลที่โหลดใน LM Studio
#     messages=[
#         {"role": "system", "content": "คุณคือผู้ช่วยที่ฉลาดและใจดี"},
#         {"role": "user", "content": "ประเทศไทยคืออะไร?"}
#     ],
#     temperature=0.7,
#     max_tokens=512
# )

# print(response.choices[0].message.content)


from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
# สร้าง chain สำหรับสรุปแบบ map_reduce

from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="llama-3.2-1b-instruct",  # ชื่อโมเดลที่เปิดไว้ใน LM Studio
    temperature=0.7,
    max_tokens=512,
    openai_api_base="http://localhost:1234/v1",  # ชี้ไปยัง LM Studio
    openai_api_key="lm-studio",  # LM Studio ใช้ key อะไรก็ได้ (mock key)
)

# chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)

# text = "something"
# docs = [Document(page_content=text)]
# print(docs)
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_docs = splitter.split_documents(docs)

# summary = chain.run(split_docs)
# print(summary)


# from langchain_core.prompts import PromptTemplate
# map_prompt = """
# สรุปข้อความด้านล่างเป็นภาษาไทย:
# "{text}"
# """
# map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

# combine_prompt = """
# เขียนสรุปสั้นๆ เป็นภาษาไทย 3 bullet point.
# โดย point แรกคือ หัวข้อเกี่ยวกับอะไร
# เป็นภาษาไทย
# ```{text}```

# """
# combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

# # Initialize text splitter
# text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=200)
# docs =text_splitter.split_documents(docs)

# # Setup and run summarization chain
# summary_chain = load_summarize_chain(
#     llm=llm,
#     chain_type='map_reduce',
#     map_prompt=map_prompt_template,
#     combine_prompt=combine_prompt_template,
#     #verbose=True
# )

# # Run summarization chain and print output
# output = summary_chain.run(docs)
# print(output)

# response = llm.invoke("สรุปข้อความนี้ให้สั้นลง: วันนี้ฉันไปเดินตลาดซื้อผลไม้และขนมไทยหลายอย่าง")
# print(response.content)




#Chatgpt
####################


import pandas as pd
from langchain.docstore.document import Document

# def has_too_many_spaces(text, threshold=0.15):
#     text = text.strip()
#     if len(text) == 0:
#         return True  # ข้อความว่าง ให้ตัดทิ้ง
#     space_ratio = text.count(" ") / len(text)
#     return space_ratio > threshold

# filtered_texts = []
# for item in results:
#     text = item['text'].strip()
#     if len(text) > 5 and not has_too_many_spaces(text):
#         filtered_texts.append(text)

df = pd.read_csv("chunk_transcript.csv")
all_text = "\n".join(df["text"].dropna().astype(str))
docs = [Document(page_content=all_text)]
print(docs)

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="llama-3.2-1b-instruct",  # ชื่อโมเดลที่เปิดไว้ใน LM Studio
    temperature=0.7,
    max_tokens=512,
    openai_api_base="http://localhost:1234/v1",  # ชี้ไปยัง LM Studio
    openai_api_key="lm-studio",  # LM Studio ใช้ key อะไรก็ได้ (mock key)
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# สร้าง prompt สำหรับ map
map_prompt = PromptTemplate.from_template("""
ช่วยสรุปข้อความถอดเสียงในที่ประชุมนี้ให้กระชับและชัดเจนเป็นภาษาไทย  
และถ้ามีคำที่เป็นภาษาอังกฤษในประโยค ให้ใช้คำภาษาอังกฤษนั้นแทน  
เช่น โปรเจค -> project, เดดไล -> deadline เป็นต้น:
"{text}"

สรุป:
""")

# เตรียม chain สำหรับ map
map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)

# docs = [Document(page_content=text)]
# print(docs)
splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# สรุปแต่ละ chunk แล้วเก็บไว้
summaries = []
for i, doc in enumerate(split_docs):
    summary = map_chain.run({"text": doc.page_content})
    summaries.append(summary)
    # จะ print และ save ทีละ chunk
    print(f"💡 Chunk {i+1} summary:\n{summary}\n")

# ต่อเป็น string เพื่อใช้ตอน reduce หรือ save
combined_text = "\n\n".join(summaries)

# ถ้าต้องการใช้ combine prompt:
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

combine_prompt = PromptTemplate.from_template("""
ข้อความสรุปย่อยต่อไปนี้มาจากเนื้อหาที่ยาว:

{text}

กรุณาสรุปรวมทั้งหมด เป็นภาษาไทย:
""")

combine_chain = LLMChain(llm=llm, prompt=combine_prompt, verbose=True)
final_summary = combine_chain.run({"text": combined_text})
print(f"\n🔖 Final Summary:\n{final_summary}")


###################