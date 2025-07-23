"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

type TranscriptItem = {
  time: string;
  text: string;
};

function formatTimeRange(timeStr: string) {
  // แปลง "090758_00_00_00-00_00_15" เป็น "00:00:00-00:00:15"
  const parts = timeStr.split("_");
  if (parts.length < 2) return timeStr;
  const rangePart = parts.slice(1).join("_");
  return rangePart.replace(/_/g, ":");
}

export default function RecordPage() {
  const params = useParams();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;

  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [summaryText, setSummaryText] = useState("");

  useEffect(() => {
    if (!id || typeof id !== "string") return;

    async function fetchData() {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/record/${id}/maincontent`);
        const data = await res.json();
        if (res.ok && data.transcript) {
          setTranscript(data.transcript);
        } else {
          console.error("No transcript or error:", data);
          setTranscript([]);
        }
      } catch (e) {
        console.error("Error loading transcript:", e);
        setTranscript([]);
      }

      try {
        const resSum = await fetch(`http://127.0.0.1:8000/api/record/${id}/maincontent`);
        const sumText = await resSum.text();
        if (resSum.ok) {
          setSummaryText(sumText);
        } else {
          setSummaryText("");
        }
      } catch (e) {
        console.error("Error loading summary:", e);
        setSummaryText("");
      }
    }

    fetchData();
  }, [id]);

  return (
    <div className="flex h-screen bg-black text-white">
      {/* Sidebar */}
      <aside className="w-80 border-r border-gray-700 p-6 overflow-y-auto">
        <h1 className="text-xl font-semibold mb-4">📹 {id}</h1>
        <h2 className="text-lg font-semibold mb-2">📝 Summary</h2>
        <ol className="list-decimal list-inside space-y-4 text-sm text-gray-300">
          <li>
            การประชุมมหาวิทยาลัย{" "}
            <span className="text-blue-400">00:00:51</span>
            <p className="mt-1 ml-4 text-gray-400 text-xs">
              กล่าวถึงการประชุมในวันพรุ่งนี้ เวลา 7:00 น. วันที่ 9 เมษายน 2562
            </p>
          </li>
          <li>
            ระเบียบวาระการประชุม{" "}
            <span className="text-blue-400">00:01:16</span>
            <p className="mt-1 ml-4 text-gray-400 text-xs">
              กล่าวถึงระเบียบวาระการประชุมทั้งหมด 5 ระเบียบวาระ ได้แก่ เรื่องที่ประธานแจ้งให้ที่ประชุมทราบ, เรื่องระเบียบวาระการประชุม, เรื่องเสนอเพื่อพิจารณา, และเรื่องแจ้งให้ทราบตามระเบียบ
            </p>
          </li>
        </ol>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        <div className="flex flex-1 overflow-hidden">
          {/* Transcript */}
          <section className="w-2/3 overflow-y-auto p-6 space-y-4 text-sm border-r border-gray-700">
            {transcript.map((entry, i) => (
              <p key={i} className="break-words">
                <span className="text-blue-400 font-mono mr-2">{formatTimeRange(entry.time)}</span>
                {entry.text}
              </p>
            ))}
          </section>

          {/* Chatbot */}
          <section className="w-1/3 flex flex-col border-l border-gray-700 p-4 overflow-y-auto">
            <h2 className="text-lg font-semibold mb-2">💬 Chat Assistant</h2>
            <div className="flex-1 bg-gray-900 rounded p-3 overflow-y-auto">
              <p className="text-sm text-gray-400">🤖 สวัสดี! มีอะไรให้ช่วยสรุปไหม?</p>
              <div className="mt-4">
                <h3 className="text-sm font-semibold text-gray-300 mb-2">คำถามแนะนำ:</h3>
                <ul className="space-y-2">
                  <li>
                    <button className="text-blue-400 hover:underline text-sm">
                      สรุปหัวข้อสำคัญของการประชุมคืออะไร?
                    </button>
                  </li>
                  <li>
                    <button className="text-blue-400 hover:underline text-sm">
                      มีการตัดสินใจอะไรบ้างในที่ประชุม?
                    </button>
                  </li>
                </ul>
              </div>
            </div>
            <form className="mt-3 flex gap-2">
              <input
                type="text"
                placeholder="ถามอะไรบางอย่าง..."
                className="flex-1 px-3 py-2 border border-gray-600 rounded-md text-sm bg-black text-white placeholder-gray-500"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
              >
                ส่ง
              </button>
            </form>
          </section>
        </div>

        {/* Audio Player */}
        <div className="border-t border-gray-700 p-4">
          <audio controls className="w-full bg-black">
            <source src={`/media/${id}.mp3`} type="audio/mp3" />
            Your browser does not support the audio element.
          </audio>
        </div>
      </main>
    </div>
  );
}
