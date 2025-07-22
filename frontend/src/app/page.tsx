"use client"
import { Button } from "@/components/button";
import SearchBar from "@/components/SearchBar";
import Link from "next/link";
import { useState, useRef } from "react";
import { useEffect } from "react";

export default function Home() {
  const [isOpen, setIsOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const inputFileRef = useRef<HTMLInputElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const openFilePicker = () => {
    inputFileRef.current?.click();
  };

  const openModal = () => setIsOpen(true);
  const closeModal = () => {
    setIsOpen(false);
    setFile(null);
    setResult(null);
    setLoading(false);
  };
  const [records, setRecords] = useState<
    { name: string; created: string; size: number }[]
  >([]);
  useEffect(() => {
    const fetchRecords = async () => {
      try {
        const res = await fetch("/api/chunk-files");
        const data = await res.json();
        setRecords(data);
      } catch (e) {
        console.error("❌ load failed", e);
      }
    };
    fetchRecords();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      console.log("📁 เลือกไฟล์:", selected.name);
    } else {
      setFile(null);
      console.log("❌ ไม่มีไฟล์ถูกเลือก");
    }
  };


  const handleUpload = async () => {
    console.log("Upload clicked, file =", file);
    if (!file) {
      alert("Please select a file first");
      return;
    }

    const formData = new FormData();
    formData.append("audio", file);

    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/transcribe", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
    
  };

  const openModal1 = async () => {
    if (!file) return alert("Please select a file first");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload-audio", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
    } catch (err) {
      console.error(err);
      alert("Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-gray-100 text-gray-900">
      {/* Sidebar ซ้าย */}
      <aside className="w-60 bg-white border-r p-6 space-y-4 text-sm">
        <h1 className="text-xl font-semibold text-black mb-4">MeetMate</h1>
        <nav className="space-y-2">
          <div className="font-semibold text-blue-600">🏠 Home</div>
          <div>📁 All Records</div>
          <div>⭐ Starred</div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8">
        <div className="flex flex-col">
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-2xl font-bold">Home</h2>
            </div>

            {/* Search bar ด้านขวา */}
            <SearchBar />
          </div>

          {/* Cards สำหรับ action */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <Button className="w-full mt-4">🎙 Instant record</Button>
            <Button 
              onClick={openModal}
              className="w-full mt-4">
                Upload & transcribe
            </Button>
            <div className="p-6 font-sans">
              <h2 className="text-xl mb-4">🎵 Upload Audio File</h2>

              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="mb-4"
              />

              <Button onClick={openModal1} className="w-full mt-4">
                {loading ? "Uploading..." : "Upload & transcribe 1"}
              </Button>
            </div>
            {/* <Button 
              onClick={openModal1}
              className="w-full mt-4">
                Upload & transcribe 1
            </Button> */}
            {result && (
              <div className="mt-4 p-3 border rounded bg-gray-100 max-h-48 overflow-auto text-sm text-gray-800">
                <h3 className="font-semibold mb-2">Transcript Result:</h3>
                <pre className="whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>
              </div>
            )}
          </div>
          

          {/* Records Table */}
          <section>
            <h3 className="text-lg font-semibold mb-2">My Records</h3>
            <div className="overflow-auto bg-white rounded-xl border shadow-sm">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50 text-left">
                  <tr>
                    <th className="px-4 py-2">Name</th>
                    <th className="px-4 py-2">Size</th>
                    <th className="px-4 py-2">Date Created</th>
                  </tr>
                </thead>
                <tbody>
                  {/* ✅ แถวคงที่ */}
                  <tr className="border-t">
                    <td className="px-4 py-2">
                      <Link href="/records/videoplayback">📹 videoplayback</Link>
                    </td>
                    <td className="px-4 py-2">7min 54s</td>
                    <td className="px-4 py-2">06/16/2025 21:41</td>
                  </tr>

                  {/* ✅ แถวจากไฟล์ใน public/chunk */}
                  {records.map((record, index) => (
                    <tr key={index} className="border-t hover:bg-gray-50">
                      <td className="px-4 py-2">
                        <Link href={`/records/${record.name.replace(".csv", "")}`}>
                          📄 {record.name.replace(".csv", "")}
                        </Link>
                      </td>
                      <td className="px-4 py-2">
                        {(record.size / 1024).toFixed(1)} KB
                      </td>
                      <td className="px-4 py-2">
                        {new Date(record.created).toLocaleString("th-TH")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

        </div>
        {isOpen && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-white rounded-lg shadow-lg w-96 p-6 relative text-gray-800">
              <button
                onClick={closeModal}
                className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
                aria-label="Close modal"
              >
                ✕
              </button>

              <h2 className="text-xl font-semibold mb-4">Upload Audio File</h2>

              {/* ปุ่มกดเพื่อเปิด File Picker */}
              <Button onClick={openFilePicker} className="mb-4 w-full">
                Choose Audio File
              </Button>

              {/* input file ซ่อน */}
              <input
                type="file"
                accept="audio/*"
                ref={inputFileRef}
                onChange={handleFileChange}
                style={{ display: "none" }}
              />

              {/* แสดงชื่อไฟล์ที่เลือก */}
              {file && <p className="mb-4">Selected file: {file.name}</p>}

              {/* ปุ่ม Upload */}
              <Button
                onClick={handleUpload}
                disabled={!file || loading}
                className="w-full"
              >
                {loading ? "Uploading..." : "Upload & Transcribe"}
              </Button>

              {/* แสดงผลลัพธ์ */}
              {result && (
                <div className="mt-4 p-3 border rounded bg-green-100 text-green-800 text-center font-semibold">
                  Finish
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
