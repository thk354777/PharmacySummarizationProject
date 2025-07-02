import { Button } from "@/components/button";
import SearchBar from "@/components/SearchBar";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen bg-gray-100 text-gray-900">
      {/* Sidebar ซ้าย */}
      <aside className="w-60 bg-white border-r p-6 space-y-4 text-sm">
        <h1 className="text-xl font-semibold text-black mb-4">MeetMate</h1>
        <nav className="space-y-2">
          <div className="font-semibold text-blue-600">🏠 Home</div>
          <div>📁 All Records</div>
          <div>📅 Upcoming events</div>
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
            <Button className="w-full mt-4">Upload & transcribe</Button>
          </div>

          {/* Records Table */}
          <section>
            <h3 className="text-lg font-semibold mb-2">My Records</h3>
            <div className="overflow-auto bg-white rounded-xl border shadow-sm">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50 text-left">
                  <tr>
                    <th className="px-4 py-2">Name</th>
                    <th className="px-4 py-2">Duration</th>
                    <th className="px-4 py-2">Date Created</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t">
                    <td className="px-4 py-2">
                      <Link href="/records/videoplayback">
                        📹 videoplayback
                      </Link>
                    </td>
                    <td className="px-4 py-2">7min 54s</td>
                    <td className="px-4 py-2">06/16/2025 21:41</td>
                  </tr>
                  <tr className="border-t">
                    <td className="px-4 py-2">📘 Notta quick guide</td>
                    <td className="px-4 py-2">1min 21s</td>
                    <td className="px-4 py-2">06/16/2025 21:40</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
