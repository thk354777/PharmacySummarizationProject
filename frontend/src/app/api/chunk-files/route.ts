import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  const chunkDir = path.join(process.cwd(), "public", "chunk");

  try {
    const files = fs.readdirSync(chunkDir).filter(file => file.endsWith(".csv"));
    const stats = files.map(file => {
      const fullPath = path.join(chunkDir, file);
      const stat = fs.statSync(fullPath);
      return {
        name: file,
        size: stat.size,
        created: stat.ctime,
      };
    });

    return NextResponse.json(stats);
  } catch (err) {
    return NextResponse.json({ error: "Failed to read chunk dir" }, { status: 500 });
  }
}
