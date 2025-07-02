export default function SearchBar() {
  return (
    <div className="w-full flex justify-end mb-6">
      <div className="w-full max-w-xs">
        <input
          type="text"
          placeholder="🔍 Search records..."
          className="w-full px-3 py-2 border rounded-md shadow-sm"
        />
      </div>
    </div>
  );
}