import React from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "outline";
};

export function Button({
  variant = "default",
  className = "",
  children,
  ...props
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center rounded-xl text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 px-4 py-2";

  const styles =
    variant === "outline"
      ? "border border-gray-300 text-gray-700 hover:bg-gray-100"
      : "bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500";

  return (
    <button className={`${base} ${styles} ${className}`} {...props}>
      {children}
    </button>
  );
}
