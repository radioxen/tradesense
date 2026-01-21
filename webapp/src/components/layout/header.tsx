"use client";

import { Bell, RefreshCw } from "lucide-react";
import { ShimmerButton } from "@/components/ui/shimmer-button";

interface HeaderProps {
  title: string;
  subtitle?: string;
}

export function Header({ title, subtitle }: HeaderProps) {
  return (
    <header className="flex h-16 items-center justify-between border-b border-navy-800 bg-navy-950/50 px-6 backdrop-blur-sm">
      <div>
        <h1 className="text-xl font-bold text-white">{title}</h1>
        {subtitle && <p className="text-sm text-navy-400">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3">
        <button className="relative flex h-10 w-10 items-center justify-center rounded-lg border border-navy-700 bg-navy-900 text-navy-400 transition-colors hover:border-orange-500/50 hover:text-orange-500">
          <Bell className="h-5 w-5" />
          <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-orange-500 text-[10px] font-bold text-white">
            3
          </span>
        </button>

        <ShimmerButton className="h-10 gap-2 text-sm">
          <RefreshCw className="h-4 w-4" />
          Sync Data
        </ShimmerButton>
      </div>
    </header>
  );
}
