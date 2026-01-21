"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  LineChart,
  Bot,
  Search,
  Settings,
  Activity,
  Wallet,
  PieChart,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Command Center", href: "/command", icon: Activity },
  { name: "Holdings", href: "/holdings", icon: PieChart },
  { name: "Agents", href: "/agents", icon: Bot },
  { name: "Trading", href: "/trading", icon: LineChart },
  { name: "Scanner", href: "/scanner", icon: Search },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-full w-64 flex-col bg-navy-950 border-r border-navy-800">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 px-6 border-b border-navy-800">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-orange-500 to-orange-600">
          <Wallet className="h-5 w-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white">TradeSense</h1>
          <p className="text-xs text-navy-400">AI Trading System</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                isActive
                  ? "bg-orange-500/10 text-orange-500 border border-orange-500/20"
                  : "text-navy-300 hover:bg-navy-800 hover:text-white"
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-navy-800 p-4">
        <div className="rounded-lg bg-navy-900 p-3">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-navy-400">Paper Trading Active</span>
          </div>
        </div>
      </div>
    </div>
  );
}
