"use client";

import { cn } from "@/lib/utils";

interface BorderBeamProps {
  className?: string;
  size?: number;
  duration?: number;
  borderWidth?: number;
  anchor?: number;
  colorFrom?: string;
  colorTo?: string;
  delay?: number;
}

export const BorderBeam = ({
  className,
  size = 200,
  duration = 15,
  anchor = 90,
  borderWidth = 1.5,
  colorFrom = "#f97316",
  colorTo = "#0f1f49",
  delay = 0,
}: BorderBeamProps) => {
  return (
    <div
      style={{
        "--size": size,
        "--duration": duration,
        "--anchor": anchor,
        "--border-width": borderWidth,
        "--color-from": colorFrom,
        "--color-to": colorTo,
        "--delay": `-${delay}s`,
      } as React.CSSProperties}
      className={cn(
        "pointer-events-none absolute inset-0 rounded-[inherit] [border:calc(var(--border-width)*1px)_solid_transparent]",
        "[background:padding-box,linear-gradient(to_left,var(--color-from),var(--color-to),transparent,transparent)]",
        "[background-size:calc(var(--size)*1px)_calc(var(--size)*1px)]",
        "[mask-composite:exclude] [mask:linear-gradient(#fff_0_0)_border-box,linear-gradient(#fff_0_0)]",
        "motion-safe:animate-[border-beam_var(--duration,15)s_linear_infinite]",
        className
      )}
    />
  );
};
