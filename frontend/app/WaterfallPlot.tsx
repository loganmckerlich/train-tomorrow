"use client";

import { useEffect, useRef, useState } from "react";

type WaterfallRow = {
  feature: string;
  startProbability: number | null;
  endProbability: number | null;
  direction: "helping" | "hurting";
};

export default function WaterfallPlot({
  rows,
  baselineProbability,
  baselineLabel,
  finalProbability,
  finalLabel,
  probabilityDomain,
  ticks,
  height,
}: {
  rows: WaterfallRow[];
  baselineProbability: number;
  baselineLabel: string;
  finalProbability: number;
  finalLabel: string;
  probabilityDomain: [number, number];
  ticks: number[];
  height: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      setWidth(entry.contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const left = Math.min(32, width * 0.12);
  const right = width - 12;
  const top = 12;
  const axisY = height - 20;
  const x = (probability: number) =>
    left + ((probability - probabilityDomain[0]) / (probabilityDomain[1] - probabilityDomain[0])) * (right - left);
  const rowY = (index: number) => top + ((axisY - top) * (index + 1)) / (rows.length + 1);

  return (
    <div ref={containerRef} className="relative w-full" style={{ height }}>
      {width > 0 ? (
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="block h-full w-full"
          role="img"
          aria-label="Waterfall of cumulative train probability after each SHAP-IQ effect"
        >
          {ticks.map((percentage) => (
            <line
              key={`grid-${percentage}`}
              x1={x(percentage / 100)}
              x2={x(percentage / 100)}
              y1={top}
              y2={axisY}
              className="stroke-slate-200"
              strokeWidth="1"
            />
          ))}
          {rows.map((row, index) => {
            if (row.startProbability === null || row.endProbability === null) {
              return null;
            }
            const y = rowY(index);
            const startX = x(row.startProbability);
            const endX = x(row.endProbability);
            return (
              <g key={`${row.feature}-${index}`}>
                <line
                  x1={startX}
                  x2={endX}
                  y1={y}
                  y2={y}
                  className={row.direction === "helping" ? "stroke-emerald-500" : "stroke-rose-500"}
                  strokeWidth="6"
                  strokeLinecap="round"
                />
                <circle cx={startX} cy={y} r="2.5" className="fill-white stroke-slate-400" strokeWidth="1" />
                <circle cx={endX} cy={y} r="3" className="fill-slate-900" />
              </g>
            );
          })}
          <line
            x1={x(baselineProbability)}
            x2={x(baselineProbability)}
            y1={top}
            y2={axisY}
            className="stroke-slate-500"
            strokeDasharray="3 3"
            strokeWidth="1.5"
          >
            <title>{baselineLabel}</title>
          </line>
          <line
            x1={x(finalProbability)}
            x2={x(finalProbability)}
            y1={top}
            y2={axisY}
            className="stroke-amber-500"
            strokeDasharray="3 3"
            strokeWidth="1.5"
          >
            <title>{finalLabel}</title>
          </line>
          <line x1={left} x2={right} y1={axisY} y2={axisY} className="stroke-slate-500" strokeWidth="1" />
          {ticks.map((percentage) => (
            <g key={`axis-${percentage}`}>
              <line
                x1={x(percentage / 100)}
                x2={x(percentage / 100)}
                y1={axisY}
                y2={axisY + 4}
                className="stroke-slate-500"
                strokeWidth="1"
              />
              <text
                x={x(percentage / 100)}
                y={axisY + 16}
                textAnchor="middle"
                className="fill-slate-600"
                fontSize="11"
              >
                {percentage}
              </text>
            </g>
          ))}
        </svg>
      ) : null}
    </div>
  );
}
