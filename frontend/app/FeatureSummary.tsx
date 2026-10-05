export type FeatureStats = {
  count: number;
  mean: number;
  std: number;
  min: number;
  p05: number;
  p25: number;
  median: number;
  p75: number;
  p95: number;
  max: number;
};

export type FeatureSummaryItem = {
  feature: string;
  label: string;
  importance: number;
  current_value: number | null;
  current_percentile: number | null;
  kind: "continuous" | "categorical";
  stats?: FeatureStats;
  density?: { x: number; density: number }[];
  current_label?: string | null;
  categories?: { value: number; label: string; share: number }[];
};

const fmt = (value: number | null | undefined, digits = 2) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";

const W = 100;
const H = 34;
const MID = H / 2;
const HALF = 12;

function Violin({ item }: { item: FeatureSummaryItem }) {
  const density = item.density ?? [];
  const stats = item.stats;
  if (!stats || density.length < 2) {
    return <p className="mt-2 text-xs text-slate-500">Not enough spread to plot a distribution.</p>;
  }
  const current = item.current_value;
  const lo = Math.min(density[0].x, current ?? Infinity);
  const hi = Math.max(density[density.length - 1].x, current ?? -Infinity);
  const sx = (x: number) => 3 + ((x - lo) / (hi - lo || 1)) * (W - 6);
  const upper = density.map((p) => `${sx(p.x)},${MID - p.density * HALF}`);
  const lower = [...density].reverse().map((p) => `${sx(p.x)},${MID + p.density * HALF}`);
  const ticks: [string, number][] = [
    ["p25", stats.p25],
    ["median", stats.median],
    ["p75", stats.p75],
  ];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="mt-2 h-20 w-full"
      role="img"
      aria-label={`${item.feature} distribution; today ${fmt(current)} is at the ${fmt((item.current_percentile ?? 0) * 100, 0)}th percentile`}
    >
      <polygon points={[...upper, ...lower].join(" ")} className="fill-slate-400" fillOpacity="0.45" />
      {ticks.map(([name, value]) => (
        <line
          key={name}
          x1={sx(value)}
          x2={sx(value)}
          y1={MID - 8}
          y2={MID + 8}
          className={name === "median" ? "stroke-slate-800" : "stroke-slate-500"}
          strokeWidth={name === "median" ? 1.2 : 0.7}
          strokeDasharray={name === "median" ? undefined : "1.5 1"}
        >
          <title>{`${name}: ${fmt(value)}`}</title>
        </line>
      ))}
      {current !== null ? (
        <line x1={sx(current)} x2={sx(current)} y1="1" y2={H - 1} className="stroke-amber-500" strokeWidth="1.8">
          <title>{`Today: ${fmt(current)}`}</title>
        </line>
      ) : null}
    </svg>
  );
}

function CategoryBars({ item }: { item: FeatureSummaryItem }) {
  const categories = item.categories ?? [];
  return (
    <ul className="mt-2 space-y-1">
      {categories.map((category) => {
        const isToday = category.value === item.current_value;
        return (
          <li key={category.value} className="grid grid-cols-[3.5rem_1fr_2.5rem] items-center gap-2 text-xs">
            <span className={isToday ? "font-semibold text-amber-600" : "text-slate-500"}>{category.label}</span>
            <div className="h-2 rounded-sm bg-slate-100">
              <div
                className={`h-2 rounded-sm ${isToday ? "bg-amber-500" : "bg-slate-400"}`}
                style={{ width: `${category.share * 100}%` }}
              />
            </div>
            <span className="text-right tabular-nums text-slate-500">{(category.share * 100).toFixed(0)}%</span>
          </li>
        );
      })}
    </ul>
  );
}

function Card({ item }: { item: FeatureSummaryItem }) {
  const stats = item.stats;
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-mono text-sm font-semibold text-slate-900">{item.feature}</h3>
          <p className="text-xs text-slate-500">{item.label}</p>
        </div>
        <span
          className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] tabular-nums text-slate-600"
          title="XGBoost feature_importances_"
        >
          importance {fmt(item.importance, 3)}
        </span>
      </div>
      <p className="data-reading mt-3 text-2xl font-semibold text-slate-900">
        {item.kind === "categorical" ? (item.current_label ?? "n/a") : fmt(item.current_value)}
        <span className="ml-2 text-xs font-normal text-slate-500">
          at prediction
          {item.current_percentile === null ? "" : ` · ${fmt(item.current_percentile * 100, 0)}th percentile`}
        </span>
      </p>
      {item.kind === "categorical" ? <CategoryBars item={item} /> : <Violin item={item} />}
      {stats ? (
        <dl className="mt-1 grid grid-cols-5 gap-x-2 gap-y-1 text-center text-[11px] text-slate-500">
          {(
            [
              ["min", stats.min],
              ["p25", stats.p25],
              ["median", stats.median],
              ["p75", stats.p75],
              ["max", stats.max],
              ["mean", stats.mean],
              ["std", stats.std],
              ["p05", stats.p05],
              ["p95", stats.p95],
              ["n", stats.count],
            ] as [string, number][]
          ).map(([name, value]) => (
            <div key={name}>
              <dt>{name}</dt>
              <dd className="font-medium tabular-nums text-slate-800">{name === "n" ? value : fmt(value)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </article>
  );
}

export default function FeatureSummary({ items }: { items: FeatureSummaryItem[] }) {
  return (
    <section className="logbook-panel">
      <p className="text-sm text-slate-600">
        Every model feature, ordered by model feature importance. The amber line is the value used for this
        prediction; the violin shows the training-data distribution with dashed p25/p75 and solid median.
      </p>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {items.map((item) => (
          <Card key={item.feature} item={item} />
        ))}
      </div>
    </section>
  );
}
