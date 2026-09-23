import { unstable_noStore as noStore } from "next/cache";

type ContinuousPoint = {
  feature_value: number;
  shap_value: number;
};

type ContinuousPlot = {
  kind: "continuous";
  current_value: number | null;
  current_shap: number | null;
  points: ContinuousPoint[];
};

type CategoricalPoint = {
  value: number;
  label: string;
  mean_shap: number;
};

type CategoricalPlot = {
  kind: "categorical";
  current_value: number | null;
  current_label: string | null;
  categories: CategoricalPoint[];
};

type Contributor = {
  feature: string;
  signed_contribution: number;
  direction: "helping" | "hurting";
  phrase: string;
  plot?: ContinuousPlot | CategoricalPlot;
};

type PredictionPayload = {
  date: string;
  will_train: boolean;
  probability: number;
  predicted_effort: number | null;
  top_contributors: Contributor[];
  blurb: string;
};

const DEFAULT_URL =
  "https://raw.githubusercontent.com/loganmckerlich/train-tomorrow/master/data/latest.json";

async function getPrediction(): Promise<PredictionPayload | null> {
  noStore();
  const response = await fetch(process.env.NEXT_PUBLIC_PREDICTION_URL ?? DEFAULT_URL, {
    cache: "no-store",
  });

  if (!response.ok) {
    return null;
  }

  return (await response.json()) as PredictionPayload;
}

function contributorBarWidth(score: number): string {
  return `${Math.max(10, Math.min(100, Math.round(Math.abs(score) * 100)))}%`;
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatSigned(value: number | null, digits = 2): string {
  return isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "n/a";
}

function paddedExtent(values: number[], fallbackPadding = 0.5): [number, number] | null {
  if (values.length === 0) {
    return null;
  }

  let min = values[0];
  let max = values[0];
  for (const value of values) {
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }

  if (min === max) {
    return [min - fallbackPadding, max + fallbackPadding];
  }

  const padding = (max - min) * 0.08;
  return [min - padding, max + padding];
}

function scale(value: number, domain: [number, number], range: [number, number]): number {
  const [domainMin, domainMax] = domain;
  const [rangeMin, rangeMax] = range;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function ContinuousFeaturePlot({ feature, plot }: { feature: string; plot: ContinuousPlot }) {
  const historicalPoints = plot.points.filter(
    (point) => Number.isFinite(point.feature_value) && Number.isFinite(point.shap_value),
  );
  const currentVisible = isFiniteNumber(plot.current_value) && isFiniteNumber(plot.current_shap);
  const currentValue = currentVisible ? plot.current_value : null;
  const currentShap = currentVisible ? plot.current_shap : null;
  const xExtent = paddedExtent([
    ...historicalPoints.map((point) => point.feature_value),
    ...(currentValue === null ? [] : [currentValue]),
  ]);
  const yExtent = paddedExtent([
    ...historicalPoints.map((point) => point.shap_value),
    ...(currentShap === null ? [0] : [currentShap, 0]),
  ]);

  if (!xExtent || !yExtent) {
    return null;
  }

  const zeroY = scale(0, yExtent, [90, 10]);
  const historicalCount = historicalPoints.length;
  const summary = `Historical days: ${historicalCount}. Feature values ranged from ${xExtent[0].toFixed(1)} to ${xExtent[1].toFixed(1)}. SHAP contributions ranged from ${yExtent[0].toFixed(2)} to ${yExtent[1].toFixed(2)}.`;
  const idBase = `${feature}-continuous-plot`;

  return (
    <figure
      className="mt-3"
      role="img"
      aria-labelledby={`${idBase}-title`}
      aria-describedby={`${idBase}-today ${idBase}-summary`}
    >
      <figcaption id={`${idBase}-title`} className="text-xs text-slate-500">
        SHAP contribution vs. feature value. Above 0 pushes toward training; below 0 pushes away.
      </figcaption>
      <p id={`${idBase}-today`} className="mt-1 text-xs text-slate-500">
        Today: value {currentValue === null ? "n/a" : currentValue.toFixed(1)}, SHAP{" "}
        {formatSigned(currentShap)}.
      </p>
      <p id={`${idBase}-summary`} className="mt-1 text-xs text-slate-500">
        {summary}
      </p>
      <div className="mt-3 rounded-lg border border-slate-200 bg-white p-2">
        <svg viewBox="0 0 100 100" className="h-36 w-full" aria-hidden="true">
          <line
            x1="8"
            x2="96"
            y1={zeroY}
            y2={zeroY}
            className="stroke-slate-400"
            strokeDasharray="4 3"
            strokeWidth="1"
          />
          {historicalPoints.map((point, index) => (
            <circle
              key={`${point.feature_value}-${point.shap_value}-${index}`}
              cx={scale(point.feature_value, xExtent, [8, 96])}
              cy={scale(point.shap_value, yExtent, [90, 10])}
              r="1.9"
              className="fill-slate-500"
              fillOpacity="0.45"
            />
          ))}
          {currentValue !== null && currentShap !== null ? (
            <circle
              cx={scale(currentValue, xExtent, [8, 96])}
              cy={scale(currentShap, yExtent, [90, 10])}
              r="3.4"
              className="fill-amber-400 stroke-slate-900"
              strokeWidth="1.5"
            />
          ) : null}
        </svg>
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500">
        <span>{xExtent[0].toFixed(1)}</span>
        <span>feature value</span>
        <span>{xExtent[1].toFixed(1)}</span>
      </div>
      <p className="mt-1 text-center text-[11px] text-slate-500">
        SHAP range {yExtent[0].toFixed(2)} to {yExtent[1].toFixed(2)}
      </p>
    </figure>
  );
}

function CategoricalFeaturePlot({ feature, plot }: { feature: string; plot: CategoricalPlot }) {
  if (plot.categories.length === 0) {
    return null;
  }

  const yExtent = paddedExtent([...plot.categories.map((category) => category.mean_shap), 0]);
  if (!yExtent) {
    return null;
  }

  const zeroY = scale(0, yExtent, [90, 10]);
  const barWidth = 84 / plot.categories.length;
  const currentLabel = plot.current_label ?? "n/a";
  const idBase = `${feature}-categorical-plot`;

  return (
    <figure
      className="mt-3"
      role="img"
      aria-labelledby={`${idBase}-title`}
      aria-describedby={`${idBase}-today ${idBase}-values`}
    >
      <figcaption id={`${idBase}-title`} className="text-xs text-slate-500">
        Mean SHAP contribution by category. Above 0 pushes toward training; below 0 pushes away.
      </figcaption>
      <p id={`${idBase}-today`} className="mt-1 text-xs text-slate-500">
        Today&apos;s category: {currentLabel}.
      </p>
      <ul id={`${idBase}-values`} className="mt-1 space-y-1 text-xs text-slate-500">
        {plot.categories.map((category) => (
          <li key={`summary-${category.value}`}>
            {category.label}: {formatSigned(category.mean_shap)}
            {category.value === plot.current_value ? " (today)" : ""}
          </li>
        ))}
      </ul>
      <div className="mt-3 rounded-lg border border-slate-200 bg-white p-2">
        <svg viewBox="0 0 100 100" className="h-36 w-full" aria-hidden="true">
          <line
            x1="8"
            x2="96"
            y1={zeroY}
            y2={zeroY}
            className="stroke-slate-400"
            strokeDasharray="4 3"
            strokeWidth="1"
          />
          {plot.categories.map((category, index) => {
            const x = 8 + index * barWidth + barWidth * 0.15;
            const y = scale(category.mean_shap, yExtent, [90, 10]);
            const isToday = category.value === plot.current_value;
            return (
              <rect
                key={`${category.value}-${category.mean_shap}`}
                x={x}
                y={Math.min(y, zeroY)}
                width={Math.max(barWidth * 0.7, 2)}
                height={Math.max(Math.abs(zeroY - y), 1)}
                rx="1.5"
                className={isToday ? "fill-amber-400 stroke-slate-900" : "fill-slate-500"}
                fillOpacity={isToday ? 1 : 0.55}
                strokeWidth={isToday ? "1.2" : "0"}
              />
            );
          })}
        </svg>
      </div>
    </figure>
  );
}

function FeaturePlot({ contributor }: { contributor: Contributor }) {
  if (!contributor.plot) {
    return null;
  }

  return (
    <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
      <summary className="cursor-pointer text-xs font-medium text-slate-700">Show model effect plot</summary>
      {contributor.plot.kind === "continuous" ? (
        <ContinuousFeaturePlot feature={contributor.feature} plot={contributor.plot} />
      ) : (
        <CategoricalFeaturePlot feature={contributor.feature} plot={contributor.plot} />
      )}
    </details>
  );
}

export default async function Home() {
  const prediction = await getPrediction();

  if (!prediction) {
    return (
      <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col justify-center px-6 py-16">
        <h1 className="text-3xl font-bold text-slate-900">train tomorrow</h1>
        <p className="mt-4 text-slate-600">Couldn’t load today’s prediction payload.</p>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-8 px-6 py-14">
      <header>
        <p className="text-sm font-medium uppercase tracking-wide text-slate-500">{prediction.date}</p>
        <h1 className="mt-2 text-4xl font-bold text-slate-900">train tomorrow</h1>
      </header>

      <section className="rounded-2xl bg-slate-900 p-6 text-slate-50 shadow-sm">
        <p className="text-lg leading-relaxed">{prediction.blurb}</p>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="text-xs uppercase tracking-wide text-slate-500">Train probability</p>
          <p className="mt-2 text-3xl font-semibold text-slate-900">
            {Math.round(prediction.probability * 100)}%
          </p>
          <p className="mt-1 text-sm text-slate-600">
            {prediction.will_train ? "Likely training day" : "Likely recovery day"}
          </p>
        </article>

        <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="text-xs uppercase tracking-wide text-slate-500">Predicted effort</p>
          <p className="mt-2 text-3xl font-semibold text-slate-900">
            {prediction.predicted_effort === null ? "—" : prediction.predicted_effort.toFixed(1)}
          </p>
          <p className="mt-1 text-sm text-slate-600">relative-effort points</p>
        </article>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Top contributors</h2>
        <ul className="mt-4 space-y-4">
          {prediction.top_contributors.map((item) => (
            <li key={item.feature}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="font-medium text-slate-800">{item.phrase}</span>
                <span
                  className={
                    item.direction === "helping"
                      ? "font-semibold text-emerald-600"
                      : "font-semibold text-rose-600"
                  }
                >
                  {item.direction}
                </span>
              </div>
              <div className="h-2 rounded-full bg-slate-200">
                <div
                  className={`h-2 rounded-full ${
                    item.direction === "helping" ? "bg-emerald-500" : "bg-rose-500"
                  }`}
                  style={{ width: contributorBarWidth(item.signed_contribution) }}
                />
              </div>
              <FeaturePlot contributor={item} />
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
