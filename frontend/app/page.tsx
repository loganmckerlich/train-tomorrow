import { unstable_noStore as noStore } from "next/cache";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import WaterfallPlot from "./WaterfallPlot";

const blurbMarkdownComponents: Components = {
  p: ({ children }) => <p className="mt-3 leading-relaxed first:mt-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-amber-300">{children}</strong>,
  ul: ({ children }) => <ul className="mt-3 list-disc space-y-1 pl-5">{children}</ul>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
};

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
  kind?: "feature" | "interaction";
  features?: [string, string];
  signed_contribution: number;
  direction: "helping" | "hurting";
  phrase: string;
  plot?: ContinuousPlot | CategoricalPlot;
};

type CalibrationBucket = {
  lower_bound: number;
  upper_bound: number;
  predicted_rate: number | null;
  actual_rate: number | null;
  sample_size: number;
};

type CalibrationSummary = {
  total_samples: number;
  buckets: CalibrationBucket[];
};

type PredictionPayload = {
  date: string;
  generated_at?: string;
  will_train: boolean;
  probability: number;
  predicted_effort: number | null;
  top_contributors: Contributor[];
  baseline_log_odds?: number;
  baseline_probability?: number;
  other_contribution?: number;
  calibration?: CalibrationSummary;
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

function contributorBarWidth(score: number, maxMagnitude: number): string {
  const ratio = maxMagnitude <= 0 ? 0 : Math.abs(score) / maxMagnitude;
  return `${Math.max(10, Math.min(100, Math.round(ratio * 100)))}%`;
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatSigned(value: number | null, digits = 2): string {
  return isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "n/a";
}

function formatPercent(value: number | null, digits = 0): string {
  return isFiniteNumber(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
}

function clampProbability(probability: number): number {
  return Math.max(1e-6, Math.min(1 - 1e-6, probability));
}

function logit(probability: number): number {
  const bounded = clampProbability(probability);
  return Math.log(bounded / (1 - bounded));
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function formatLogOdds(value: number | null): string {
  return isFiniteNumber(value) ? `${formatSigned(value, 3)} log odds` : "n/a";
}

function formatPointChange(value: number | null): string {
  if (!isFiniteNumber(value)) {
    return "n/a";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)} points`;
}

function formatGeneratedAt(isoTimestamp: string | undefined): string | null {
  if (!isoTimestamp) {
    return null;
  }
  const parsed = new Date(isoTimestamp);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Los_Angeles",
    timeZoneName: "short",
  });
}

function paddedExtent(
  values: number[],
  fallbackPadding = 0.5,
  paddingFraction = 0.08,
): [number, number] | null {
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

  const padding = (max - min) * paddingFraction;
  return [min - padding, max + padding];
}

function scale(value: number, domain: [number, number], range: [number, number]): number {
  const [domainMin, domainMax] = domain;
  const [rangeMin, rangeMax] = range;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function ContinuousFeaturePlot({
  feature,
  plot,
}: {
  feature: string;
  plot: ContinuousPlot;
}) {
  const historicalPoints = plot.points.filter(
    (point) => Number.isFinite(point.feature_value) && Number.isFinite(point.shap_value),
  );
  const currentVisible = isFiniteNumber(plot.current_value) && isFiniteNumber(plot.current_shap);
  const currentValue = currentVisible ? plot.current_value : null;
  const currentShap = currentVisible ? plot.current_shap : null;
  const currentLogOdds = currentShap;
  const xExtent = paddedExtent([
    ...historicalPoints.map((point) => point.feature_value),
    ...(currentValue === null ? [] : [currentValue]),
  ]);
  const historicalEffects = historicalPoints.map((point) => point.shap_value);
  const yExtent = paddedExtent([
    ...historicalEffects,
    ...(currentLogOdds === null ? [0] : [currentLogOdds, 0]),
  ]);

  if (!xExtent || !yExtent) {
    return null;
  }

  const zeroY = scale(0, yExtent, [90, 10]);
  const historicalCount = historicalPoints.length;
  const summary = `Historical days: ${historicalCount}. Feature values ranged from ${xExtent[0].toFixed(1)} to ${xExtent[1].toFixed(1)}. SHAP-IQ log-odds ranged from ${formatLogOdds(yExtent[0])} to ${formatLogOdds(yExtent[1])}.`;
  const idBase = `${feature}-continuous-plot`;

  return (
    <figure
      className="mt-3"
      role="img"
      aria-labelledby={`${idBase}-title`}
      aria-describedby={`${idBase}-today ${idBase}-summary`}
    >
      <figcaption id={`${idBase}-title`} className="text-xs text-slate-500">
        SHAP-IQ log odds vs. feature value. Above 0 pushes toward training; below 0 pushes away.
      </figcaption>
      <p id={`${idBase}-today`} className="mt-1 text-xs text-slate-500">
        Today: value {currentValue === null ? "n/a" : currentValue.toFixed(1)},{" "}
        <span title={currentShap === null ? undefined : `Raw SHAP-IQ ${formatSigned(currentShap, 4)} log odds`}>
          {formatLogOdds(currentLogOdds)}
        </span>
        .
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
          {historicalPoints.map((point, index) => {
            const effectLogOdds = point.shap_value;
            return (
              <circle
                key={`${point.feature_value}-${point.shap_value}-${index}`}
                cx={scale(point.feature_value, xExtent, [8, 96])}
                cy={scale(effectLogOdds, yExtent, [90, 10])}
                r="1.9"
                className="fill-slate-500"
                fillOpacity="0.45"
              >
                <title>
                  {`SHAP-IQ ${formatLogOdds(effectLogOdds)}`}
                </title>
              </circle>
            );
          })}
          {currentValue !== null && currentShap !== null && currentLogOdds !== null ? (
            <circle
              cx={scale(currentValue, xExtent, [8, 96])}
              cy={scale(currentLogOdds, yExtent, [90, 10])}
              r="3.4"
              className="fill-amber-400 stroke-slate-900"
              strokeWidth="1.5"
            >
              <title>{`Today: SHAP-IQ ${formatLogOdds(currentLogOdds)}`}</title>
            </circle>
          ) : null}
        </svg>
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-slate-500">
        <span>{xExtent[0].toFixed(1)}</span>
        <span>feature value</span>
        <span>{xExtent[1].toFixed(1)}</span>
      </div>
      <p className="mt-1 text-center text-[11px] text-slate-500">
        SHAP-IQ log-odds range {formatLogOdds(yExtent[0])} to {formatLogOdds(yExtent[1])}
      </p>
    </figure>
  );
}

function CategoricalFeaturePlot({
  feature,
  plot,
}: {
  feature: string;
  plot: CategoricalPlot;
}) {
  if (plot.categories.length === 0) {
    return null;
  }

  const categoryEffects = plot.categories.map((category) => category.mean_shap);
  const yExtent = paddedExtent([...categoryEffects, 0]);
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
        Mean SHAP-IQ log odds by category. Above 0 pushes toward training; below 0 pushes away.
      </figcaption>
      <p id={`${idBase}-today`} className="mt-1 text-xs text-slate-500">
        Today&apos;s category: {currentLabel}.
      </p>
      <ul id={`${idBase}-values`} className="mt-1 space-y-1 text-xs text-slate-500">
        {plot.categories.map((category) => {
          const effectLogOdds = category.mean_shap;
          return (
            <li key={`summary-${category.value}`}>
              <span title={`SHAP-IQ ${formatSigned(category.mean_shap, 4)} log odds`}>
                {category.label}: {formatLogOdds(effectLogOdds)}
              </span>
              {category.value === plot.current_value ? " (today)" : ""}
            </li>
          );
        })}
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
            const effectLogOdds = category.mean_shap;
            const y = scale(effectLogOdds, yExtent, [90, 10]);
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
              >
                <title>
                  {`${category.label}: SHAP-IQ ${formatLogOdds(effectLogOdds)}`}
                </title>
              </rect>
            );
          })}
        </svg>
      </div>
      <p className="mt-1 text-center text-[11px] text-slate-500">
        SHAP-IQ log-odds range {formatLogOdds(yExtent[0])} to {formatLogOdds(yExtent[1])}
      </p>
    </figure>
  );
}

function FeaturePlot({ contributor }: { contributor: Contributor }) {
  if (contributor.kind === "interaction" || !contributor.plot) {
    return null;
  }

  return (
    <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
      <summary className="cursor-pointer text-xs font-medium text-slate-700">
        Show model effect plot
      </summary>
      {contributor.plot.kind === "continuous" ? (
        <ContinuousFeaturePlot
          feature={contributor.feature}
          plot={contributor.plot}
        />
      ) : (
        <CategoricalFeaturePlot
          feature={contributor.feature}
          plot={contributor.plot}
        />
      )}
    </details>
  );
}

function ContributorSummary({
  topContributors,
  baselineLogOdds,
  otherContribution,
  finalProbability,
}: {
  topContributors: Contributor[];
  baselineLogOdds?: number;
  otherContribution?: number;
  finalProbability: number;
}) {
  const steps: Contributor[] = [...topContributors];
  if (otherContribution !== undefined) {
    const topContributionTotal = topContributors.reduce(
      (total, contributor) => total + contributor.signed_contribution,
      baselineLogOdds ?? 0,
    );
    const remainderContribution =
      baselineLogOdds === undefined
        ? otherContribution
        : logit(finalProbability) - topContributionTotal;
    steps.push({
      feature: "other_features",
      signed_contribution: remainderContribution,
      direction: remainderContribution >= 0 ? "helping" : "hurting",
      phrase: "all other features",
    });
  }
  let runningLogOdds = baselineLogOdds;
  const rows = steps.map((step) => {
    if (runningLogOdds === undefined) {
      return { ...step, startProbability: null, endProbability: null, pointChange: null };
    }
    const startProbability = sigmoid(runningLogOdds);
    runningLogOdds += step.signed_contribution;
    const endProbability = sigmoid(runningLogOdds);
    return {
      ...step,
      startProbability,
      endProbability,
      pointChange: (endProbability - startProbability) * 100,
    };
  });
  const maxContributionMagnitude = Math.max(
    ...rows.map((row) => Math.abs(row.pointChange ?? 0)),
    1,
  );
  const baselineProbability =
    baselineLogOdds === undefined ? undefined : sigmoid(baselineLogOdds);
  const chartProbabilityExtent = paddedExtent(
    [
      baselineProbability ?? 0,
      finalProbability,
      ...rows.flatMap((row) => [row.startProbability, row.endProbability]),
    ].filter(isFiniteNumber),
    0.01,
    0.02,
  ) ?? [0, 1];
  const chartProbabilityDomain: [number, number] = [
    Math.max(0, Math.floor(chartProbabilityExtent[0] * 100) / 100),
    Math.min(1, Math.ceil(chartProbabilityExtent[1] * 100) / 100),
  ];
  const firstTick = Math.round(chartProbabilityDomain[0] * 100);
  const lastTick = Math.round(chartProbabilityDomain[1] * 100);
  const tickStep = Math.max(1, Math.ceil((lastTick - firstTick) / 12));
  const chartTicks = Array.from(
    { length: Math.floor((lastTick - firstTick) / tickStep) + 1 },
    (_, index) => firstTick + index * tickStep,
  )
    .concat(lastTick)
    .filter((tick, index, ticks) => ticks.indexOf(tick) === index)
    .map((percentage) => ({
      percentage,
      x: scale(percentage / 100, chartProbabilityDomain, [8, 96]),
    }));
  const chartHeightPx = Math.max(180, rows.length * 28);

  return (
    <>
      {baselineProbability === undefined ? null : (
        <p className="mt-1 text-sm text-slate-600">
          Starting from the model&apos;s average day ({formatPercent(baselineProbability, 1)}), each row applies the next SHAP-IQ effect
          {otherContribution === undefined ? "." : ` to reach ${formatPercent(finalProbability, 1)}.`}
        </p>
      )}
      <p className="mt-1 text-xs text-slate-500">
        Feature rows show main effects after separating pairwise interactions; interaction rows show the pair effect.
      </p>
      {baselineProbability === undefined ? null : (
        <figure className="mt-4 rounded-lg border border-slate-200 bg-white p-3">
          <figcaption className="sr-only">Waterfall showing cumulative train probability after each SHAP-IQ effect</figcaption>
          <WaterfallPlot
            rows={rows.map((row) => ({
              feature: row.feature,
              startProbability: row.startProbability,
              endProbability: row.endProbability,
              direction: row.direction,
            }))}
            baselineProbability={baselineProbability}
            baselineLabel={`Model average: ${formatPercent(baselineProbability, 1)}`}
            finalProbability={finalProbability}
            finalLabel={`Final prediction: ${formatPercent(finalProbability, 1)}`}
            probabilityDomain={chartProbabilityDomain}
            ticks={chartTicks.map((tick) => tick.percentage)}
            height={chartHeightPx}
          />
          <p className="mt-1 text-center text-[11px] text-slate-500">train probability (%)</p>
          <div className="mt-1 flex justify-center gap-4 text-[11px] text-slate-500">
            <span><span className="mr-1 inline-block w-3 border-t border-dotted border-slate-500 align-middle" />Model average</span>
            <span><span className="mr-1 inline-block w-3 border-t border-dotted border-amber-500 align-middle" />Final prediction</span>
          </div>
        </figure>
      )}
      <ul className="mt-4 space-y-4">
        {rows.map((row) => {
          const pointChange = row.pointChange;
          return (
            <li key={row.feature}>
              <div className="mb-1 flex items-center justify-between gap-4 text-sm">
                <span className="font-medium text-slate-800">{row.phrase}</span>
                <div className="text-right">
                  <span
                    className={
                      row.direction === "helping"
                        ? "font-semibold text-emerald-600"
                        : "font-semibold text-rose-600"
                    }
                    title={`SHAP-IQ ${formatSigned(row.signed_contribution, 3)} log odds`}
                  >
                    {formatLogOdds(row.signed_contribution)} ({formatPointChange(pointChange)})
                    {row.endProbability === null ? "" : ` → ${formatPercent(row.endProbability, 1)}`}
                  </span>
                  <p className="text-xs text-slate-500">{row.direction}</p>
                </div>
              </div>
              <div className="h-2 rounded-full bg-slate-200">
                <div
                  className={`h-2 rounded-full ${
                    row.direction === "helping" ? "bg-emerald-500" : "bg-rose-500"
                  }`}
                  style={{ width: contributorBarWidth(pointChange ?? 0, maxContributionMagnitude) }}
                  title={`Probability change ${formatPointChange(pointChange)}`}
                />
              </div>
              <FeaturePlot contributor={row} />
            </li>
          );
        })}
      </ul>
    </>
  );
}

function CalibrationPlot({ calibration }: { calibration: CalibrationSummary }) {
  const populatedBuckets = calibration.buckets.filter(
    (bucket) =>
      bucket.sample_size > 0 &&
      isFiniteNumber(bucket.predicted_rate) &&
      isFiniteNumber(bucket.actual_rate),
  );

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Calibration / track record</h2>
          <p className="mt-1 text-sm text-slate-600">
            When the model says X% likely, how often did I actually train?
          </p>
        </div>
        <p className="text-sm text-slate-500">{calibration.total_samples} resolved historical predictions</p>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
        <svg viewBox="0 0 100 100" className="h-48 w-full" aria-hidden="true">
          <line x1="8" y1="92" x2="96" y2="92" className="stroke-slate-300" strokeWidth="1" />
          <line x1="8" y1="92" x2="8" y2="10" className="stroke-slate-300" strokeWidth="1" />
          <line x1="8" y1="92" x2="96" y2="10" className="stroke-slate-400" strokeDasharray="4 3" strokeWidth="1" />
          {populatedBuckets.map((bucket) => {
            const predictedRate = bucket.predicted_rate as number;
            const actualRate = bucket.actual_rate as number;
            return (
              <circle
                key={`${bucket.lower_bound}-${bucket.upper_bound}`}
                cx={scale(predictedRate, [0, 1], [8, 96])}
                cy={scale(actualRate, [0, 1], [92, 10])}
                r={Math.min(4.5, 2 + bucket.sample_size * 0.35)}
                className="fill-amber-400 stroke-slate-900"
                strokeWidth="1"
              >
                <title>
                  {`${formatPercent(bucket.lower_bound)}–${formatPercent(bucket.upper_bound)} bucket: predicted ${formatPercent(predictedRate)}, trained ${formatPercent(actualRate)}, n=${bucket.sample_size}`}
                </title>
              </circle>
            );
          })}
        </svg>
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
          <span>0% predicted</span>
          <span>predicted train rate</span>
          <span>100% predicted</span>
        </div>
        <p className="mt-1 text-center text-[11px] text-slate-500">actual train rate climbs up the chart</p>
      </div>

      <ul className="mt-4 grid gap-2 text-sm text-slate-700 sm:grid-cols-2">
        {calibration.buckets.map((bucket) => (
          <li key={`bucket-${bucket.lower_bound}`} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <p className="font-medium text-slate-800">
              {formatPercent(bucket.lower_bound)}–{formatPercent(bucket.upper_bound)}
            </p>
            <p className="mt-1 text-slate-600">
              {bucket.sample_size === 0
                ? "No resolved days yet."
                : `${formatPercent(bucket.predicted_rate)} predicted, ${formatPercent(bucket.actual_rate)} actually trained.`}
            </p>
            <p className="mt-1 text-xs text-slate-500">Sample size: {bucket.sample_size}</p>
          </li>
        ))}
      </ul>

      {calibration.total_samples < 20 ? (
        <p className="mt-4 text-xs text-slate-500">
          Still building up history — early buckets are honest but noisy until more daily predictions accumulate.
        </p>
      ) : null}
    </section>
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

  const generatedAt = formatGeneratedAt(prediction.generated_at);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-8 px-6 py-14">
      <header>
        <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
          Predicting for {prediction.date}
        </p>
        <h1 className="mt-2 text-4xl font-bold text-slate-900">train tomorrow</h1>
        {generatedAt ? <p className="mt-1 text-xs text-slate-500">Prediction generated {generatedAt}</p> : null}
      </header>

      <section className="rounded-2xl bg-slate-900 p-6 text-lg text-slate-50 shadow-sm">
        <ReactMarkdown components={blurbMarkdownComponents}>{prediction.blurb}</ReactMarkdown>
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
        <ContributorSummary
          topContributors={prediction.top_contributors}
          baselineLogOdds={
            prediction.baseline_log_odds ?? (
              prediction.baseline_probability === undefined
                ? undefined
                : logit(prediction.baseline_probability)
            )
          }
          otherContribution={prediction.other_contribution}
          finalProbability={prediction.probability}
        />
      </section>

      {prediction.calibration ? <CalibrationPlot calibration={prediction.calibration} /> : null}
    </main>
  );
}
