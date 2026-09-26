import { unstable_noStore as noStore } from "next/cache";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

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
  signed_contribution: number;
  direction: "helping" | "hurting";
  phrase: string;
  plot?: ContinuousPlot | CategoricalPlot;
};

type WaterfallStep = {
  feature: string;
  signed_contribution: number;
  direction: "helping" | "hurting";
  phrase: string;
};

type WaterfallSummary = {
  baseline_probability: number;
  final_probability: number;
  steps: WaterfallStep[];
};

type WaterfallSegment = WaterfallStep & {
  startProbability: number;
  endProbability: number;
  deltaPoints: number;
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
  waterfall?: WaterfallSummary;
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

function contributionPoints(shapValue: number, probability: number): number {
  const boundedProbability = clampProbability(probability);
  return (boundedProbability - sigmoid(logit(boundedProbability) - shapValue)) * 100;
}

function formatContributionPoints(value: number | null, digits = 1): string {
  return isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)} points` : "n/a";
}

function buildWaterfallSegments(waterfall: WaterfallSummary): WaterfallSegment[] {
  return waterfall.steps.reduce<{
    items: WaterfallSegment[];
    runningLogOdds: number;
  }>(
    (state, step) => {
      const startProbability = sigmoid(state.runningLogOdds);
      const nextLogOdds = state.runningLogOdds + step.signed_contribution;
      const endProbability = sigmoid(nextLogOdds);
      return {
        runningLogOdds: nextLogOdds,
        items: [
          ...state.items,
          {
            ...step,
            startProbability,
            endProbability,
            deltaPoints: (endProbability - startProbability) * 100,
          },
        ],
      };
    },
    { items: [], runningLogOdds: logit(waterfall.baseline_probability) },
  ).items;
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
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }) + " UTC";
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

function ContinuousFeaturePlot({
  feature,
  plot,
  probability,
}: {
  feature: string;
  plot: ContinuousPlot;
  probability: number;
}) {
  const historicalPoints = plot.points.filter(
    (point) => Number.isFinite(point.feature_value) && Number.isFinite(point.shap_value),
  );
  const currentVisible = isFiniteNumber(plot.current_value) && isFiniteNumber(plot.current_shap);
  const currentValue = currentVisible ? plot.current_value : null;
  const currentShap = currentVisible ? plot.current_shap : null;
  const currentPoints = currentShap === null ? null : contributionPoints(currentShap, probability);
  const xExtent = paddedExtent([
    ...historicalPoints.map((point) => point.feature_value),
    ...(currentValue === null ? [] : [currentValue]),
  ]);
  const historicalEffects = historicalPoints.map((point) => contributionPoints(point.shap_value, probability));
  const yExtent = paddedExtent([
    ...historicalEffects,
    ...(currentPoints === null ? [0] : [currentPoints, 0]),
  ]);

  if (!xExtent || !yExtent) {
    return null;
  }

  const zeroY = scale(0, yExtent, [90, 10]);
  const historicalCount = historicalPoints.length;
  const summary = `Historical days: ${historicalCount}. Feature values ranged from ${xExtent[0].toFixed(1)} to ${xExtent[1].toFixed(1)}. Approximate model effect ranged from ${formatContributionPoints(yExtent[0])} to ${formatContributionPoints(yExtent[1])}.`;
  const idBase = `${feature}-continuous-plot`;

  return (
    <figure
      className="mt-3"
      role="img"
      aria-labelledby={`${idBase}-title`}
      aria-describedby={`${idBase}-today ${idBase}-summary`}
    >
      <figcaption id={`${idBase}-title`} className="text-xs text-slate-500">
        Approximate training-probability lift vs. feature value. Above 0 pushes toward training; below 0 pushes away.
      </figcaption>
      <p id={`${idBase}-today`} className="mt-1 text-xs text-slate-500">
        Today: value {currentValue === null ? "n/a" : currentValue.toFixed(1)}, about{" "}
        <span title={currentShap === null ? undefined : `Raw SHAP ${formatSigned(currentShap, 4)} log-odds`}>
          {formatContributionPoints(currentPoints)}
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
            const effectPoints = contributionPoints(point.shap_value, probability);
            return (
              <circle
                key={`${point.feature_value}-${point.shap_value}-${index}`}
                cx={scale(point.feature_value, xExtent, [8, 96])}
                cy={scale(effectPoints, yExtent, [90, 10])}
                r="1.9"
                className="fill-slate-500"
                fillOpacity="0.45"
              >
                <title>
                  {`Approx. ${formatContributionPoints(effectPoints)} (raw SHAP ${formatSigned(point.shap_value, 4)})`}
                </title>
              </circle>
            );
          })}
          {currentValue !== null && currentShap !== null && currentPoints !== null ? (
            <circle
              cx={scale(currentValue, xExtent, [8, 96])}
              cy={scale(currentPoints, yExtent, [90, 10])}
              r="3.4"
              className="fill-amber-400 stroke-slate-900"
              strokeWidth="1.5"
            >
              <title>{`Today: ${formatContributionPoints(currentPoints)} (raw SHAP ${formatSigned(currentShap, 4)})`}</title>
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
        Approx. effect range {formatContributionPoints(yExtent[0])} to {formatContributionPoints(yExtent[1])}
      </p>
    </figure>
  );
}

function CategoricalFeaturePlot({
  feature,
  plot,
  probability,
}: {
  feature: string;
  plot: CategoricalPlot;
  probability: number;
}) {
  if (plot.categories.length === 0) {
    return null;
  }

  const categoryEffects = plot.categories.map((category) =>
    contributionPoints(category.mean_shap, probability),
  );
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
        Approximate mean training-probability lift by category. Above 0 pushes toward training; below 0 pushes away.
      </figcaption>
      <p id={`${idBase}-today`} className="mt-1 text-xs text-slate-500">
        Today&apos;s category: {currentLabel}.
      </p>
      <ul id={`${idBase}-values`} className="mt-1 space-y-1 text-xs text-slate-500">
        {plot.categories.map((category) => {
          const effectPoints = contributionPoints(category.mean_shap, probability);
          return (
            <li key={`summary-${category.value}`}>
              <span title={`Raw SHAP ${formatSigned(category.mean_shap, 4)} log-odds`}>
                {category.label}: {formatContributionPoints(effectPoints)}
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
            const effectPoints = contributionPoints(category.mean_shap, probability);
            const y = scale(effectPoints, yExtent, [90, 10]);
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
                  {`${category.label}: ${formatContributionPoints(effectPoints)} (raw SHAP ${formatSigned(category.mean_shap, 4)})`}
                </title>
              </rect>
            );
          })}
        </svg>
      </div>
      <p className="mt-1 text-center text-[11px] text-slate-500">
        Approx. effect range {formatContributionPoints(yExtent[0])} to {formatContributionPoints(yExtent[1])}
      </p>
    </figure>
  );
}

function FeaturePlot({ contributor, probability }: { contributor: Contributor; probability: number }) {
  if (!contributor.plot) {
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
          probability={probability}
        />
      ) : (
        <CategoricalFeaturePlot
          feature={contributor.feature}
          plot={contributor.plot}
          probability={probability}
        />
      )}
    </details>
  );
}

function WaterfallPlot({ waterfall }: { waterfall: WaterfallSummary }) {
  if (waterfall.steps.length === 0) {
    return null;
  }

  const segments = buildWaterfallSegments(waterfall);
  const chartHeight = 16 + segments.length * 12;

  return (
    <figure className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
      <figcaption className="text-sm font-medium text-slate-800">How the model got here</figcaption>
      <p className="mt-1 text-xs text-slate-500">
        Starts from the model&apos;s average day ({formatPercent(waterfall.baseline_probability)})
        and walks through today&apos;s biggest pushes to land at{" "}
        {formatPercent(waterfall.final_probability)}.
      </p>
      <div className="mt-4 rounded-lg border border-slate-200 bg-white p-3">
        <svg
          viewBox={`0 0 100 ${chartHeight}`}
          className="w-full"
          style={{ height: `${Math.max(180, segments.length * 28)}px` }}
          aria-hidden="true"
        >
          {segments.map((segment, index) => {
            const y = 10 + index * 12;
            const startX = scale(segment.startProbability, [0, 1], [8, 96]);
            const endX = scale(segment.endProbability, [0, 1], [8, 96]);
            return (
              <g key={`${segment.feature}-${index}`}>
                <line
                  x1={startX}
                  x2={endX}
                  y1={y}
                  y2={y}
                  className={segment.direction === "helping" ? "stroke-emerald-500" : "stroke-rose-500"}
                  strokeWidth="6"
                  strokeLinecap="round"
                />
                <circle cx={startX} cy={y} r="1.5" className="fill-white stroke-slate-400" strokeWidth="0.8" />
                <circle cx={endX} cy={y} r="1.8" className="fill-slate-900" />
              </g>
            );
          })}
        </svg>
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
          <span>0%</span>
          <span>train probability</span>
          <span>100%</span>
        </div>
      </div>
      <ol className="mt-4 space-y-2">
        {segments.map((segment, index) => (
          <li key={`${segment.feature}-summary-${index}`} className="flex items-start justify-between gap-4 text-sm">
            <span className="text-slate-700">{segment.phrase}</span>
            <span
              className={
                segment.direction === "helping"
                  ? "text-right font-medium text-emerald-600"
                  : "text-right font-medium text-rose-600"
              }
              title={`Raw SHAP ${formatSigned(segment.signed_contribution, 4)} log-odds`}
            >
              {formatContributionPoints(segment.deltaPoints)} → {formatPercent(segment.endProbability)}
            </span>
          </li>
        ))}
      </ol>
    </figure>
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

  const waterfallPointsByFeature = (prediction.waterfall ? buildWaterfallSegments(prediction.waterfall) : []).reduce(
    (pointsByFeature, segment) => {
      pointsByFeature.set(segment.feature, (pointsByFeature.get(segment.feature) ?? 0) + segment.deltaPoints);
      return pointsByFeature;
    },
    new Map<string, number>(),
  );
  const contributionPointValues = prediction.top_contributors.map(
    (item) =>
      waterfallPointsByFeature.get(item.feature) ??
      contributionPoints(item.signed_contribution, prediction.probability),
  );
  const contributionMagnitudes = contributionPointValues.map((value) => Math.abs(value));
  const maxContributionMagnitude = Math.max(...contributionMagnitudes, 1);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col gap-8 px-6 py-14">
      <header>
        <p className="text-sm font-medium uppercase tracking-wide text-slate-500">
          Predicting for {prediction.date}
        </p>
        <h1 className="mt-2 text-4xl font-bold text-slate-900">train tomorrow</h1>
        {formatGeneratedAt(prediction.generated_at) ? (
          <p className="mt-1 text-xs text-slate-500">
            Prediction generated {formatGeneratedAt(prediction.generated_at)}
          </p>
        ) : null}
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

      {prediction.calibration ? <CalibrationPlot calibration={prediction.calibration} /> : null}

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Top contributors</h2>
        {prediction.waterfall ? <WaterfallPlot waterfall={prediction.waterfall} /> : null}
        <ul className="mt-4 space-y-4">
          {prediction.top_contributors.map((item) => {
            const effectPoints =
              waterfallPointsByFeature.get(item.feature) ??
              contributionPoints(item.signed_contribution, prediction.probability);
            return (
              <li key={item.feature}>
                <div className="mb-1 flex items-center justify-between gap-4 text-sm">
                  <span className="font-medium text-slate-800">{item.phrase}</span>
                  <div className="text-right">
                    <span
                      className={
                        item.direction === "helping"
                          ? "font-semibold text-emerald-600"
                          : "font-semibold text-rose-600"
                      }
                      title={`Raw SHAP ${formatSigned(item.signed_contribution, 4)} log-odds`}
                    >
                      {formatContributionPoints(effectPoints)}
                    </span>
                    <p className="text-xs text-slate-500">{item.direction}</p>
                  </div>
                </div>
                <div className="h-2 rounded-full bg-slate-200">
                  <div
                    className={`h-2 rounded-full ${
                      item.direction === "helping" ? "bg-emerald-500" : "bg-rose-500"
                    }`}
                    style={{ width: contributorBarWidth(effectPoints, maxContributionMagnitude) }}
                    title={`Raw SHAP ${formatSigned(item.signed_contribution, 4)} log-odds`}
                  />
                </div>
                <FeaturePlot contributor={item} probability={prediction.probability} />
              </li>
            );
          })}
        </ul>
      </section>
    </main>
  );
}
