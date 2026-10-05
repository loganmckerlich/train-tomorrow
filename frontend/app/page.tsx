import { unstable_noStore as noStore } from "next/cache";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import WaterfallPlot from "./WaterfallPlot";
import Tabs from "./Tabs";
import FeatureSummary from "./FeatureSummary";
import type { FeatureSummaryItem } from "./FeatureSummary";
import MetricsPanel from "./MetricsPanel";
import type { ModelMetrics } from "./MetricsPanel";

const WATERFALL_CHART_MIN_HEIGHT = 180;
const WATERFALL_CHART_ROW_HEIGHT = 28;

const blurbMarkdownComponents: Components = {
  p: ({ children }) => <p className="mt-3 leading-relaxed first:mt-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-paper">{children}</strong>,
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

type ImpactDirection = "false_negative" | "false_positive";

type ImpactComparison = {
  lower_bound: number;
  upper_bound: number;
  direction: ImpactDirection;
  baseline_n: number;
  baseline_rate: number | null;
  live_n: number;
  live_rate: number | null;
  p_value: number | null;
  significant: boolean;
};

type RollingRate = {
  lower_bound: number;
  upper_bound: number;
  direction: ImpactDirection;
  baseline_rate: number | null;
  points: { date: string; n: number; rate: number }[];
};

type ImpactTracking = {
  deployment_date: string;
  significance_level: number;
  baseline_created_at: string | null;
  comparisons: ImpactComparison[];
  rolling: RollingRate[];
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
  impact_tracking?: ImpactTracking;
  feature_summary?: FeatureSummaryItem[];
  model_metrics?: ModelMetrics;
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
    <div className="mt-3 border-l border-slate-300 pl-3">
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
    </div>
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
  const chartHeightPx = Math.max(
    WATERFALL_CHART_MIN_HEIGHT,
    rows.length * WATERFALL_CHART_ROW_HEIGHT,
  );

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
        <figure className="mt-4 border-y border-slate-200 bg-white py-3">
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
          <p className="mt-1 font-mono text-left text-[11px] text-slate-600">
            Local scale: {formatPercent(chartProbabilityDomain[0], 0)} to {formatPercent(chartProbabilityDomain[1], 0)}
          </p>
          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-slate-600">
            <span><span className="mr-1 inline-block w-3 border-t border-dotted border-slate-500 align-middle" />Model average</span>
            <span><span className="mr-1 inline-block w-3 border-t border-slate-800 align-middle" />Final prediction</span>
          </div>
        </figure>
      )}
      <ul className="mt-4">
        {rows.map((row) => {
          const pointChange = row.pointChange;
          return (
            <li key={row.feature} className="contributor-row">
              <details>
                <summary className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-2">
                  <span className="font-medium text-slate-800">{row.phrase}</span>
                  <span
                    className={`data-reading text-right text-sm font-semibold ${row.direction === "helping" ? "text-emerald-600" : "text-rose-600"}`}
                    title={`SHAP-IQ ${formatSigned(row.signed_contribution, 3)} log odds`}
                  >
                    {formatPointChange(pointChange)}
                    {row.endProbability === null ? "" : ` · ${formatPercent(row.endProbability, 1)}`}
                  </span>
                  <span className="col-span-2 block h-1 bg-slate-200" title={`Probability change ${formatPointChange(pointChange)}`}>
                    <span
                      className={`block h-full ${row.direction === "helping" ? "bg-emerald-500" : "bg-rose-500"}`}
                      style={{ width: contributorBarWidth(pointChange ?? 0, maxContributionMagnitude) }}
                    />
                  </span>
                  <span className="col-span-2 font-mono text-xs text-slate-500">
                    {row.direction} · {formatLogOdds(row.signed_contribution)}
                  </span>
                </summary>
                <div className="mt-3">
                  <p className="font-mono text-xs text-slate-600">
                    SHAP-IQ contribution {formatLogOdds(row.signed_contribution)}; probability change {formatPointChange(pointChange)}.
                  </p>
                  <FeaturePlot contributor={row} />
                </div>
              </details>
            </li>
          );
        })}
      </ul>
    </>
  );
}

function ImpactRateChart({
  direction,
  comparisons,
}: {
  direction: ImpactDirection;
  comparisons: ImpactComparison[];
}) {
  const title = direction === "false_negative" ? "False negatives" : "False positives";
  return (
    <section className="mt-5">
      <h3 className="font-semibold text-slate-900">{title}: baseline vs. live</h3>
      <div className="mt-2 space-y-3">
        {comparisons.map((comparison) => (
          <div key={`${direction}-${comparison.lower_bound}`}>
            <p className="text-xs font-medium text-slate-700">
              {formatPercent(comparison.lower_bound)}–{formatPercent(comparison.upper_bound)}
            </p>
            <div className="mt-1 grid grid-cols-[4.5rem_1fr_auto] items-center gap-x-2 gap-y-1 text-xs">
              <span className="text-slate-500">Baseline</span>
              <div className="h-3 rounded-sm bg-slate-100">
                <div
                  className="h-3 rounded-sm bg-sky-500"
                  style={{ width: `${(comparison.baseline_rate ?? 0) * 100}%` }}
                  title={`Baseline rate ${formatPercent(comparison.baseline_rate)}`}
                />
              </div>
              <span className="tabular-nums text-slate-600">{formatPercent(comparison.baseline_rate)}</span>
              <span className="text-slate-500">Live</span>
              <div className="h-3 rounded-sm bg-slate-100">
                <div
                  className={`h-3 rounded-sm ${comparison.significant ? "bg-rose-600" : "bg-amber-500"}`}
                  style={{ width: `${(comparison.live_rate ?? 0) * 100}%` }}
                  title={`Live rate ${formatPercent(comparison.live_rate)}; p=${comparison.p_value ?? "n/a"}`}
                />
              </div>
              <span className="tabular-nums text-slate-600">
                {formatPercent(comparison.live_rate)}
                {comparison.significant ? " *" : ""}
              </span>
            </div>
            <p className="mt-1 text-[11px] text-slate-500">
              Sample size: baseline n={comparison.baseline_n}, live n={comparison.live_n}
              {comparison.significant ? " · significant excess" : ""}
            </p>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        Baseline and live bars share a 0–100% scale; * marks a one-sided significant excess.
      </p>
    </section>
  );
}

function ImpactRollingChart({ series }: { series: RollingRate }) {
  const dates = series.points.map((point) => Date.parse(point.date));
  const firstDate = Math.min(...dates);
  const lastDate = Math.max(...dates);
  const coordinates = series.points.map((point, index) => {
    const x = firstDate === lastDate ? 52 : 8 + ((dates[index] - firstDate) / (lastDate - firstDate)) * 88;
    const y = 92 - point.rate * 82;
    return `${x},${y}`;
  });
  const baselineY = series.baseline_rate === null ? null : 92 - series.baseline_rate * 82;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-2">
      <p className="text-xs font-medium text-slate-700">
        {formatPercent(series.lower_bound)}–{formatPercent(series.upper_bound)}
      </p>
      {series.points.length ? (
        <>
          <svg viewBox="0 0 100 100" className="mt-1 h-24 w-full" role="img" aria-label={`Rolling ${series.direction} rate`}>
            <line x1="8" x2="96" y1="92" y2="92" className="stroke-slate-300" strokeWidth="1" />
            {baselineY !== null ? (
              <line
                x1="8"
                x2="96"
                y1={baselineY}
                y2={baselineY}
                className="stroke-sky-500"
                strokeDasharray="4 3"
                strokeWidth="1.5"
              />
            ) : null}
            {coordinates.length > 1 ? (
              <polyline points={coordinates.join(" ")} fill="none" className="stroke-rose-500" strokeWidth="2" />
            ) : null}
            {coordinates.map((coordinate, index) => {
              const [x, y] = coordinate.split(",");
              return <circle key={series.points[index].date} cx={x} cy={y} r="2.5" className="fill-rose-500" />;
            })}
          </svg>
          <p className="text-[11px] text-slate-500">
            Live cumulative rate <span className="font-mono">{formatPercent(series.points.at(-1)?.rate ?? null)} · n={series.points.at(-1)?.n}</span>
          </p>
        </>
      ) : (
        <p className="mt-2 text-xs text-slate-500">No resolved live predictions in this bucket.</p>
      )}
    </div>
  );
}

function ImpactTrackingPanel({ impact }: { impact: ImpactTracking }) {
  return (
    <section className="logbook-panel">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Prediction impact by probability bucket</h2>
        <p className="mt-1 text-sm text-slate-600">
          Fixed pre-launch validation baseline vs. resolved daily predictions since {impact.deployment_date}.
        </p>
      </div>

      <ImpactRateChart
        direction="false_negative"
        comparisons={impact.comparisons.filter((comparison) => comparison.direction === "false_negative")}
      />
      <ImpactRateChart
        direction="false_positive"
        comparisons={impact.comparisons.filter((comparison) => comparison.direction === "false_positive")}
      />

      <div className="mt-5">
        <h3 className="font-semibold text-slate-900">Rolling live rates by bucket</h3>
        <p className="mt-1 text-xs text-slate-500">
          Cumulative live rate over time; the dashed line marks the frozen baseline. Each point includes resolved sample count.
        </p>
        {(["false_negative", "false_positive"] as const).map((direction) => (
          <div key={direction} className="mt-3">
            <h4 className="mb-2 text-sm font-medium text-slate-700">
              {direction === "false_negative" ? "False negatives" : "False positives"}
            </h4>
            <div className="grid gap-2">
              {impact.rolling
                .filter((series) => series.direction === direction)
                .map((series) => (
                  <ImpactRollingChart key={`${direction}-${series.lower_bound}`} series={series} />
                ))}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 overflow-x-auto">
        <h3 className="font-semibold text-slate-900">Comparison summary</h3>
        <table className="mt-2 min-w-full border-collapse text-left font-mono text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500">
              <th className="py-2 pr-3">Bucket</th>
              <th className="py-2 pr-3">Direction</th>
              <th className="py-2 pr-3">Baseline rate (n)</th>
              <th className="py-2 pr-3">Live rate (n)</th>
              <th className="py-2 pr-3">One-sided p-value</th>
              <th className="py-2">Significant</th>
            </tr>
          </thead>
          <tbody>
            {impact.comparisons.map((comparison) => (
              <tr
                key={`${comparison.direction}-${comparison.lower_bound}`}
                className="border-b border-slate-100 text-slate-700"
              >
                <td className="py-2 pr-3">
                  {formatPercent(comparison.lower_bound)}–{formatPercent(comparison.upper_bound)}
                </td>
                <td className="py-2 pr-3">{comparison.direction === "false_negative" ? "FN" : "FP"}</td>
                <td className="py-2 pr-3">
                  {formatPercent(comparison.baseline_rate)} (n={comparison.baseline_n})
                </td>
                <td className="py-2 pr-3">{formatPercent(comparison.live_rate)} (n={comparison.live_n})</td>
                <td className="py-2 pr-3">{comparison.p_value === null ? "n/a" : comparison.p_value.toFixed(4)}</td>
                <td className="py-2">{comparison.significant ? "Y" : "N"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-5 rounded-lg bg-amber-50 p-3 text-xs leading-relaxed text-amber-950">
        This is a correlational proxy, not a controlled experiment. It assumes baseline rates are stationary; season,
        training blocks, and life circumstances can shift training propensity independently of the app. Small daily
        samples make early results underpowered, so treat them as provisional as live n grows. Whether the prediction
        was viewed before training is not observable with available data, so this analysis cannot establish causality.
        Every rate is shown with sample sizes and a one-sided Fisher exact p-value; significance is not proof of a
        behavioral effect.
      </p>
    </section>
  );
}

export default async function Home() {
  const prediction = await getPrediction();

  if (!prediction) {
    return (
      <main className="instrument-page flex min-h-screen flex-col justify-center">
        <h1 className="font-mono text-xl font-semibold text-paper">train tomorrow</h1>
        <p className="mt-4 text-paper">Couldn’t load today’s prediction payload.</p>
      </main>
    );
  }

  const generatedAt = formatGeneratedAt(prediction.generated_at);

  return (
    <main className="instrument-page flex min-h-screen flex-col gap-8">
      <header className="border-b border-[var(--line)] pb-5">
        <p className="font-mono text-sm text-paper">{prediction.date}</p>
        <h1 className="mt-2 font-mono text-lg font-semibold tracking-tight text-paper">train tomorrow</h1>
        {generatedAt ? <p className="mt-1 font-mono text-xs text-slate-300">Prediction generated {generatedAt}</p> : null}
      </header>

      <section className="reading">
        <p className="text-xs text-slate-300">Train probability</p>
        <p className="headline-probability mt-3">{Math.round(prediction.probability * 100)}%</p>
        <p className="mt-3 text-xl leading-snug text-paper">
          {prediction.will_train ? "Likely training day" : "Likely recovery day"}
        </p>
        <p className="mt-5 font-mono text-sm text-slate-300">
          Predicted effort <span className="text-paper">{prediction.predicted_effort === null ? "n/a" : prediction.predicted_effort.toFixed(1)}</span>
          <span className="ml-2 font-sans text-xs">relative-effort points</span>
        </p>
      </section>

      <section className="daily-narrative max-w-2xl border-b border-[var(--line)] pb-6 text-base leading-relaxed text-paper">
        <ReactMarkdown components={blurbMarkdownComponents}>{prediction.blurb}</ReactMarkdown>
      </section>

      <Tabs
        tabs={[
          {
            id: "waterfall",
            label: "Why this prediction",
            content: (
              <section className="logbook-panel">
                <h2 className="text-xl font-semibold text-slate-900">How the estimate moves</h2>
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
            ),
          },
          ...(prediction.feature_summary?.length
            ? [{ id: "features", label: "Features", content: <FeatureSummary items={prediction.feature_summary} /> }]
            : []),
          ...(prediction.model_metrics
            ? [{ id: "metrics", label: "Model metrics", content: <MetricsPanel metrics={prediction.model_metrics} /> }]
            : []),
          ...(prediction.impact_tracking
            ? [{ id: "impact", label: "Prediction impact", content: <ImpactTrackingPanel impact={prediction.impact_tracking} /> }]
            : []),
        ]}
      />
    </main>
  );
}
