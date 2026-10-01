import Image from "next/image";
import { unstable_noStore as noStore } from "next/cache";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";

const blurbMarkdownComponents: Components = {
  p: ({ children }) => <p className="mt-3 leading-relaxed first:mt-0">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-amber-300">{children}</strong>,
  ul: ({ children }) => <ul className="mt-3 list-disc space-y-1 pl-5">{children}</ul>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
};

type Contributor = {
  indices?: number[];
  feature: string;
  signed_contribution: number;
  direction: "helping" | "hurting";
  phrase: string;
  is_interaction?: boolean;
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
  waterfall_steps?: Contributor[];
  explanation_plots?: Record<string, string>;
  baseline_probability?: number;
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

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isPngDataUrl(value: string | undefined): value is string {
  return typeof value === "string" && /^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(value);
}

function formatSigned(value: number | null, digits = 4): string {
  return isFiniteNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "n/a";
}

function formatPercent(value: number | null, digits = 0): string {
  return isFiniteNumber(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
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

function scale(value: number, domain: [number, number], range: [number, number]): number {
  const [domainMin, domainMax] = domain;
  const [rangeMin, rangeMax] = range;
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function PlotImage({
  src,
  alt,
  className,
}: {
  src: string | undefined;
  alt: string;
  className: string;
}) {
  if (!isPngDataUrl(src)) {
    return null;
  }

  return (
    <div className={`relative ${className}`}>
      <Image
        src={src}
        alt={alt}
        fill
        sizes="(max-width: 768px) 100vw, 768px"
        unoptimized
        className="object-contain"
      />
    </div>
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

  const waterfallSteps = prediction.waterfall_steps ?? prediction.top_contributors;
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
        <h2 className="text-lg font-semibold text-slate-900">Waterfall summary</h2>
        <p className="mt-1 text-xs text-slate-500">
          Contributions are shown in log-odds. Only the baseline and final endpoints are probabilities.
          Baseline {formatPercent(prediction.baseline_probability ?? null)} → final{" "}
          {formatPercent(prediction.probability)}.
        </p>
        <PlotImage
          src={prediction.explanation_plots?.waterfall}
          alt="shapiq waterfall showing log-odds contributions and baseline and prediction probabilities"
          className="mt-3 h-[460px] w-full"
        />
        <ol className="mt-4 space-y-2">
          {waterfallSteps.slice(0, 10).map((item) => (
            <li
              key={item.indices?.join("-") ?? item.feature}
              className="flex items-start justify-between gap-4 text-sm"
            >
              <span className="text-slate-700">{item.phrase}</span>
              <span
                className={
                  item.direction === "helping"
                    ? "text-right font-medium text-emerald-600"
                    : "text-right font-medium text-rose-600"
                }
                title="Contribution in raw model-margin log-odds"
              >
                {formatSigned(item.signed_contribution)} log-odds
              </span>
            </li>
          ))}
        </ol>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Top contributors</h2>
        <PlotImage
          src={prediction.explanation_plots?.top_n}
          alt="Top feature and interaction contributions in log-odds"
          className="mt-3 h-[260px] w-full"
        />
        <ul className="mt-4 space-y-3">
          {prediction.top_contributors.map((item) => (
            <li
              key={item.indices?.join("-") ?? item.feature}
              className="flex items-start justify-between gap-4 text-sm"
            >
              <span className="font-medium text-slate-800">{item.phrase}</span>
              <span
                className={
                  item.direction === "helping"
                    ? "text-right font-semibold text-emerald-600"
                    : "text-right font-semibold text-rose-600"
                }
                title="Contribution in raw model-margin log-odds"
              >
                {formatSigned(item.signed_contribution)} log-odds
                <span className="block text-xs font-normal text-slate-500">
                  {item.is_interaction ? "interaction" : item.direction}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section className="grid gap-4 sm:grid-cols-2">
        <figure className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <figcaption className="text-lg font-semibold text-slate-900">Interaction network</figcaption>
          <PlotImage
            src={prediction.explanation_plots?.network}
            alt="Network of main effects and pairwise interactions in log-odds"
            className="relative mt-3 h-[360px] w-full"
          />
        </figure>
        <figure className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <figcaption className="text-lg font-semibold text-slate-900">Force plot</figcaption>
          <PlotImage
            src={prediction.explanation_plots?.force}
            alt="Force plot of the same prediction explanation in log-odds"
            className="relative mt-3 h-[360px] w-full"
          />
        </figure>
      </section>

      {prediction.calibration ? <CalibrationPlot calibration={prediction.calibration} /> : null}
    </main>
  );
}
