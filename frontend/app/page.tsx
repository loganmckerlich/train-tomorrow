import { unstable_noStore as noStore } from "next/cache";

type Contributor = {
  feature: string;
  signed_contribution: number;
  direction: "helping" | "hurting";
  phrase: string;
  distribution?: {
    current_value: number;
    train_values: number[];
    rest_values: number[];
  };
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

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

type HistogramPoint = {
  restCount: number;
  trainCount: number;
};

function buildHistogram(distribution: Contributor["distribution"], bins = 16) {
  if (!distribution) {
    return null;
  }

  const values = [
    ...distribution.rest_values,
    ...distribution.train_values,
    distribution.current_value,
  ].filter(Number.isFinite);

  if (values.length === 0) {
    return null;
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 0.5;
    max += 0.5;
  }

  const range = max - min;
  const points: HistogramPoint[] = Array.from({ length: bins }, () => ({ restCount: 0, trainCount: 0 }));
  const addValues = (sample: number[], key: keyof HistogramPoint) => {
    sample.forEach((value) => {
      if (!Number.isFinite(value)) {
        return;
      }
      const position = (value - min) / range;
      const index = Math.min(bins - 1, Math.max(0, Math.floor(position * bins)));
      points[index][key] += 1;
    });
  };

  addValues(distribution.rest_values, "restCount");
  addValues(distribution.train_values, "trainCount");

  return {
    points,
    currentPosition: Math.max(0, Math.min(100, ((distribution.current_value - min) / range) * 100)),
    maxCount: Math.max(1, ...points.flatMap((point) => [point.restCount, point.trainCount])),
  };
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
              {(() => {
                const distribution = item.distribution;
                if (!distribution) {
                  return null;
                }

                const histogram = buildHistogram(distribution);
                if (!histogram) {
                  return null;
                }

                return (
                  <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
                      <span className="inline-flex items-center gap-1">
                        <span className="h-2.5 w-3 rounded-sm border border-amber-600 bg-amber-100" />
                        rest
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="h-2.5 w-1.5 rounded-sm bg-sky-500" />
                        train
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="h-3 w-px bg-slate-900" />
                        value {distribution.current_value.toFixed(1)}
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">
                      Rest avg {average(distribution.rest_values)?.toFixed(1) ?? "—"} across{" "}
                      {distribution.rest_values.length} days; train avg{" "}
                      {average(distribution.train_values)?.toFixed(1) ?? "—"} across{" "}
                      {distribution.train_values.length} days.
                    </p>
                    <div className="relative mt-3 h-32">
                      <svg viewBox="0 0 100 100" className="h-full w-full" aria-hidden="true">
                        {histogram.points.map((point, index) => {
                          const x = (index * 100) / histogram.points.length;
                          const width = 100 / histogram.points.length;
                          const restHeight = (point.restCount / histogram.maxCount) * 100;
                          const trainHeight = (point.trainCount / histogram.maxCount) * 100;
                          return (
                            <g key={`${item.feature}-${x}`}>
                              <rect
                                x={x + 0.5}
                                y={100 - restHeight}
                                width={Math.max(width - 1, 1)}
                                height={restHeight}
                                rx="1"
                                className="fill-amber-100 stroke-amber-600"
                                strokeWidth="0.5"
                              />
                              <rect
                                x={x + width * 0.25}
                                y={100 - trainHeight}
                                width={Math.max(width * 0.5, 1)}
                                height={trainHeight}
                                rx="1"
                                className="fill-sky-500"
                              />
                            </g>
                          );
                        })}
                        <line
                          x1={histogram.currentPosition}
                          x2={histogram.currentPosition}
                          y1="0"
                          y2="100"
                          className="stroke-slate-900"
                          strokeWidth="1.5"
                        />
                      </svg>
                    </div>
                  </div>
                );
              })()}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
