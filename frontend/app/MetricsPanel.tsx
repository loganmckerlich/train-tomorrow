import type { ReactNode } from "react";

export type ModelMetrics = {
  classifier: {
    n_train: number;
    n_holdout: number;
    holdout_start: string;
    holdout_end: string;
    positive_rate: number | null;
    accuracy: number | null;
    baseline_accuracy: number | null;
    auc: number | null;
    precision: number | null;
    recall: number | null;
    f1: number | null;
    log_loss: number | null;
    brier: number | null;
    roc_curve: { fpr: number; tpr: number }[];
    calibration: { mean_predicted: number; observed_rate: number; n: number }[];
  };
  regressor: { n_train: number; n_holdout: number; mae: number | null; baseline_mae: number | null };
};

const fmt = (value: number | null, digits = 3) => (value === null ? "n/a" : value.toFixed(digits));
const pct = (value: number | null) => (value === null ? "n/a" : `${(value * 100).toFixed(1)}%`);

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="font-mono text-xs text-slate-600">{label}</p>
      <p className="mt-1 font-mono text-2xl font-semibold tabular-nums text-slate-900">{value}</p>
      {hint ? <p className="mt-1 text-[11px] text-slate-500">{hint}</p> : null}
    </div>
  );
}

function Chart({ title, caption, children }: { title: string; caption: string; children: ReactNode }) {
  return (
    <figure className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <figcaption className="text-sm font-semibold text-slate-900">{title}</figcaption>
      <svg viewBox="0 0 100 100" className="mt-2 aspect-square w-full max-w-xs" role="img" aria-label={title}>
        <rect x="8" y="4" width="88" height="88" className="fill-slate-50 stroke-slate-200" strokeWidth="0.5" />
        <line x1="8" y1="92" x2="96" y2="4" className="stroke-slate-400" strokeDasharray="3 2" strokeWidth="0.7" />
        {children}
      </svg>
      <p className="mt-1 text-[11px] text-slate-500">{caption}</p>
    </figure>
  );
}

const x = (v: number) => 8 + v * 88;
const y = (v: number) => 92 - v * 88;

export default function MetricsPanel({ metrics }: { metrics: ModelMetrics }) {
  const c = metrics.classifier;
  const r = metrics.regressor;
  return (
    <section className="logbook-panel">
      <p className="rounded-lg bg-amber-50 p-3 text-xs leading-relaxed text-amber-950">
        These metrics come from a quick holdout of the most recent {c.n_holdout} days ({c.holdout_start} to{" "}
        {c.holdout_end}), scored by a model trained only on the {c.n_train} earlier days. The deployed model is then
        retrained on all data, including the holdout, because recency matters; its true accuracy is not measured here.
      </p>

      <h2 className="mt-5 text-sm font-semibold text-slate-900">Will I train? (classifier)</h2>
      <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Tile label="AUC" value={fmt(c.auc)} hint="0.5 = coin flip" />
        <Tile label="Accuracy" value={pct(c.accuracy)} hint={`always-majority baseline ${pct(c.baseline_accuracy)}`} />
        <Tile label="Precision" value={pct(c.precision)} />
        <Tile label="Recall" value={pct(c.recall)} />
        <Tile label="F1" value={fmt(c.f1)} />
        <Tile label="Log loss" value={fmt(c.log_loss)} hint="lower is better" />
        <Tile label="Brier" value={fmt(c.brier)} hint="lower is better" />
        <Tile label="Train rate in holdout" value={pct(c.positive_rate)} />
      </div>

      <div className="mt-4 grid gap-4">
        <Chart title="ROC curve" caption="False-positive rate (x) vs. true-positive rate (y); above the dashed line beats chance.">
          <polyline
            points={c.roc_curve.map((p) => `${x(p.fpr)},${y(p.tpr)}`).join(" ")}
            fill="none"
            className="stroke-sky-500"
            strokeWidth="1.6"
          />
        </Chart>
        <Chart title="Calibration" caption="Mean predicted probability (x) vs. observed train rate (y); dot area scales with n.">
          {c.calibration.map((p) => (
            <circle
              key={p.mean_predicted}
              cx={x(p.mean_predicted)}
              cy={y(p.observed_rate)}
              r={Math.min(7, 1.5 + Math.sqrt(p.n) / 2)}
              className="fill-amber-400 stroke-slate-900"
              strokeWidth="0.6"
            >
              <title>{`predicted ${pct(p.mean_predicted)}, observed ${pct(p.observed_rate)}, n=${p.n}`}</title>
            </circle>
          ))}
        </Chart>
      </div>

      <h2 className="mt-6 text-sm font-semibold text-slate-900">How hard? (effort regressor)</h2>
      <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Tile label="MAE" value={fmt(r.mae, 1)} hint="relative-effort points" />
        <Tile label="Median baseline MAE" value={fmt(r.baseline_mae, 1)} hint="always predict the training median" />
        <Tile label="Holdout rows" value={String(r.n_holdout)} hint={`${r.n_train} train rows (training days only)`} />
      </div>
    </section>
  );
}
