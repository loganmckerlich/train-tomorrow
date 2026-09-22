import { readFileSync } from "node:fs";
import { join } from "node:path";

type TopContributor = {
  feature: string;
  signed_contribution: number;
  phrase: string;
};

type Prediction = {
  prediction_date: string;
  will_train_tomorrow: boolean;
  train_probability: number;
  predicted_workload_tss: number | null;
  top_contributors: TopContributor[];
  challenge_blurb: string;
};

export default function HomePage() {
  const predictionPath = join(process.cwd(), "..", "data", "latest_prediction.json");
  const prediction = JSON.parse(readFileSync(predictionPath, "utf8")) as Prediction;
  const tssText =
    prediction.predicted_workload_tss === null
      ? "Rest or recovery likely"
      : `${prediction.predicted_workload_tss} TSS predicted`;

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", margin: "2rem auto", maxWidth: 640 }}>
      <h1>train-tomorrow</h1>
      <p>
        <strong>Will I train tomorrow?</strong>{" "}
        {prediction.will_train_tomorrow ? "Yes" : "No"} ({Math.round(prediction.train_probability * 100)}%)
      </p>
      <p>
        <strong>How hard?</strong> {tssText}
      </p>
      <p>{prediction.challenge_blurb}</p>
    </main>
  );
}
