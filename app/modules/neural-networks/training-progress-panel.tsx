"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type RunStatus = "queued" | "running" | "completed" | "failed";

type ProgressEvent = {
  stage?: string;
  epoch?: number;
  epochs?: number;
  batch?: number;
  batches?: number;
  loss?: number;
  accuracy?: number;
  samples?: unknown;
  [key: string]: unknown;
};

type RunSnapshot = {
  status: RunStatus;
  result?: Record<string, unknown> | null;
  error?: string | null;
};

type StepLossPoint = {
  step: number;
  loss: number;
};

type SampleDescriptor = {
  sample_id: number;
  sample_index: number;
  class_index: number;
  class_name: string;
  slot: number;
};

type SampleEval = {
  sample_id: number;
  predicted_label: number;
  predicted_class_name: string;
  correct: boolean;
};

function parseSampleDescriptors(value: unknown): SampleDescriptor[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (typeof item !== "object" || item === null) return null;
      const candidate = item as Record<string, unknown>;
      if (
        typeof candidate.sample_id !== "number" ||
        typeof candidate.sample_index !== "number" ||
        typeof candidate.class_index !== "number" ||
        typeof candidate.class_name !== "string" ||
        typeof candidate.slot !== "number"
      ) {
        return null;
      }
      return {
        sample_id: candidate.sample_id,
        sample_index: candidate.sample_index,
        class_index: candidate.class_index,
        class_name: candidate.class_name,
        slot: candidate.slot,
      };
    })
    .filter((item): item is SampleDescriptor => item !== null)
    .sort((a, b) => a.class_index - b.class_index || a.slot - b.slot);
}

function parseSampleEvaluations(value: unknown): SampleEval[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (typeof item !== "object" || item === null) return null;
      const candidate = item as Record<string, unknown>;
      if (
        typeof candidate.sample_id !== "number" ||
        typeof candidate.predicted_label !== "number" ||
        typeof candidate.predicted_class_name !== "string" ||
        typeof candidate.correct !== "boolean"
      ) {
        return null;
      }
      return {
        sample_id: candidate.sample_id,
        predicted_label: candidate.predicted_label,
        predicted_class_name: candidate.predicted_class_name,
        correct: candidate.correct,
      };
    })
    .filter((item): item is SampleEval => item !== null);
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const CHART_WIDTH = 560;
const CHART_HEIGHT = 220;
const CHART_PADDING = 28;

function wsUrlForRun(): string {
  const normalized = API_BASE_URL.replace(/\/+$/, "");
  if (normalized.startsWith("https://")) {
    return `${normalized.replace("https://", "wss://")}/chapters/torch/ws`;
  }
  if (normalized.startsWith("http://")) {
    return `${normalized.replace("http://", "ws://")}/chapters/torch/ws`;
  }
  return `ws://${normalized}/chapters/torch/ws`;
}

export default function TrainingProgressPanel() {
  const [isStarting, setIsStarting] = useState(false);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [latestProgress, setLatestProgress] = useState<ProgressEvent | null>(
    null,
  );
  const [stepLoss, setStepLoss] = useState<StepLossPoint[]>([]);
  const [samples, setSamples] = useState<SampleDescriptor[]>([]);
  const [sampleEvalById, setSampleEvalById] = useState<Record<number, SampleEval>>(
    {},
  );
  const [sampleEvalEpoch, setSampleEvalEpoch] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canStart =
    !isStarting && runStatus !== "running" && runStatus !== "queued";

  const statusLabel = useMemo(() => {
    if (!runStatus) return "idle";
    return runStatus;
  }, [runStatus]);

  const lossSeries = useMemo(
    () => ({
      train: stepLoss.map((point) => ({ x: point.step, y: point.loss })),
      test: [] as Array<{ x: number; y: number }>,
    }),
    [stepLoss],
  );

  const stepCount = useMemo(() => {
    if (!stepLoss.length) return 1;
    return Math.max(1, stepLoss[stepLoss.length - 1]?.step ?? 1);
  }, [stepLoss]);

  const loadSamples = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chapters/torch/samples`);
      if (!response.ok) {
        return;
      }
      const data = (await response.json()) as { samples?: unknown };
      setSamples(parseSampleDescriptors(data.samples));
    } catch {
      // No-op: sample preview is optional and should not block training UI.
    }
  }, []);

  useEffect(() => {
    void loadSamples();
  }, [loadSamples]);

  const connectRunSocket = () => {
    const socket = new WebSocket(wsUrlForRun());

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as
          | { type: "snapshot"; run: RunSnapshot; run_status?: RunStatus }
          | { type: "status"; status: RunStatus }
          | ({ type: "progress" } & ProgressEvent)
          | { type: "stdout"; line: string }
          | { type: "terminal"; status: RunStatus };

        if (payload.type === "snapshot") {
          setRunStatus(payload.run.status);
          if (payload.run.error) {
            setError(payload.run.error);
          }
          return;
        }
        if (payload.type === "status") {
          setRunStatus(payload.status);
          return;
        }
        if (payload.type === "progress") {
          setLatestProgress(payload);
          if (payload.stage === "samples_ready") {
            setSamples(parseSampleDescriptors(payload.samples));
          }
          if (
            payload.stage === "sample_eval" &&
            typeof payload.epoch === "number"
          ) {
            const evaluations = parseSampleEvaluations(payload.samples);
            const nextMap: Record<number, SampleEval> = {};
            for (const evaluation of evaluations) {
              nextMap[evaluation.sample_id] = evaluation;
            }
            setSampleEvalById(nextMap);
            setSampleEvalEpoch(payload.epoch);
          }
          if (
            payload.stage === "train" &&
            typeof payload.epoch === "number" &&
            typeof payload.batch === "number" &&
            typeof payload.batches === "number" &&
            typeof payload.loss === "number"
          ) {
            const step = (payload.epoch - 1) * payload.batches + payload.batch;
            const loss = payload.loss;
            setStepLoss((previous) => [...previous, { step, loss }]);
          }
          return;
        }
        if (payload.type === "terminal") {
          setRunStatus(payload.status);
          socket.close();
        }
      } catch (parseError) {
        setError(`Unable to parse socket message: ${String(parseError)}`);
      }
    };

    socket.onerror = () => {
      setError("WebSocket error while streaming training progress.");
    };

    socket.onclose = () => {};
  };

  const startTraining = async () => {
    setIsStarting(true);
    setError(null);
    setRunStatus(null);
    setLatestProgress(null);
    setStepLoss([]);
    setSamples([]);
    setSampleEvalById({});
    setSampleEvalEpoch(null);

    try {
      const response = await fetch(`${API_BASE_URL}/chapters/torch/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: "callable",
          target: "modules.torch.train:train_reference_model",
          kwargs: {
            epochs: 5,
          },
        }),
      });
      if (!response.ok) {
        throw new Error(`Training request failed (${response.status})`);
      }
      const data = (await response.json()) as { run: RunSnapshot };
      setRunStatus(data.run.status);
      void loadSamples();
      connectRunSocket();
    } catch (startError) {
      setError(String(startError));
      setRunStatus("failed");
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="rounded-xl border border-[var(--surface-border)] bg-[var(--surface)] p-5">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={startTraining}
          disabled={!canStart}
          className="rounded-lg border border-[var(--accent)] px-4 py-2 text-sm font-semibold text-[var(--accent)] transition hover:bg-[color-mix(in_oklab,var(--accent),white_88%)] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isStarting ? "Starting..." : "Start Fashion-MNIST Training"}
        </button>
        <span className="text-sm text-[var(--muted)]">
          status: {statusLabel}
        </span>
      </div>

      {latestProgress && (
        <div className="mb-4 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
          <div>stage: {String(latestProgress.stage ?? "-")}</div>
          <div>
            epoch: {latestProgress.epoch ?? "-"} /{" "}
            {latestProgress.epochs ?? "-"}
          </div>
          <div>
            loss:{" "}
            {typeof latestProgress.loss === "number"
              ? latestProgress.loss.toFixed(4)
              : "-"}
          </div>
          <div>
            accuracy:{" "}
            {typeof latestProgress.accuracy === "number"
              ? `${(latestProgress.accuracy * 100).toFixed(2)}%`
              : "-"}
          </div>
        </div>
      )}

      {error && (
        <p className="mb-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      )}

      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <MetricChart
          title="Loss by step"
          xMax={stepCount}
          yMin={0}
          yMax={Math.max(1, ...lossSeries.train.map((point) => point.y))}
          primarySeries={lossSeries.train}
          secondarySeries={lossSeries.test}
          primaryLabel="Train"
          secondaryLabel=""
          yFormatter={(value) => value.toFixed(3)}
          xLabelPrefix="step"
        />
        <FashionMnistSampleGrid
          samples={samples}
          sampleEvalById={sampleEvalById}
          sampleEvalEpoch={sampleEvalEpoch}
        />
      </div>
    </div>
  );
}

function MetricChart({
  title,
  xMax,
  yMin,
  yMax,
  primarySeries,
  secondarySeries,
  primaryLabel,
  secondaryLabel,
  yFormatter,
  xLabelPrefix,
}: {
  title: string;
  xMax: number;
  yMin: number;
  yMax: number;
  primarySeries: Array<{ x: number; y: number }>;
  secondarySeries: Array<{ x: number; y: number }>;
  primaryLabel: string;
  secondaryLabel: string;
  yFormatter: (value: number) => string;
  xLabelPrefix: string;
}) {
  const primaryPath = linePath(primarySeries, xMax, yMin, yMax);
  const secondaryPath = linePath(secondarySeries, xMax, yMin, yMax);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="font-medium">{title}</span>
        <span className="flex items-center gap-3 text-[var(--muted)]">
          <LegendChip color="#2563eb" label={primaryLabel} />
          {secondaryLabel ? (
            <LegendChip color="#16a34a" label={secondaryLabel} />
          ) : null}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        className="h-56 w-full overflow-visible"
        role="img"
        aria-label={title}
      >
        <line
          x1={CHART_PADDING}
          y1={CHART_HEIGHT - CHART_PADDING}
          x2={CHART_WIDTH - CHART_PADDING}
          y2={CHART_HEIGHT - CHART_PADDING}
          stroke="#cbd5e1"
        />
        <line
          x1={CHART_PADDING}
          y1={CHART_PADDING}
          x2={CHART_PADDING}
          y2={CHART_HEIGHT - CHART_PADDING}
          stroke="#cbd5e1"
        />
        <text
          x={CHART_PADDING}
          y={CHART_PADDING - 8}
          fontSize="12"
          fill="#64748b"
        >
          {yFormatter(yMax)}
        </text>
        <text
          x={CHART_PADDING}
          y={CHART_HEIGHT - CHART_PADDING + 16}
          fontSize="12"
          fill="#64748b"
        >
          {yFormatter(yMin)}
        </text>
        <text
          x={CHART_WIDTH - CHART_PADDING - 8}
          y={CHART_HEIGHT - CHART_PADDING + 16}
          fontSize="12"
          fill="#64748b"
          textAnchor="end"
        >
          {xLabelPrefix} {xMax}
        </text>
        {primaryPath && (
          <path d={primaryPath} fill="none" stroke="#2563eb" strokeWidth="3" />
        )}
        {secondaryPath && (
          <path d={secondaryPath} fill="none" stroke="#16a34a" strokeWidth="3" />
        )}
      </svg>
    </div>
  );
}

function linePath(
  points: Array<{ x: number; y: number }>,
  xMax: number,
  yMin: number,
  yMax: number,
): string {
  if (points.length === 0) return "";
  const xRange = Math.max(1, xMax - 1);
  const yRange = Math.max(1e-9, yMax - yMin);

  return points
    .map((point, index) => {
      const x =
        CHART_PADDING +
        ((point.x - 1) / xRange) * (CHART_WIDTH - CHART_PADDING * 2);
      const y =
        CHART_HEIGHT -
        CHART_PADDING -
        ((point.y - yMin) / yRange) * (CHART_HEIGHT - CHART_PADDING * 2);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function LegendChip({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      <span>{label}</span>
    </span>
  );
}

function FashionMnistSampleGrid({
  samples,
  sampleEvalById,
  sampleEvalEpoch,
}: {
  samples: SampleDescriptor[];
  sampleEvalById: Record<number, SampleEval>;
  sampleEvalEpoch: number | null;
}) {
  const orderedSamples = [...samples].sort(
    (a, b) => a.class_index - b.class_index || a.slot - b.slot,
  );

  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="font-medium">Fashion-MNIST samples</span>
        <span className="text-[var(--muted)]">
          {sampleEvalEpoch ? `tint from epoch ${sampleEvalEpoch} test run` : "awaiting test run"}
        </span>
      </div>
      {orderedSamples.length > 0 ? (
        <div className="max-h-[32rem] overflow-auto">
          <div className="grid grid-cols-10 gap-0">
            {orderedSamples.map((sample) => {
              const evaluation = sampleEvalById[sample.sample_id];
              const tintClass = evaluation
                ? evaluation.correct
                  ? "bg-green-500/30"
                  : "bg-red-500/30"
                : "bg-transparent";
              const title = evaluation
                ? `${sample.class_name} | predicted ${evaluation.predicted_class_name}`
                : `${sample.class_name} | prediction pending`;
              return (
                <div
                  key={sample.sample_id}
                  className="relative aspect-square overflow-hidden"
                  title={title}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API_BASE_URL}/chapters/torch/samples/${sample.sample_id}.png`}
                    alt={`${sample.class_name} sample ${sample.slot + 1}`}
                    className="block h-full w-full object-cover"
                    loading="lazy"
                  />
                  <div className={`pointer-events-none absolute inset-0 ${tintClass}`} />
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="text-xs text-[var(--muted)]">
          Samples will appear here after the server prepares the dataset.
        </p>
      )}
      <p className="mt-2 text-xs text-[var(--muted)]">
        Images are streamed from the FastAPI sample endpoint.
      </p>
    </div>
  );
}
