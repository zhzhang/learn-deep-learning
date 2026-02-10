"use client";

import { useMemo, useState } from "react";

type RunStatus = "queued" | "running" | "completed" | "failed";

type ProgressEvent = {
  stage?: string;
  epoch?: number;
  epochs?: number;
  batch?: number;
  batches?: number;
  loss?: number;
  accuracy?: number;
  model_path?: string;
  [key: string]: unknown;
};

type RunSnapshot = {
  status: RunStatus;
  result?: Record<string, unknown> | null;
  error?: string | null;
};

type EpochMetric = {
  epoch: number;
  trainLoss?: number;
  testLoss?: number;
  trainAccuracy?: number;
  testAccuracy?: number;
};

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
  const [epochMetrics, setEpochMetrics] = useState<EpochMetric[]>([]);
  const [error, setError] = useState<string | null>(null);

  const canStart =
    !isStarting && runStatus !== "running" && runStatus !== "queued";

  const statusLabel = useMemo(() => {
    if (!runStatus) return "idle";
    return runStatus;
  }, [runStatus]);

  const lossSeries = useMemo(() => {
    return {
      train: epochMetrics
        .map((metric) =>
          typeof metric.trainLoss === "number"
            ? { x: metric.epoch, y: metric.trainLoss }
            : null,
        )
        .filter((value): value is { x: number; y: number } => value !== null),
      test: epochMetrics
        .map((metric) =>
          typeof metric.testLoss === "number"
            ? { x: metric.epoch, y: metric.testLoss }
            : null,
        )
        .filter((value): value is { x: number; y: number } => value !== null),
    };
  }, [epochMetrics]);

  const accuracySeries = useMemo(() => {
    return {
      train: epochMetrics
        .map((metric) =>
          typeof metric.trainAccuracy === "number"
            ? { x: metric.epoch, y: metric.trainAccuracy }
            : null,
        )
        .filter((value): value is { x: number; y: number } => value !== null),
      test: epochMetrics
        .map((metric) =>
          typeof metric.testAccuracy === "number"
            ? { x: metric.epoch, y: metric.testAccuracy }
            : null,
        )
        .filter((value): value is { x: number; y: number } => value !== null),
    };
  }, [epochMetrics]);

  const epochCount = useMemo(() => {
    if (!epochMetrics.length) return 1;
    const maxEpoch = Math.max(...epochMetrics.map((metric) => metric.epoch));
    return Math.max(1, maxEpoch);
  }, [epochMetrics]);

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
          if (
            payload.stage === "train_summary" &&
            typeof payload.epoch === "number" &&
            typeof payload.loss === "number" &&
            typeof payload.accuracy === "number"
          ) {
            const epoch = payload.epoch;
            const loss = payload.loss;
            const accuracy = payload.accuracy;
            setEpochMetrics((previous) => {
              const next = [...previous];
              const index = next.findIndex((metric) => metric.epoch === epoch);
              if (index === -1) {
                next.push({
                  epoch,
                  trainLoss: loss,
                  trainAccuracy: accuracy,
                });
              } else {
                next[index] = {
                  ...next[index],
                  trainLoss: loss,
                  trainAccuracy: accuracy,
                };
              }
              next.sort((a, b) => a.epoch - b.epoch);
              return next;
            });
          }
          if (
            payload.stage === "test_summary" &&
            typeof payload.epoch === "number" &&
            typeof payload.loss === "number" &&
            typeof payload.accuracy === "number"
          ) {
            const epoch = payload.epoch;
            const loss = payload.loss;
            const accuracy = payload.accuracy;
            setEpochMetrics((previous) => {
              const next = [...previous];
              const index = next.findIndex((metric) => metric.epoch === epoch);
              if (index === -1) {
                next.push({
                  epoch,
                  testLoss: loss,
                  testAccuracy: accuracy,
                });
              } else {
                next[index] = {
                  ...next[index],
                  testLoss: loss,
                  testAccuracy: accuracy,
                };
              }
              next.sort((a, b) => a.epoch - b.epoch);
              return next;
            });
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
    setEpochMetrics([]);

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

      <div className="grid gap-4 rounded-md border border-[var(--surface-border)] bg-white/50 p-3 md:grid-cols-2">
        <MetricChart
          title="Loss by epoch"
          xMax={epochCount}
          yMin={0}
          yMax={Math.max(1, ...lossSeries.train.map((point) => point.y), ...lossSeries.test.map((point) => point.y))}
          primarySeries={lossSeries.train}
          secondarySeries={lossSeries.test}
          primaryLabel="Train"
          secondaryLabel="Test"
          yFormatter={(value) => value.toFixed(3)}
        />
        <MetricChart
          title="Accuracy by epoch"
          xMax={epochCount}
          yMin={0}
          yMax={1}
          primarySeries={accuracySeries.train}
          secondarySeries={accuracySeries.test}
          primaryLabel="Train"
          secondaryLabel="Test"
          yFormatter={(value) => `${(value * 100).toFixed(1)}%`}
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
}) {
  const primaryPath = linePath(primarySeries, xMax, yMin, yMax);
  const secondaryPath = linePath(secondarySeries, xMax, yMin, yMax);

  return (
    <div className="rounded-md border border-[var(--surface-border)] p-3">
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="font-medium">{title}</span>
        <span className="flex items-center gap-3 text-[var(--muted)]">
          <LegendChip color="#2563eb" label={primaryLabel} />
          <LegendChip color="#16a34a" label={secondaryLabel} />
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
          epoch {xMax}
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
