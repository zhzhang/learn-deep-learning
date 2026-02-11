import { InlineMath } from "react-katex";

const PLOT_SIZE = 320;
const PLOT_PADDING = 32;
const PLOT_DOMAIN: [number, number] = [-4, 4];
const PLOT_RANGE: [number, number] = [0, 4];

function scaleX(x: number): number {
  const [xMin, xMax] = PLOT_DOMAIN;
  return (
    PLOT_PADDING +
    ((x - xMin) / (xMax - xMin)) * (PLOT_SIZE - 2 * PLOT_PADDING)
  );
}

function scaleY(y: number): number {
  const [yMin, yMax] = PLOT_RANGE;
  return (
    PLOT_SIZE -
    PLOT_PADDING -
    ((y - yMin) / (yMax - yMin)) * (PLOT_SIZE - 2 * PLOT_PADDING)
  );
}

function relu(x: number): number {
  return Math.max(0, x);
}

function reluPath(stepCount = 200): string {
  const [xMin, xMax] = PLOT_DOMAIN;
  const step = (xMax - xMin) / stepCount;
  const points: string[] = [];

  for (let i = 0; i <= stepCount; i += 1) {
    const x = xMin + i * step;
    points.push(`${scaleX(x)},${scaleY(relu(x))}`);
  }

  return points.join(" ");
}

export default function ReluPlot() {
  return (
    <div className="mt-6 flex flex-col items-center rounded-md border border-[var(--border)] bg-[var(--card)] p-4">
      <svg
        role="img"
        aria-label="Plot of the ReLU activation function"
        viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`}
        className="h-auto w-full max-w-sm"
      >
        <line
          x1={scaleX(PLOT_DOMAIN[0])}
          y1={scaleY(0)}
          x2={scaleX(PLOT_DOMAIN[1])}
          y2={scaleY(0)}
          stroke="currentColor"
          strokeOpacity={0.4}
          strokeWidth={1.5}
        />
        <line
          x1={scaleX(0)}
          y1={scaleY(PLOT_RANGE[0])}
          x2={scaleX(0)}
          y2={scaleY(PLOT_RANGE[1])}
          stroke="currentColor"
          strokeOpacity={0.4}
          strokeWidth={1.5}
        />
        <polyline
          points={reluPath()}
          fill="none"
          stroke="currentColor"
          strokeWidth={3}
        />
        <text
          x={scaleX(PLOT_DOMAIN[1]) - 6}
          y={scaleY(0) - 6}
          textAnchor="end"
          className="fill-current text-[11px]"
        >
          x
        </text>
        <text
          x={scaleX(0) + 8}
          y={scaleY(PLOT_RANGE[1]) + 12}
          className="fill-current text-[11px]"
        >
          ReLU(x)
        </text>
      </svg>
      <p className="mt-3 text-center leading-relaxed text-[var(--foreground)]">
        <InlineMath>{`\\mathrm{ReLU}(x) = \\max(0, x)`}</InlineMath>
      </p>
    </div>
  );
}
