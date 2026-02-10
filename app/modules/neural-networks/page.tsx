import Link from "next/link";

export const metadata = {
  title: "Neural Networks | Deep Learning Textbook",
  description: "Introduction to neural networks: layers, weights, activations, and learning.",
};

export default function NeuralNetworksPage() {
  return (
    <div className="min-h-screen px-6 py-12 font-sans text-[var(--foreground)]">
      <main className="mx-auto w-full max-w-3xl">
        <nav aria-label="Breadcrumb" className="mb-10">
          <ol className="flex flex-wrap items-center gap-2 text-sm text-[var(--muted)]">
            <li>
              <Link
                href="/"
                className="transition hover:text-[var(--accent)] focus:underline focus:outline-none"
              >
                Home
              </Link>
            </li>
            <li aria-hidden="true">/</li>
            <li className="text-[var(--foreground)]" aria-current="page">
              Neural Networks
            </li>
          </ol>
        </nav>

        <header className="mb-12">
          <h1 className="font-[var(--font-display)] text-3xl font-semibold tracking-tight text-[var(--foreground)] sm:text-4xl">
            Neural Networks and Backpropagation
          </h1>
          <p className="mt-3 text-lg text-[var(--muted)]">
            The building blocks of deep learning: from a single neuron to layers
            that learn from data.
          </p>
        </header>

        <article className="space-y-12">
          <section>
            <h2 className="font-[var(--font-display)] text-xl font-semibold text-[var(--foreground)]">
              What&apos;s the big deal?
            </h2>
            <p className="mt-3 leading-relaxed text-[var(--foreground)]">
              A long time ago, researchers were trying to figure out how to draw lines between data points
              in some N-dimensional space. 
            </p>
          </section>

          <section>
            <h2 className="font-[var(--font-display)] text-xl font-semibold text-[var(--foreground)]">
              The forward pass
            </h2>
            <p className="mt-3 leading-relaxed text-[var(--foreground)]">
              Given an input (e.g. a vector or an image flattened into numbers),
              we pass it through the first layer: multiply by a weight matrix,
              add biases, then apply an activation (e.g. ReLU). The result is
              the input to the next layer. Repeating this layer by layer until
              the final layer produces the network’s <strong>output</strong>—
              for example a class score or a regression value. This sequence is
              called the <strong>forward pass</strong>.
            </p>
          </section>

          <section>
            <h2 className="font-[var(--font-display)] text-xl font-semibold text-[var(--foreground)]">
              Loss and the backward pass
            </h2>
            <p className="mt-3 leading-relaxed text-[var(--foreground)]">
              We compare the output to the desired target with a{" "}
              <strong>loss function</strong> (e.g. cross-entropy for
              classification, mean squared error for regression). To improve, we
              need to know how each weight and bias affects the loss.{" "}
              <strong>Backpropagation</strong> computes these gradients
              efficiently by applying the chain rule from the loss back through
              every layer. With the gradients in hand, we take a step in the
              opposite direction—<strong>gradient descent</strong>—to reduce the
              loss and, over many steps, fit the network to the data.
            </p>
          </section>

          <section className="rounded-2xl border border-[var(--surface-border)] bg-[var(--surface)] p-6 shadow-[0_2px_10px_rgba(47,35,24,0.06)]">
            <h2 className="font-[var(--font-display)] text-xl font-semibold text-[var(--foreground)]">
              Key ideas
            </h2>
            <ul className="mt-4 space-y-2 text-[var(--foreground)]" role="list">
              <li className="flex gap-3">
                <span className="text-[var(--accent)]" aria-hidden="true">
                  •
                </span>
                <span>
                  <strong>Layers</strong> transform inputs using weights and
                  biases, then activations.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="text-[var(--accent)]" aria-hidden="true">
                  •
                </span>
                <span>
                  <strong>Forward pass</strong>: input → layer 1 → … → output.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="text-[var(--accent)]" aria-hidden="true">
                  •
                </span>
                <span>
                  <strong>Loss</strong> measures how far the output is from the
                  target.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="text-[var(--accent)]" aria-hidden="true">
                  •
                </span>
                <span>
                  <strong>Backpropagation</strong> computes gradients of the
                  loss with respect to every parameter.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="text-[var(--accent)]" aria-hidden="true">
                  •
                </span>
                <span>
                  <strong>Gradient descent</strong> updates weights to minimize
                  the loss.
                </span>
              </li>
            </ul>
          </section>
        </article>

        <footer className="mt-16 flex flex-wrap items-center justify-between gap-4 border-t border-[var(--surface-border)] pt-8">
          <Link
            href="/"
            className="text-[var(--accent)] transition hover:text-[var(--accent-strong)] focus:underline focus:outline-none"
          >
            ← Back to modules
          </Link>
        </footer>
      </main>
    </div>
  );
}
