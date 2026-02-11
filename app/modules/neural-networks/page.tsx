import {
  BreadcrumbLink,
  BodyText,
  CodeBlock,
  DisplayTitle,
  InlineLink,
  LeadText,
  SectionTitle,
} from "@/app/components/typography";
import TrainingProgressPanel from "./training-progress-panel";
import ReluPlot from "@/app/modules/neural-networks/relu-plot";
import { BlockMath } from "react-katex";
import { InlineMath } from "react-katex";

export const metadata = {
  title: "Neural Networks | Deep Learning Textbook",
  description: "Let's dive into the deep end and replicate Pytorch",
};

export default function NeuralNetworksPage() {
  return (
    <div className="min-h-screen py-12 font-sans text-[var(--foreground)]">
      <main className="space-y-12">
        <section className="mx-auto w-full max-w-3xl px-6">
          <nav aria-label="Breadcrumb" className="mb-10">
            <ol className="flex flex-wrap items-center gap-2 text-sm text-[var(--muted)]">
              <li>
                <BreadcrumbLink href="/">Home</BreadcrumbLink>
              </li>
              <li aria-hidden="true">/</li>
              <li className="text-[var(--foreground)]" aria-current="page">
                Neural Networks
              </li>
            </ol>
          </nav>

          <header>
            <DisplayTitle>Tensors and Backpropagation</DisplayTitle>
            <LeadText>
              Let&apos;s dive into the deep end and replicate Pytorch
            </LeadText>
          </header>
        </section>

        <article className="space-y-12">
          <section className="mx-auto w-full max-w-3xl px-6">
            <BodyText>
              We&apos;re going to replicate, largely from scratch, enough of
              Pytorch&apos;s API to train the neural network that is in
              Pytorch&apos;s{" "}
              <InlineLink href="https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html">
                quickstart tutorial
              </InlineLink>{" "}
              on Fashion MNIST.
            </BodyText>
          </section>

          <section className="mx-auto w-full max-w-6xl px-6">
            <TrainingProgressPanel />
          </section>

          <section className="mx-auto w-full max-w-3xl px-6">
            <SectionTitle>How does that work?</SectionTitle>
            <BodyText>
              If you looked at `modules/torch/reference.py`, you&apos;ll see the
              reference implement of a multi-layer perceptron (MLP).
              Specifically, the part that defines the model architecture are the
              lines:
            </BodyText>
            <CodeBlock language="python">
              {`
nn.Sequential(
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)`}
            </CodeBlock>
            <BodyText>
              What this code says is that we have a model with 3 linear layers,
              with rectified linear units (ReLU) between each layer.
            </BodyText>
            <BodyText>
              A linear layer is a matrix multiplication followed by a bias
              addition.
            </BodyText>
            <BlockMath>y = Wx + b</BlockMath>
            <BodyText>For example, with concrete vectors:</BodyText>
            <BlockMath>{`\\begin{bmatrix}y_1 \\\\ y_2\\end{bmatrix}
=
\\begin{bmatrix}2 & -1 \\\\ 0 & 3\\end{bmatrix}
\\begin{bmatrix}4 \\\\ 5\\end{bmatrix}
+
\\begin{bmatrix}1 \\\\ -2\\end{bmatrix}
=
\\begin{bmatrix}4 \\\\ 13\\end{bmatrix}`}</BlockMath>
            <BodyText>
              If this looks the same as the formula for a line you learned in
              algebra <InlineMath>y = mx + b</InlineMath>, that&apos;s because
              it is! Linear layers define a line in n-dimensional space.
            </BodyText>
            <BodyText>
              The role of the ReLU activation function is to introduce
              non-linearity into the model, and allow us to &quot;pick&quot; the
              data that falls only on one side of the line defined by the linear
              layer. All that ReLU does is set any negative values to 0.
            </BodyText>
            <BlockMath>{`\\text{ReLU}\\left(
\\begin{bmatrix}
-2.3 & 1.7 & 0.0 \\\\
3.2 & -4.8 & 5.1 \\\\
-1.4 & -6.0 & 2.9
\\end{bmatrix}
\\right)
=
\\begin{bmatrix}
0.0 & 1.7 & 0.0 \\\\
3.2 & 0.0 & 5.1 \\\\
0.0 & 0.0 & 2.9
\\end{bmatrix}`}</BlockMath>
          </section>
        </article>
      </main>
    </div>
  );
}
