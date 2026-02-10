import {
  BreadcrumbLink,
  BodyText,
  DisplayTitle,
  InlineLink,
  LeadText,
  SectionTitle,
} from "@/app/components/typography";

export const metadata = {
  title: "Neural Networks | Deep Learning Textbook",
  description: "Let's dive into the deep end and replicate Pytorch",
};

export default function NeuralNetworksPage() {
  return (
    <div className="min-h-screen px-6 py-12 font-sans text-[var(--foreground)]">
      <main className="mx-auto w-full max-w-3xl">
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

        <header className="mb-12">
          <DisplayTitle>Neural Networks and Backpropagation</DisplayTitle>
          <LeadText>
            Let&apos;s dive into the deep end and replicate Pytorch
          </LeadText>
        </header>

        <article className="space-y-12">
          <section>
            <BodyText>
              We&apos;re going to replicate, largely from scratch, enough of
              Pytorch&apos;s API to train the neural network that is in
              Pytorch&apos;s{" "}
              <InlineLink href="https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html">
                quickstart tutorial on Fashion MNIST
              </InlineLink>
              .
            </BodyText>
          </section>

          <section>
            <SectionTitle>The forward pass</SectionTitle>
            <BodyText>
              Given an input (e.g. a vector or an image flattened into numbers),
              we pass it through the first layer: multiply by a weight matrix,
              add biases, then apply an activation (e.g. ReLU). The result is
              the input to the next layer. Repeating this layer by layer until
              the final layer produces the network’s <strong>output</strong>—
              for example a class score or a regression value. This sequence is
              called the <strong>forward pass</strong>.
            </BodyText>
          </section>

          <section>
            <SectionTitle>Loss and the backward pass</SectionTitle>
            <BodyText>
              We compare the output to the desired target with a{" "}
              <strong>loss function</strong> (e.g. cross-entropy for
              classification, mean squared error for regression). To improve, we
              need to know how each weight and bias affects the loss.{" "}
              <strong>Backpropagation</strong> computes these gradients
              efficiently by applying the chain rule from the loss back through
              every layer. With the gradients in hand, we take a step in the
              opposite direction—<strong>gradient descent</strong>—to reduce the
              loss and, over many steps, fit the network to the data.
            </BodyText>
          </section>
        </article>
      </main>
    </div>
  );
}
