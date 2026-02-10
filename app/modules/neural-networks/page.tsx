import {
  BreadcrumbLink,
  BodyText,
  DisplayTitle,
  InlineLink,
  LeadText,
  SectionTitle,
} from "@/app/components/typography";
import TrainingProgressPanel from "./training-progress-panel";

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
            <SectionTitle>Tensors</SectionTitle>
            <BodyText>
              If you've ever looked up the definition of a tensor, you've
              probably seen something like this:
              <blockquote>
                A tensor is a generalization of vectors and matrices to
                potentially higher dimensions. In the context of machine
                learning, tensors are often used to represent multi-dimensional
                data, such as images, audio, and video.
              </blockquote>
              This is a good definition, but it's not very helpful. Let's try a
              different approach. A tensor is a generalization of vectors and
              matrices to potentially higher dimensions. In the context of
              machine learning, tensors are often used to represent
              multi-dimensional data, such as images, audio, and video.
            </BodyText>
          </section>
        </article>
      </main>
    </div>
  );
}
