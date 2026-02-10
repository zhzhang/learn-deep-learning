import Link from "next/link";

export default function Home() {
  const modules = [
    {
      title: "Neural Networks",
      href: "/modules/neural-networks",
    },
    { title: "Initialization", href: "/modules/initialization" },
    { title: "Batch Normalization", href: "/modules/batch-normalization" },
    { title: "Momentum", href: "/modules/momentum" },
  ];

  return (
    <div className="flex min-h-screen items-center px-6 py-12 font-sans text-[var(--foreground)]">
      <main className="mx-auto w-full max-w-7xl">
        <section
          aria-label="Deep learning modules"
          className="mx-auto w-full max-w-7xl"
        >
          <div className="overflow-x-auto pb-4">
            <div className="mx-auto flex min-w-max items-center gap-5 px-2 sm:gap-7">
              <ModuleCard title={modules[0].title} href={modules[0].href} />
              <Connector />
              <ModuleCard title={modules[1].title} href={modules[1].href} />
              <Connector />
              <ModuleCard title={modules[2].title} href={modules[2].href} />
              <Connector />
              <ModuleCard title={modules[3].title} href={modules[3].href} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

type ModuleCardProps = {
  title: string;
  href: string;
  className?: string;
};

function ModuleCard({ title, href, className = "" }: ModuleCardProps) {
  return (
    <Link
      href={href}
      className={`group flex h-40 min-w-[22rem] items-center justify-center rounded-2xl border border-[var(--surface-border)] bg-[var(--surface)] px-8 text-center text-xl font-semibold text-[var(--foreground)] shadow-[0_2px_10px_rgba(47,35,24,0.06)] transition hover:-translate-y-0.5 hover:border-[var(--accent)] hover:shadow-[0_8px_22px_rgba(47,35,24,0.12)] ${className}`}
    >
      <span>{title}</span>
    </Link>
  );
}

function Connector() {
  return (
    <div
      aria-hidden="true"
      className="flex items-center gap-2 text-[color-mix(in_oklab,var(--accent),white_45%)]"
    >
      <span className="h-px w-10 bg-current sm:w-14" />
      <span className="h-2.5 w-2.5 rounded-full bg-current" />
      <span className="h-px w-10 bg-current sm:w-14" />
    </div>
  );
}
