import { ComponentPropsWithoutRef } from "react";
import Link from "next/link";

type DisplayTitleProps = Omit<ComponentPropsWithoutRef<"h1">, "className">;
type SectionTitleProps = Omit<ComponentPropsWithoutRef<"h2">, "className">;
type ParagraphProps = Omit<ComponentPropsWithoutRef<"p">, "className">;
type TextLinkProps = Omit<ComponentPropsWithoutRef<typeof Link>, "className">;

export function DisplayTitle(props: DisplayTitleProps) {
  return (
    <h1
      className="font-[var(--font-display)] text-3xl font-semibold tracking-tight text-[var(--foreground)] sm:text-4xl"
      {...props}
    />
  );
}

export function SectionTitle(props: SectionTitleProps) {
  return (
    <h2
      className="font-[var(--font-display)] text-xl font-semibold text-[var(--foreground)]"
      {...props}
    />
  );
}

export function LeadText(props: ParagraphProps) {
  return <p className="mt-3 text-lg text-[var(--muted)]" {...props} />;
}

export function BodyText(props: ParagraphProps) {
  return <p className="mt-3 leading-relaxed text-[var(--foreground)]" {...props} />;
}

export function BreadcrumbLink(props: TextLinkProps) {
  return (
    <Link
      className="transition hover:text-[var(--accent)] focus:underline focus:outline-none"
      {...props}
    />
  );
}

export function InlineLink(props: TextLinkProps) {
  return (
    <Link
      className="text-[var(--accent)] transition hover:text-[var(--accent-strong)] focus:underline focus:outline-none"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    />
  );
}

export function PrimaryLink(props: TextLinkProps) {
  return (
    <Link
      className="text-[var(--accent)] transition hover:text-[var(--accent-strong)] focus:underline focus:outline-none"
      {...props}
    />
  );
}
