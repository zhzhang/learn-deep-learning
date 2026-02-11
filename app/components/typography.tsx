import { ComponentPropsWithoutRef } from "react";
import Link from "next/link";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { prism } from "react-syntax-highlighter/dist/esm/styles/prism";

type DisplayTitleProps = Omit<ComponentPropsWithoutRef<"h1">, "className">;
type SectionTitleProps = Omit<ComponentPropsWithoutRef<"h2">, "className">;
type ParagraphProps = Omit<ComponentPropsWithoutRef<"p">, "className">;
type CodeBlockProps = Omit<ComponentPropsWithoutRef<"pre">, "className"> & {
  language?: string;
};
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

export function CodeBlock({ children, language = "text", ...props }: CodeBlockProps) {
  return (
    <SyntaxHighlighter
      language={language}
      style={prism}
      customStyle={{
        marginTop: "1rem",
        borderRadius: "0.5rem",
        border: "1px solid var(--border)",
        background: "var(--surface)",
        color: "var(--foreground)",
        padding: "1rem",
        fontSize: "0.875rem",
        lineHeight: "1.5",
        overflowX: "auto",
      }}
      {...props}
    >
      {typeof children === "string" ? children.trim() : children}
    </SyntaxHighlighter>
  );
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
