export default function Footer() {
  return (
    <footer className="relative z-10 mt-auto border-t border-white/[0.06] bg-[#020203]">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 py-8 sm:flex-row">
        <p className="text-xs text-[#8a8f98]">© {new Date().getFullYear()} Baby Names Explorer</p>

        <nav aria-label="Footer navigation" className="flex gap-6">
          <a
            href="https://www.ssa.gov/oact/babynames/limits.html"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-[#8a8f98] transition-colors duration-150 hover:text-[#ededef]"
          >
            Data: Social Security Administration
          </a>
        </nav>
      </div>
    </footer>
  )
}
