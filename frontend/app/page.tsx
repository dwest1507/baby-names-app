import Hero from '@/components/home/Hero'
import FeatureNav from '@/components/home/FeatureNav'

const STACK = ['Next.js', 'FastAPI', 'SQLite', 'LightGBM', 'Groq']

export default function Home() {
  return (
    /* The three entry points sit beside the headline rather than below the
       fold, so the first screen is already something to click, and the strip
       below is pinned to the bottom of it rather than floating mid-page. */
    <div className="flex flex-1 flex-col">
      <section className="relative z-10 flex flex-1 items-center py-16 lg:py-24">
        <div className="mx-auto grid w-full max-w-7xl items-center gap-12 px-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:gap-20">
          <Hero />
          <FeatureNav />
        </div>
      </section>

      <div className="relative z-10 border-t border-white/[0.06]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-6 py-5">
          <span className="font-mono text-[10px] tracking-widest text-[#8a8f98]/60">
            BUILT WITH
          </span>
          {STACK.map((tool) => (
            <span key={tool} className="font-mono text-[10px] tracking-widest text-[#8a8f98]">
              {tool}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
