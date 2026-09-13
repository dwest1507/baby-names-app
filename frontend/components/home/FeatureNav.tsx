import Link from 'next/link'
import Card from '@/components/ui/Card'
import FadeIn from '@/components/ui/FadeIn'
import Tag from '@/components/ui/Tag'

const FEATURES = [
  {
    href: '/explore',
    index: '01',
    tag: 'RANKINGS',
    title: 'Top Names',
    description: 'The most popular names for any year since 1880, by sex — charted and ranked.',
  },
  {
    href: '/search',
    index: '02',
    tag: 'FORECASTS',
    title: 'Name Search',
    description:
      "One name's full history and current rank, with a five-year forecast drawn as a conformal band.",
  },
  {
    href: '/chat',
    index: '03',
    tag: 'AI CHAT',
    title: 'Ask in Plain English',
    description:
      'Ask a question; the chatbot writes the SQL, runs it, and shows you both the answer and the query.',
  },
]

export default function FeatureNav() {
  return (
    <nav aria-label="Explore the data" className="flex flex-col gap-3">
      {FEATURES.map(({ href, index, tag, title, description }, i) => (
        <FadeIn key={href} delay={i * 90} distance={16}>
          <Link href={href} className="group block">
            <Card
              variant="default"
              spotlight
              className="p-5 transition-transform duration-200 group-hover:-translate-y-0.5 sm:p-6"
            >
              <div className="flex items-start gap-4 sm:gap-5">
                <span className="mt-0.5 shrink-0 font-mono text-[11px] tracking-widest text-[#0ea5e9]/60">
                  {index}
                </span>

                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <h2 className="text-base font-semibold text-[#ededef]">{title}</h2>
                    <Tag variant="muted">{tag}</Tag>
                  </div>
                  <p className="mt-1.5 text-sm leading-relaxed text-[#8a8f98]">{description}</p>
                </div>

                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  className="mt-1 h-4 w-4 shrink-0 text-[#8a8f98]/50 transition-all duration-200 group-hover:translate-x-1 group-hover:text-[#0ea5e9]"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M13.5 4.5 21 12l-7.5 7.5M21 12H3"
                  />
                </svg>
              </div>
            </Card>
          </Link>
        </FadeIn>
      ))}
    </nav>
  )
}
