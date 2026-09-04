import Link from 'next/link'
import Hero from '@/components/home/Hero'
import Card from '@/components/ui/Card'
import FadeIn from '@/components/ui/FadeIn'
import Section from '@/components/layout/Section'
import Tag from '@/components/ui/Tag'

const FEATURES = [
  {
    href: '/explore',
    tag: 'RANKINGS',
    title: 'Top Names',
    description:
      'The most popular names for any year since 1880, filterable by sex, charted and tabulated.',
  },
  {
    href: '/search',
    tag: 'FORECASTS',
    title: 'Name Search',
    description:
      'Look up any name for its full popularity history, current rank, and a 5-year ARIMA forecast with confidence intervals and holdout validation.',
  },
  {
    href: '/chat',
    tag: 'AI CHAT',
    title: 'Ask in Plain English',
    description:
      'A chatbot that translates your question into SQL, runs it against the dataset, and explains the answer — with the query shown for transparency.',
  },
]

export default function Home() {
  return (
    <>
      <Hero />
      <Section>
        <div className="grid gap-6 md:grid-cols-3">
          {FEATURES.map(({ href, tag, title, description }, i) => (
            <FadeIn key={href} delay={i * 100}>
              <Link href={href} className="block h-full">
                <Card
                  variant="default"
                  spotlight
                  className="h-full p-6 transition-transform duration-200 hover:-translate-y-1"
                >
                  <Tag variant="accent">{tag}</Tag>
                  <h3 className="mt-4 text-lg font-semibold text-[#ededef]">{title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-[#8a8f98]">{description}</p>
                  <span className="mt-4 inline-flex items-center gap-1 text-sm text-[#0ea5e9]">
                    Open
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      className="h-4 w-4"
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M13.5 4.5 21 12l-7.5 7.5M21 12H3"
                      />
                    </svg>
                  </span>
                </Card>
              </Link>
            </FadeIn>
          ))}
        </div>
      </Section>
    </>
  )
}
