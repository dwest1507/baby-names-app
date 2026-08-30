import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import Header from '@/components/layout/Header'
import Footer from '@/components/layout/Footer'
import AmbientBackground from '@/components/layout/AmbientBackground'
import './globals.css'

const inter = Inter({
  variable: '--font-inter',
  subsets: ['latin'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Baby Names Explorer',
  description:
    'Explore 145 years of baby name popularity from the Social Security Administration dataset — trends, ARIMA forecasts, and an AI chatbot that answers questions in natural language.',
  openGraph: {
    title: 'Baby Names Explorer',
    description:
      'Explore 145 years of baby name popularity — trends, ARIMA forecasts, and an AI chatbot.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="flex min-h-screen flex-col">
        <AmbientBackground />
        <Header />
        <main className="relative z-10 flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  )
}
