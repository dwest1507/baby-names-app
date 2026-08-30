import { type ReactNode } from 'react'

interface NoticeProps {
  variant?: 'error' | 'info'
  children: ReactNode
}

export default function Notice({ variant = 'info', children }: NoticeProps) {
  const styles =
    variant === 'error'
      ? 'border-red-500/30 bg-red-500/[0.06] text-red-200'
      : 'border-[#0ea5e9]/30 bg-[#0ea5e9]/[0.06] text-[#bae6fd]'

  return (
    <div className={`rounded-xl border px-4 py-3 text-sm leading-relaxed ${styles}`} role="status">
      {children}
    </div>
  )
}
