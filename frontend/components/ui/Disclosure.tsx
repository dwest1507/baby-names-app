'use client'

import { useId, useState, type ReactNode } from 'react'

interface DisclosureProps {
  summary: string
  children: ReactNode
}

/** A section collapsed until asked for. A real button, so it is reachable and
 *  toggled from the keyboard, announcing whether it is open via
 *  `aria-expanded`; collapsed content stays mounted but `hidden`, so the
 *  button's `aria-controls` always names an element that exists. */
export default function Disclosure({ summary, children }: DisclosureProps) {
  const [open, setOpen] = useState(false)
  const contentId = useId()

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-2 rounded-lg px-1 py-1 text-sm text-[#8a8f98] transition-colors duration-150 hover:text-[#ededef] focus-visible:shadow-[0_0_0_2px_rgba(14,165,233,0.5)] focus-visible:outline-none"
      >
        <span
          aria-hidden="true"
          className={`inline-block text-xs transition-transform duration-150 ${open ? 'rotate-90' : ''}`}
        >
          ▶
        </span>
        {summary}
      </button>
      <div id={contentId} hidden={!open} className="mt-4">
        {children}
      </div>
    </div>
  )
}
