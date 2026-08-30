'use client'

interface SexToggleProps {
  value: 'M' | 'F'
  onChange: (value: 'M' | 'F') => void
}

const OPTIONS: { value: 'M' | 'F'; label: string }[] = [
  { value: 'F', label: 'Female' },
  { value: 'M', label: 'Male' },
]

export default function SexToggle({ value, onChange }: SexToggleProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Sex"
      className="inline-flex rounded-lg border border-white/[0.08] bg-white/[0.03] p-0.5"
    >
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          role="radio"
          aria-checked={value === option.value}
          onClick={() => onChange(option.value)}
          className={`rounded-md px-4 py-1.5 text-sm transition-all duration-150 ${
            value === option.value
              ? 'bg-[#0ea5e9]/15 text-[#38bdf8] shadow-[inset_0_0_0_1px_rgba(14,165,233,0.3)]'
              : 'text-[#8a8f98] hover:text-[#ededef]'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}
