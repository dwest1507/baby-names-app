import { describe, expect, it } from 'vitest'
import { formatCount, formatPercent, formatRank } from '@/lib/format'

describe('formatPercent', () => {
  it('converts a fraction to a percent string', () => {
    expect(formatPercent(0.01234)).toBe('1.234%')
  })

  it('respects the digits argument', () => {
    expect(formatPercent(0.5, 1)).toBe('50.0%')
  })
})

describe('formatCount', () => {
  it('adds thousands separators', () => {
    expect(formatCount(1234567)).toBe('1,234,567')
  })
})

describe('formatRank', () => {
  it('handles standard ordinal suffixes', () => {
    expect(formatRank(1)).toBe('1st')
    expect(formatRank(2)).toBe('2nd')
    expect(formatRank(3)).toBe('3rd')
    expect(formatRank(4)).toBe('4th')
  })

  it('handles the teens', () => {
    expect(formatRank(11)).toBe('11th')
    expect(formatRank(12)).toBe('12th')
    expect(formatRank(13)).toBe('13th')
    expect(formatRank(111)).toBe('111th')
  })

  it('handles higher ranks', () => {
    expect(formatRank(21)).toBe('21st')
    expect(formatRank(102)).toBe('102nd')
  })
})
