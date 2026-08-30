export function formatPercent(fraction: number, digits = 3): string {
  return `${(fraction * 100).toFixed(digits)}%`
}

export function formatCount(count: number): string {
  return count.toLocaleString('en-US')
}

export function formatRank(rank: number): string {
  const mod100 = rank % 100
  if (mod100 >= 11 && mod100 <= 13) {
    return `${rank}th`
  }

  switch (rank % 10) {
    case 1:
      return `${rank}st`
    case 2:
      return `${rank}nd`
    case 3:
      return `${rank}rd`
    default:
      return `${rank}th`
  }
}
