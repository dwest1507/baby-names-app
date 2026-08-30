import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

const getMeta = vi.fn()
const getTopNames = vi.fn()

vi.mock('@/lib/api', async () => {
  const actual = await vi.importActual<typeof import('@/lib/api')>('@/lib/api')
  return {
    ...actual,
    getMeta: () => getMeta(),
    getTopNames: (...a: unknown[]) => getTopNames(...a),
  }
})

import ExplorePage from '@/app/explore/page'

describe('ExplorePage year input', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMeta.mockResolvedValue({ min_year: 1880, max_year: 2024 })
    getTopNames.mockResolvedValue({ names: [] })
  })

  it('never requests a year outside the dataset while the field is being edited', async () => {
    const user = userEvent.setup()
    render(<ExplorePage />)

    const input = await screen.findByLabelText<HTMLInputElement>('Year')
    await waitFor(() => expect(input.value).toBe('2024'))

    // Retyping a year one digit at a time must not fire 1, 19, 199, ...
    await user.clear(input)
    await user.type(input, '1990')

    const requestedYears = getTopNames.mock.calls.map((call) => call[1])
    expect(requestedYears.every((y) => y >= 1880 && y <= 2024)).toBe(true)
    await waitFor(() => expect(requestedYears).toContain(1990))
  })

  it('restores the last valid year when the field is left empty', async () => {
    const user = userEvent.setup()
    render(<ExplorePage />)

    const input = await screen.findByLabelText<HTMLInputElement>('Year')
    await waitFor(() => expect(input.value).toBe('2024'))

    await user.clear(input)
    expect(input.value).toBe('')
    input.blur()
    await waitFor(() => expect(input.value).toBe('2024'))
  })
})
