import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Tag from '@/components/ui/Tag'
import Button from '@/components/ui/Button'
import SexToggle from '@/components/ui/SexToggle'

describe('Tag', () => {
  it('renders its children', () => {
    render(<Tag>FORECASTS</Tag>)
    expect(screen.getByText('FORECASTS')).toBeInTheDocument()
  })
})

describe('Button', () => {
  it('renders an anchor when given a href', () => {
    render(<Button href="/search">Search a Name</Button>)
    const link = screen.getByRole('link', { name: 'Search a Name' })
    expect(link).toHaveAttribute('href', '/search')
  })

  it('renders a button otherwise', () => {
    render(<Button>Go</Button>)
    expect(screen.getByRole('button', { name: 'Go' })).toBeInTheDocument()
  })
})

describe('SexToggle', () => {
  it('marks the selected option and reports changes', async () => {
    const onChange = vi.fn()
    render(<SexToggle value="F" onChange={onChange} />)

    expect(screen.getByRole('radio', { name: 'Female' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Male' })).toHaveAttribute('aria-checked', 'false')

    await userEvent.click(screen.getByRole('radio', { name: 'Male' }))
    expect(onChange).toHaveBeenCalledWith('M')
  })
})
