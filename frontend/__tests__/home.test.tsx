/**
 * What the landing page claims the site does.
 *
 * The forecast the site ships is the pooled gradient-boosted model (ADR 0010),
 * not the per-name ARIMA fit it replaced. The search page was updated with the
 * model; this page advertises the same feature and was not, so it went on
 * naming a model that no longer produces anything here.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Hero from '@/components/home/Hero'

describe('Hero', () => {
  it('does not advertise a model the site no longer runs', () => {
    render(<Hero />)

    expect(document.body.textContent).not.toMatch(/ARIMA/)
  })

  it('names the span of data the database actually holds', () => {
    render(<Hero />)

    expect(screen.getByText(/1880/)).toHaveTextContent('2025')
  })
})
