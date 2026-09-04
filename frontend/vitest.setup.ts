import '@testing-library/jest-dom'
import { vi } from 'vitest'

// jsdom does not implement IntersectionObserver (used by FadeIn).
// Fire the callback immediately as intersecting so wrapped content is visible.
class MockIntersectionObserver {
  private callback: IntersectionObserverCallback

  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback
  }

  observe(target: Element) {
    this.callback(
      [{ isIntersecting: true, target } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver
    )
  }

  unobserve() {}
  disconnect() {}
  takeRecords(): IntersectionObserverEntry[] {
    return []
  }
}

// Route-handler tests run in the node environment, where there is no DOM to
// patch and nothing below applies.
if (typeof window !== 'undefined') {
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver)

  // jsdom does not implement scrollIntoView (used by ChatbotWidget autoscroll)
  window.HTMLElement.prototype.scrollIntoView = vi.fn()
}
