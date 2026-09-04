// @vitest-environment node
import { NextRequest } from 'next/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GET, POST } from '../app/api/[...path]/route'

const SECRET = 'test-shared-secret'
const BACKEND = 'http://backend.internal:8000'

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  vi.stubEnv('BACKEND_SHARED_SECRET', SECRET)
  vi.stubEnv('NAMES_API_URL', BACKEND)
  fetchMock = vi.fn(
    async () =>
      new Response('{"ok":true}', {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
  )
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

function call(
  method: 'GET' | 'POST',
  path: string,
  { headers = {}, search = '' }: { headers?: Record<string, string>; search?: string } = {}
) {
  const request = new NextRequest(`http://localhost:3000/api/${path}${search}`, {
    method,
    headers,
    body: method === 'POST' ? '{"message":"hi"}' : undefined,
  })
  const params = { params: Promise.resolve({ path: path.split('/') }) }
  return method === 'GET' ? GET(request, params) : POST(request, params)
}

function backendCallInit(): RequestInit {
  return fetchMock.mock.calls[0][1] as RequestInit
}

describe('the API proxy', () => {
  it('attaches the shared secret to the backend call', async () => {
    await call('GET', 'meta')

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(new Headers(backendCallInit().headers).get('x-backend-secret')).toBe(SECRET)
  })

  it("forwards the visitor's address, so the backend limits per person", async () => {
    await call('GET', 'meta', { headers: { 'x-forwarded-for': '203.0.113.10' } })

    expect(new Headers(backendCallInit().headers).get('x-forwarded-for')).toBe('203.0.113.10')
  })

  it('prefers the address the platform set over one the visitor claims', async () => {
    // A visitor can send any x-forwarded-for they like; the platform's own
    // x-real-ip is the one it observed.
    await call('GET', 'meta', {
      headers: { 'x-forwarded-for': '9.9.9.9, 203.0.113.10', 'x-real-ip': '203.0.113.10' },
    })

    expect(new Headers(backendCallInit().headers).get('x-forwarded-for')).toBe('203.0.113.10')
  })

  it('attaches the secret to POST calls too, and still forwards the body', async () => {
    await call('POST', 'chat')

    const init = backendCallInit()
    expect(new Headers(init.headers).get('x-backend-secret')).toBe(SECRET)
    expect(init.body).toBe('{"message":"hi"}')
  })

  it('never lets the secret reach the browser', async () => {
    // Even if the backend were to echo it back.
    fetchMock.mockResolvedValueOnce(
      new Response('{"ok":true}', {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'X-Backend-Secret': SECRET },
      })
    )

    const response = await call('GET', 'meta')

    const returned = [...response.headers.entries()].flat().join(' ')
    expect(returned).not.toContain(SECRET)
    expect(await response.text()).not.toContain(SECRET)
  })

  it('rejects a path the backend does not serve without calling it', async () => {
    const response = await call('GET', 'admin/secrets')

    expect(response.status).toBe(404)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('sends no secret header when none is configured, so local dev works', async () => {
    vi.stubEnv('BACKEND_SHARED_SECRET', '')

    await call('GET', 'meta')

    expect(new Headers(backendCallInit().headers).has('x-backend-secret')).toBe(false)
  })
})
