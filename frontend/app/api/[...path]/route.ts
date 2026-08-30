import type { NextRequest } from 'next/server'

// Proxy all /api/* calls to the FastAPI backend so the browser only ever
// talks to the Next.js origin (no CORS, backend URL stays server-side).

const BACKEND_URL = process.env.NAMES_API_URL ?? 'http://localhost:8000'

// Only routes the backend actually serves are forwarded.
const ALLOWED_GET = /^(health|meta|top-names|names\/[^/]+(\/forecast)?)$/
const ALLOWED_POST = /^chat$/

async function proxy(request: NextRequest, path: string[], method: 'GET' | 'POST') {
  const joined = path.join('/')
  const allowed = method === 'GET' ? ALLOWED_GET : ALLOWED_POST
  if (!allowed.test(joined)) {
    return Response.json({ detail: 'Not found' }, { status: 404 })
  }

  const search = request.nextUrl.search
  const url = `${BACKEND_URL}/api/${joined}${search}`

  let backendResponse: Response
  try {
    backendResponse = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: method === 'POST' ? await request.text() : undefined,
      cache: 'no-store',
    })
  } catch {
    return Response.json(
      { detail: 'The API backend is unreachable. Is it running on port 8000?' },
      { status: 502 }
    )
  }

  const body = await backendResponse.text()
  return new Response(body, {
    status: backendResponse.status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params
  return proxy(request, path, 'GET')
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params
  return proxy(request, path, 'POST')
}
