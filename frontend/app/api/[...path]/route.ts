import type { NextRequest } from 'next/server'

// Proxy all /api/* calls to the FastAPI backend so the browser only ever
// talks to the Next.js origin (no CORS, backend URL stays server-side).

// Read at request time, not module load, so the running server always sees the
// current environment. Neither of these has a NEXT_PUBLIC_ prefix, so neither is
// inlined into the client bundle: the browser learns neither the backend's URL
// nor the secret.
const backendUrl = () => process.env.NAMES_API_URL ?? 'http://localhost:8000'
const sharedSecret = () => process.env.BACKEND_SHARED_SECRET ?? ''

// Only routes the backend actually serves are forwarded.
const ALLOWED_GET = /^(health|meta|top-names|names\/[^/]+(\/forecast)?)$/
const ALLOWED_POST = /^chat$/

// Next.js removed `request.ip`; on Vercel the visitor's address arrives in these
// headers. The backend keys its rate limits on what we send here.
//
// x-real-ip is set by the platform and holds a single address, so it is preferred
// over x-forwarded-for, whose leading entries a visitor can write themselves.
function visitorAddress(request: NextRequest): string {
  const realIp = request.headers.get('x-real-ip')?.trim()
  if (realIp) return realIp
  return (request.headers.get('x-forwarded-for') ?? '').split(',')[0].trim()
}

async function proxy(request: NextRequest, path: string[], method: 'GET' | 'POST') {
  const joined = path.join('/')
  const allowed = method === 'GET' ? ALLOWED_GET : ALLOWED_POST
  if (!allowed.test(joined)) {
    return Response.json({ detail: 'Not found' }, { status: 404 })
  }

  const search = request.nextUrl.search
  const url = `${backendUrl()}/api/${joined}${search}`

  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const secret = sharedSecret()
  if (secret) headers['X-Backend-Secret'] = secret
  const visitor = visitorAddress(request)
  if (visitor) headers['X-Forwarded-For'] = visitor

  let backendResponse: Response
  try {
    backendResponse = await fetch(url, {
      method,
      headers,
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
