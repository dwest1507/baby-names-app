# 2. A shared secret is the backend's gateway

Date: 2026-08-30

## Status

Accepted

## Context

Deployed, the backend has a public URL and no authentication. Its chat endpoint spends a
metered provider quota — two calls per turn, one to generate SQL and one to phrase the answer —
so anyone who finds the URL can spend real money on the maintainer's account.

The shape of the deployment makes two existing defences useless:

- **The browser never talks to the backend.** Every request arrives from the frontend's
  server-side proxy (`frontend/app/api/[...path]/route.ts`). No browser request reaches the
  backend, so the CORS origin list has nothing to allow or deny. CORS is a rule a *browser*
  enforces on behalf of a page; a server-to-server call, or `curl`, never consults it.
- **The rate limiter sees the proxy, not the visitor.** Because every request comes from the
  platform's rotating egress address, the per-address limit degenerates into one shared bucket
  for everybody: unfair lockouts and no real ceiling at the same time.

The visitor's real address is available to the proxy and can be forwarded in a header — but a
header anyone can set is not evidence of anything on a public endpoint.

## Decision

**A shared secret, presented by the frontend proxy on every backend call, is what guards the
backend. The rate limiter keys on the forwarded visitor address, and only after that secret has
been checked.**

- The secret is required **uniformly on every path except `/api/health`**, in one middleware
  (`main.require_shared_secret`), not per route. A per-route allow-list of exceptions rots: the
  cost of forgetting a decorator on a new route is an unguarded endpoint that looks fine. Health
  is open because the deployment platform probes it and cannot present a secret.
- The secret is compared with `hmac.compare_digest` and lives in `BACKEND_SHARED_SECRET` on both
  platforms. Drift between the two fails every request; rotation is a two-place change.
- On the frontend it is read **only in the server-side proxy**, from a variable with no
  `NEXT_PUBLIC_` prefix, so it is never inlined into the client bundle. The proxy returns a
  response built from the backend's status and body alone, so no backend header can carry it
  outward either. The same is true of the backend's URL.
- **The secret is validated before the forwarded address is trusted.** The middleware marks the
  request, and the limiter's key function (`limiter.visitor_address`) reads `X-Forwarded-For`
  only on a marked request, falling back to the direct peer address otherwise. Order is the
  whole mechanism: an unauthenticated request is rejected before it can name a bucket, so an
  outsider can neither escape their own limit nor exhaust somebody else's.
- The proxy prefers `x-real-ip` — set by the platform, single-valued — over `x-forwarded-for`,
  whose leading entries a visitor can write themselves.
- **Two tiers.** A general ceiling of 60 requests/minute and 1,000/hour per visitor, as one
  bucket shared across every endpoint rather than one per endpoint; and 5/minute and 50/day per
  visitor on `/api/chat`. The chat figure is arithmetic against the provider's free allowance of
  1,000 model requests per day: two calls per turn leaves ~500 turns/day for the whole site, so
  a 50-turn daily cap spends at most a tenth of it per visitor.
- **The CORS origin list stays** as defence in depth, with a comment at the point of
  configuration saying it is not the guard, so that nobody later trusts it to do this job.
- **Limiter storage stays in process.** No Redis, no durable store.

## Consequences

- The general ceiling is enforced in this repo's own middleware rather than through slowapi's
  application limits. slowapi's middleware finds a route's handler by scanning `app.routes`, and
  this version of FastAPI hides routers registered with `include_router` behind a wrapper object
  with no `endpoint` attribute; limits configured that way are silently never evaluated. This was
  measured, not assumed: 65 consecutive requests against a 60/minute application limit all
  returned 200. The per-route decorator (used for the chat tier) is unaffected.
- **In-process limiter state is lost on every scale-to-zero sleep.** A backend that wakes up
  starts with every bucket empty, so a determined visitor can reset their own allowance by
  waiting for the container to sleep. This is accepted: the durable alternative is a persistent
  store, whose always-on cost is the very thing scale-to-zero exists to avoid, and the daily
  chat cap is a guard against a quota bill rather than a security boundary.
- **An unset secret behaves differently by environment.** Locally it means "no secret required",
  so a fresh checkout runs `make dev` with no setup. Deployed — `APP_ENV=production`, or any of
  Railway's own injected variables — it means the guard is missing, and the backend refuses every
  request but health. Inferring production from the platform's variables as well as from
  `APP_ENV` means a forgotten setting cannot leave a deployed backend open. The trade-off is that
  a misconfigured deploy still passes its health probe and is promoted; it fails visibly at the
  first real request rather than at the container's front door.
- Health is also exempt from rate limiting. A throttled probe reads as a failed deploy.
- Preview deployments share the production backend and the same secret, so a preview is
  indistinguishable from production to the backend. That was a deliberate choice in the parent
  PRD; separate secrets remain deferred.
- The secret authenticates the *frontend*, not a visitor. Anyone who extracts it from the Vercel
  project's environment has full access; it is a gate against the open internet, not an identity
  system.

## Related

- Parent PRD: dwest1507/baby-names-app#5
- Implemented by: dwest1507/baby-names-app#7
