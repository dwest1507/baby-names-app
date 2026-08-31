#!/usr/bin/env bash
#
# Deploy wizard: walks a human through the one-time manual setup for this
# app's deployment (Hugging Face dataset, Railway project, Vercel project,
# optional Sentry project, branch protection) and verifies what can be
# checked from outside each platform. See docs/deployment.md for the full
# picture; this script is the interactive companion to it.
#
# Adapted from the mattpocock-skills `wizard` template
# (skills/engineering/wizard/template.sh). That template's library also
# offers write_env/set_secret/set_var to persist captured values to a local
# .env file and to GitHub Actions secrets via `gh`. This wizard deliberately
# does NOT use them: issue #12 requires that re-running this script against a
# working deployment be provably side-effect-free -- it writes nothing to
# disk and nothing to either platform. Every value it needs (the shared
# secret, service URLs) lives only in this process's shell variables for the
# duration of the run. If you want to skip retyping URLs on a re-run, export
# them in your own shell first (see "Re-running" below) -- the wizard reads
# them as defaults but never writes them anywhere itself.
#
# Usage:
#   ./scripts/deploy-wizard.sh
#
# Re-running (e.g. to confirm a deployment is still healthy):
#   RAILWAY_URL=https://your-service.up.railway.app \
#   VERCEL_URL=https://your-app.vercel.app \
#   HF_REPO=you/baby-names-db \
#     ./scripts/deploy-wizard.sh
#   With all three set, you can Enter-through every prompt and the wizard
#   becomes a pure health check.
#
# This script never calls Vercel's, Railway's, Hugging Face's, or GitHub's
# write APIs. It opens dashboard URLs for you to act on by hand, and it makes
# read-only, unauthenticated HTTP requests (health checks, a HEAD request
# against a public dataset file) plus, if `gh` is installed and already
# authenticated as you, read-only `gh api`/`gh repo view` calls.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────
# Wizard library (adapted from mattpocock-skills/wizard/template.sh -- see
# the note above for what was deliberately removed and why).
# ──────────────────────────────────────────────────────────────────────────

if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  BLUE=$(tput setaf 4); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); RED=$(tput setaf 1)
else
  BOLD=""; DIM=""; RESET=""; BLUE=""; GREEN=""; YELLOW=""; RED=""
fi

TOTAL_STAGES=7
_STAGE_INDEX=0
CHECKED=()  # human-readable descriptions of checks that passed
FAILED=()   # checks that were attempted and failed
SKIPPED=()  # checks that could not be attempted at all (e.g. no URL given yet)

_clear() {
  [[ -t 1 ]] || return 0
  if command -v tput >/dev/null 2>&1; then tput clear; else printf '\033[2J\033[3J\033[H'; fi
}

banner() {
  _clear
  printf '\n%s%s  %s%s\n' "$BOLD" "$BLUE" "$1" "$RESET"
  printf '%s  %s stages · writes nothing to disk or to any platform%s\n\n' "$DIM" "$TOTAL_STAGES" "$RESET"
  printf '%s  You drive the browser and the dashboards; this wizard tells you exactly\n' "$DIM"
  printf '  what to do, and verifies what it can from the outside once you tell it\n'
  printf '  a value (a URL, a repo id). Stop any time with Ctrl-C -- nothing is\n'
  printf '  half-written, because nothing is written at all.%s\n' "$RESET"
  pause "Ready to start?"
}

stage() {
  _clear
  _STAGE_INDEX=$((_STAGE_INDEX + 1))
  printf '\n%s%s▸ Stage %s/%s · %s%s\n\n' \
    "$BOLD" "$BLUE" "$_STAGE_INDEX" "$TOTAL_STAGES" "$1" "$RESET"
}

say()  { printf '  %s\n' "$1"; }
step() { printf '  %s•%s %s\n' "$BLUE" "$RESET" "$1"; }
note() { printf '  %s%s%s\n' "$DIM" "$1" "$RESET"; }
warn() { printf '  %s⚠ %s%s\n' "$YELLOW" "$1" "$RESET"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
fail() { printf '  %s✗%s %s\n' "$RED" "$RESET" "$1"; }

open_url() {
  local url="$1"
  printf '  %s↗ opening%s %s\n' "$GREEN" "$RESET" "$url"
  { if   command -v wslview     >/dev/null 2>&1; then wslview "$url"
    elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$url"
    elif command -v xdg-open    >/dev/null 2>&1; then xdg-open "$url"
    elif command -v open        >/dev/null 2>&1; then open "$url"
    else warn "couldn't open a browser; visit it manually: $url"; fi
  } >/dev/null 2>&1 || warn "couldn't open a browser, so visit it manually: $url"
}

pause() {
  printf '  %s%s%s ' "$DIM" "${1:-Press Enter to continue}" "$RESET"
  read -r _ || true
}

confirm() {
  local reply=""
  printf '  %s? %s [y/N] ' "$YELLOW" "$1"
  read -r reply || true
  [[ "$reply" =~ ^[Yy] ]]
}

# ask KEY "Prompt" reads a value into $KEY. If a shell variable named KEY is
# already exported (because you set it yourself before running this script --
# see the re-running note at the top), it's offered as the default; Enter
# keeps it. Nothing is read from or written to any file.
ask() {
  local key="$1" prompt="$2" current="" input
  current="${!key:-}"
  if [[ -n "$current" ]]; then
    printf '  %s%s%s %s[Enter keeps "%s"]%s ' "$BOLD" "$prompt" "$RESET" "$DIM" "$current" "$RESET"
  else
    printf '  %s%s%s ' "$BOLD" "$prompt" "$RESET"
  fi
  read -r input || true
  [[ -z "$input" && -n "$current" ]] && input="$current"
  printf -v "$key" '%s' "$input"
}

# ask_secret KEY "Prompt" is like ask, but input is hidden (not echoed to the
# terminal/scrollback) -- used only for pasting an *existing* secret back in;
# a freshly generated secret is deliberately printed in full (see stage 2),
# since the human has to see it once to copy it somewhere durable.
ask_secret() {
  local key="$1" prompt="$2" input
  printf '  %s%s%s ' "$BOLD" "$prompt" "$RESET"
  read -rs input || true
  printf '\n'
  printf -v "$key" '%s' "$input"
}

# check_status_code URL DESC [expected_regex] -- unauthenticated GET, checks
# the HTTP status only. Records the result; never fails the script.
check_status_code() {
  local url="$1" desc="$2" expect="${3:-^[23]}" code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -L "$url" 2>/dev/null) || code="000"
  if [[ "$code" =~ $expect ]]; then
    ok "$desc -> HTTP $code"
    CHECKED+=("$desc: HTTP $code")
  else
    fail "$desc -> HTTP $code (expected $expect)"
    FAILED+=("$desc ($url): got HTTP $code")
  fi
}

# check_health URL DESC -- like check_status_code, but also greps the JSON
# body for {"status":"ok",...}, since a health endpoint can 200 while
# reporting a database problem in its body (see backend/app/routes/health.py).
check_health() {
  local url="$1" desc="$2" body
  body=$(curl -s --max-time 10 "$url" 2>/dev/null) || body=""
  if grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"' <<<"$body"; then
    ok "$desc healthy: $body"
    CHECKED+=("$desc: healthy")
  else
    fail "$desc did not report healthy. Response: ${body:-<no response -- unreachable or timed out>}"
    FAILED+=("$desc ($url): ${body:-no response}")
  fi
}

finish() {
  _clear
  printf '\n%s%s  Wizard run complete%s\n\n' "$BOLD" "$GREEN" "$RESET"
  if (( ${#CHECKED[@]} )); then
    printf '%sVerified from outside:%s\n' "$BOLD" "$RESET"
    for c in "${CHECKED[@]}"; do ok "$c"; done
    printf '\n'
  fi
  if (( ${#FAILED[@]} )); then
    printf '%s%sFailed checks:%s\n' "$BOLD" "$RED" "$RESET"
    for f in "${FAILED[@]}"; do fail "$f"; done
    printf '\n'
  fi
  if (( ${#SKIPPED[@]} )); then
    printf '%sCould not verify automatically:%s\n' "$BOLD" "$RESET"
    for s in "${SKIPPED[@]}"; do note "  - $s"; done
    printf '\n'
  fi
  note "Nothing was written to disk or to any platform during this run."
  if (( ${#FAILED[@]} == 0 )); then
    printf '%s%sEverything this wizard could check looks healthy.%s\n\n' "$BOLD" "$GREEN" "$RESET"
  else
    printf '%s%sSome checks failed -- see docs/deployment.md'"'"'s failure-symptom tables.%s\n\n' "$BOLD" "$YELLOW" "$RESET"
  fi
}

# ──────────────────────────────────────────────────────────────────────────
# STAGES
# ──────────────────────────────────────────────────────────────────────────

banner "Baby Names Explorer — deploy wizard"

# ── Stage 1: the database artifact on Hugging Face ─────────────────────────
stage "Database artifact → Hugging Face dataset"
say "The backend never builds or ships its own database; it downloads a published"
say "artifact at container-build time (docs/adr/0006). That artifact has to exist"
say "on Hugging Face before Railway can build a working image."
say ""
say "If you don't have a Hugging Face account yet:"
step "Create one, then create a token with WRITE access:"
open_url "https://huggingface.co/settings/tokens"
say ""
say "With HF_TOKEN exported in your shell, run these from the repo root, in order"
say "(the last one needs a repo id like \"yourname/baby-names-db\"):"
note "  make build-db"
note "  make precompute-forecasts   # slow against the real db -- see ADR 0004"
note "  make verify-db"
note "  HF_TOKEN=... make publish-db REPO=yourname/baby-names-db"
say ""
say "This wizard does not run those commands for you -- publishing needs a real"
say "write-scoped HF_TOKEN this environment was never handed, and precompute is a"
say "long-running batch you should kick off deliberately, not as a side effect of"
say "answering a prompt."
pause "Press Enter once you've published the dataset (or if it's already published)."

ask HF_REPO "Hugging Face dataset repo id (e.g. yourname/baby-names-db):"
if [[ -n "${HF_REPO:-}" ]]; then
  # Public dataset: a plain, unauthenticated HEAD is enough to prove it
  # resolves. This is the same file/path app/database.py and the Dockerfile
  # download at build time.
  check_status_code "https://huggingface.co/datasets/${HF_REPO}/resolve/main/names.db" \
    "Hugging Face dataset ${HF_REPO} (names.db)" "^(2|3)"
else
  SKIPPED+=("Hugging Face dataset reachability (no repo id given)")
fi

# ── Stage 2: the shared secret ──────────────────────────────────────────────
stage "Generate the shared secret"
say "This one value gates every backend endpoint but /api/health (docs/adr/0002)."
say "It goes in exactly two places -- a Railway env var and a Vercel env var --"
say "and drift between them fails every request without looking like an auth"
say "problem, because both sides fail the same hmac.compare_digest check."
say ""
warn "This wizard does not save this value anywhere. Copy it now into a password"
warn "manager or note before continuing -- it scrolls away with the terminal."
say ""
if confirm "Generate a new secret with openssl rand -hex 32?"; then
  SHARED_SECRET=$(openssl rand -hex 32)
else
  ask_secret SHARED_SECRET "Paste the existing shared secret to verify against instead:"
fi
say ""
printf '  %s%s%s\n' "$BOLD" "$SHARED_SECRET" "$RESET"
say ""
say "Set this as BACKEND_SHARED_SECRET in:"
step "Railway → your service → Variables"
step "Vercel  → your project → Settings → Environment Variables (all environments —"
say "           previews share the production backend and secret, by design)"
pause "Press Enter once it's set in both places (or already was)."

# ── Stage 3: Railway (backend) ──────────────────────────────────────────────
stage "Railway: backend service"
say "New project, from this GitHub repo, in a fresh Railway project on your"
say "existing account (not reusing another service's project)."
open_url "https://railway.app/new"
step "Deploy from GitHub repo → select this repository."
step "Settings → Source → Root Directory: backend"
step "Settings → Build → Builder: the repo ships backend/railway.json with"
say "           \"builder\": \"DOCKERFILE\" — confirm Railway actually picked it up"
say "           rather than silently defaulting to Nixpacks (the exact trap ADR"
say "           0006 and docs/deployment.md warn about: a healthy-looking build"
say "           of the wrong thing)."
step "Settings → Deploy → Healthcheck Path: /api/health"
step "Settings → Deploy → confirm scale-to-zero / sleep is enabled (railway.json"
say "           sets sleepApplication — verify against the current dashboard"
say "           wording, which has moved before)."
step "Variables → set at least: BACKEND_SHARED_SECRET (from stage 2), APP_ENV=production,"
say "           GROQ_API_KEY (for the chatbot), and SENTRY_DSN once stage 5 gives you one."
say "           NAMES_DB_PATH does not need setting — the Dockerfile bakes the database"
say "           in at a fixed path and the image already sets it."
step "Trigger a deploy and wait for it to go live."
pause "Press Enter once the Railway service is up."

ask RAILWAY_URL "Railway service URL (e.g. https://your-service.up.railway.app):"
if [[ -n "${RAILWAY_URL:-}" ]]; then
  check_health "${RAILWAY_URL%/}/api/health" "Railway backend health"
else
  SKIPPED+=("Railway health check (no URL given)")
fi

# ── Stage 4: Vercel (frontend) ──────────────────────────────────────────────
stage "Vercel: frontend project"
say "New project, from the same GitHub repo."
open_url "https://vercel.com/new"
step "Import this repository."
step "Root Directory: frontend"
step "Framework Preset: Next.js — this is a dashboard setting, not something"
say "           frontend/vercel.json can express; get it wrong and every route"
say "           404s behind an otherwise-clean build log (the second named trap)."
step "The repo ships frontend/vercel.json with an ignoreCommand pointing at"
say "           frontend/scripts/vercel-ignore-build.sh, so a backend-only push"
say "           should already skip rebuilding this project — nothing to configure"
say "           by hand here unless you want to double check it under Settings →"
say "           Build and Deployment → Ignored Build Step."
step "Environment Variables (all environments) →"
say "           NAMES_API_URL = ${RAILWAY_URL:-<your Railway URL from stage 3>}"
say "           BACKEND_SHARED_SECRET = <the secret from stage 2>"
step "Deploy and wait for it to go live."
pause "Press Enter once the Vercel deployment is up."

ask VERCEL_URL "Vercel deployment URL (e.g. https://your-app.vercel.app):"
if [[ -n "${VERCEL_URL:-}" ]]; then
  check_status_code "${VERCEL_URL%/}/" "Vercel homepage" "^2"
  check_health "${VERCEL_URL%/}/api/health" "Vercel → proxy → Railway backend health"
else
  SKIPPED+=("Vercel checks (no URL given)")
fi

# ── Stage 5: Sentry (optional) ──────────────────────────────────────────────
stage "Sentry (optional): backend error reporting"
say "Skippable. Without a DSN, app/sentry.py is a complete no-op — nothing is"
say "reported locally or in CI either way."
if confirm "Set up Sentry now?"; then
  open_url "https://sentry.io/organizations/new/"
  step "Create a project — platform: Python → FastAPI."
  step "Copy the DSN it gives you."
  step "Set it as SENTRY_DSN in Railway → your service → Variables, then redeploy"
  say "           so the running container picks it up."
  pause "Press Enter once SENTRY_DSN is set on Railway."
  note "No automated check here: proving delivery means triggering and observing a"
  note "real error in the Sentry UI, which this wizard won't do on your behalf."
  SKIPPED+=("Sentry event delivery (verify manually in the Sentry dashboard)")
else
  note "Skipped. Revisit any time — see docs/deployment.md."
fi

# ── Stage 6: branch protection ──────────────────────────────────────────────
stage "GitHub: protect main behind CI"
say "main must be protected so a red build can't reach production through either"
say "platform's push-triggered deploy. Required check names (verbatim, copy-paste"
say "these into the required-checks search box) are listed in docs/ci-cd.md:"
say ""
note "  Ruff lint & format · Pytest · Lint, format & types · Unit tests ·"
note "  Production build · Lighthouse CI · CodeQL (javascript-typescript) ·"
note "  CodeQL (python) · Secret scanning (gitleaks) · npm audit (frontend) ·"
note "  pip-audit (backend)"
note "  (\"Dependency review\" runs pull_request-only and cannot be required on push"
note "  — see docs/ci-cd.md.)"
say ""
REPO_SLUG=$(git config --get remote.origin.url 2>/dev/null \
  | sed -E 's#^(https?://github\.com/|git@github\.com:)##; s#\.git$##') || REPO_SLUG=""
open_url "https://github.com/${REPO_SLUG:-dwest1507/baby-names-app}/settings/branches"
step "Add a branch protection rule for main."
step "Require status checks to pass before merging, and add each check name above."
step "Consider also requiring branches be up to date, and applying this to admins."
pause "Press Enter once branch protection is configured."

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  slug="${REPO_SLUG:-$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)}"
  if [[ -n "$slug" ]] && contexts=$(gh api "repos/${slug}/branches/main/protection/required_status_checks/contexts" 2>/dev/null); then
    ok "main has branch protection with required checks:"
    (jq -r '.[]' <<<"$contexts" 2>/dev/null || cat <<<"$contexts") | sed 's/^/    - /'
    CHECKED+=("Branch protection on main is configured")
  else
    fail "couldn't confirm branch protection on main via gh api (not configured, or gh lacks permission to read it)."
    FAILED+=("Branch protection on main")
  fi
else
  note "gh CLI not available/authenticated here; skipping the automated check."
  SKIPPED+=("Branch protection check (gh not installed or not authenticated) -- verify manually at the URL above")
fi

# ── Stage 7: close the loop ─────────────────────────────────────────────────
stage "Close the loop: end-to-end smoke test"
say "Everything platform-side has been checked from outside where possible. The"
say "one thing only a human clicking through a browser can confirm:"
step "Open the Vercel URL, search for a name, confirm the history chart and"
say "           forecast load."
step "If GROQ_API_KEY is set on Railway, try the chatbot with a real question."
step "Open a small PR that only touches frontend/ or only backend/ and confirm"
say "           (in each platform's dashboard) that only the matching service"
say "           rebuilds — this is the concurrency PR that actually proves the"
say "           ignore-build script and railway.json watch paths, not just that"
say "           they parsed."
say ""
note "Re-run this whole script any time — with RAILWAY_URL/VERCEL_URL/HF_REPO"
note "exported, it's a pure health check end to end."

finish
