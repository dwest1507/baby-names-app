#!/usr/bin/env bash
#
# Vercel "Ignored Build Step" for the frontend project.
#
# Vercel's own semantics for this hook are the inverse of a normal shell
# script's: exit code 0 means SKIP the build, and any nonzero exit code means
# PROCEED with it (https://vercel.com/docs/project-configuration#ignoredbuildstep,
# checked at time of writing — reconfirm if Vercel's docs move). That is
# fortunate here, because `git diff --quiet` already exits 0 when there is no
# difference and 1 when there is one, so this script needs no inversion: its
# exit code IS the diff's exit code, once the pathological case below (no
# parent commit) is handled.
#
# Wired up by `ignoreCommand` in frontend/vercel.json, so no dashboard setting
# is needed; the equivalent manual control, if you ever want to inspect or
# override it, is Settings -> Build and Deployment -> Ignored Build Step. Either
# way the project's Root Directory must be `frontend`. Whether Vercel executes
# that command from the repo root or from Root Directory is not something we
# could verify without a live Vercel project (no credentials in this
# environment) — this script resolves the repo root itself via
# `git rev-parse --show-toplevel` so it behaves the same either way. Verify
# once against a real deploy log which cwd it actually ran from.
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

# Pick the base commit to diff against. Vercel sets VERCEL_GIT_PREVIOUS_SHA to
# the commit of the last deployment for this project, which is the correct base:
# diffing HEAD^..HEAD only inspects the tip commit, so a push of several commits
# that touched frontend/ in an earlier commit but not the tip would skip a build
# that was needed. The clone Vercel gives us is shallow, though, so that commit
# is often not in the local object graph -- hence the cat-file check and the
# HEAD^ fallback, which restores the old (tip-only) behaviour rather than
# failing. That leaves the multi-commit case open on shallow clones; closing it
# fully would mean deepening the fetch on every build.
#
# If VERCEL_GIT_PREVIOUS_SHA is unset or empty, this is the initial deployment of
# the project on Vercel: always proceed with the build.
if [ -z "${VERCEL_GIT_PREVIOUS_SHA:-}" ]; then
  echo "vercel-ignore-build: initial deployment (no previous SHA); proceeding with the build."
  exit 1
fi

base=""
if git cat-file -e "${VERCEL_GIT_PREVIOUS_SHA}^{commit}" 2>/dev/null; then
  base="$VERCEL_GIT_PREVIOUS_SHA"
elif git rev-parse HEAD^ >/dev/null 2>&1; then
  base=$(git rev-parse HEAD^)
else
  # No base at all -- e.g. the very first push to a fresh history, or a shallow
  # clone with depth 1 that Vercel hasn't deepened. Nothing to diff against, so
  # err toward building rather than silently skipping forever.
  echo "vercel-ignore-build: no base commit to diff against; building."
  exit 1
fi

if git diff --quiet "$base" HEAD -- frontend; then
  echo "vercel-ignore-build: no changes under frontend/ since $base; skipping the build."
  exit 0
else
  echo "vercel-ignore-build: changes detected under frontend/ since $base; proceeding with the build."
  exit 1
fi
