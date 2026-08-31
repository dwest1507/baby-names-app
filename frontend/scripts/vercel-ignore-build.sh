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
# Configure in the Vercel project as:
#   Settings -> Build and Deployment -> Ignored Build Step -> Custom command
#   -> ./scripts/vercel-ignore-build.sh
# with the project's Root Directory set to `frontend`. Whether Vercel executes
# that command from the repo root or from Root Directory is not something we
# could verify without a live Vercel project (no credentials in this
# environment) — this script resolves the repo root itself via
# `git rev-parse --show-toplevel` so it behaves the same either way. Verify
# once against a real deploy log which cwd it actually ran from.
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if ! git rev-parse HEAD^ >/dev/null 2>&1; then
  # No parent commit -- e.g. the very first push to a fresh history, or a
  # shallow clone with depth 1 that Vercel hasn't deepened. Nothing to diff
  # against, so err toward building rather than silently skipping forever.
  echo "vercel-ignore-build: no parent commit to diff against; building."
  exit 1
fi

if git diff --quiet HEAD^ HEAD -- frontend; then
  echo "vercel-ignore-build: no changes under frontend/; skipping the build."
  exit 0
else
  echo "vercel-ignore-build: changes detected under frontend/; proceeding with the build."
  exit 1
fi
