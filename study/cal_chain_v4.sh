#!/usr/bin/env bash
# Plan v4: one supervisor process runs every seed under ONE global deadline, then the post-run probe.
# Usage: cal_chain_v4.sh <harness-commit> <bundle-sha256> <protocol-sha256> [--dry-run | <authorization-ref> <ownership-ref> <lease-valid-until>]
set -euo pipefail
shopt -s nullglob
umask 022
SCOPE=/ml2/altgan
S=/ml2/altgan/studies/gene-feature-20261014
PY=/ml2/altgan/.venv/bin/python
in_scope() { local r; r=$(realpath -e -- "$1") || return 1; [[ "$r" == "$SCOPE" || "$r" == "$SCOPE"/* ]] || { echo "out of scope: $1" >&2; return 1; }; printf '%s\n' "$r"; }
[[ $# -ge 4 ]] || { echo "usage: $0 COMMIT BUNDLE_SHA PROTOCOL_SHA --dry-run|AUTH OWN LEASE" >&2; exit 2; }
COMMIT=$1; BSHA=$2; PSHA=$3; shift 3
[[ -n "$COMMIT" && -n "$BSHA" && -n "$PSHA" ]] || { echo "empty argument" >&2; exit 2; }
SUP=$(in_scope "$S/supervise_v4.py")
BUNDLE=$(in_scope "$S/study_bundle_v2.pt")
cd -- "$(in_scope /ml2/altgan/.cache)"
export CUDA_VISIBLE_DEVICES= TMPDIR=/ml2/altgan/.cache XDG_CACHE_HOME=/ml2/altgan/.cache PYTHONDONTWRITEBYTECODE=1
echo "== v4 chain $(date --iso-8601=seconds) supervisor sha $(sha256sum -- "$SUP" | cut -c1-64)"
if [[ "$1" == "--dry-run" ]]; then
  OUT=$("$PY" "$SUP" --dry-run --seeds 42,43,44 --harness-commit "$COMMIT" --bundle "$BUNDLE" --bundle-sha256 "$BSHA" --protocol-sha256 "$PSHA") || RC=$?
else
  [[ $# -eq 3 ]] || { echo "need AUTH OWN LEASE" >&2; exit 2; }
  unset CUDA_VISIBLE_DEVICES
  OUT=$("$PY" "$SUP" --seeds 42,43,44 --harness-commit "$COMMIT" --bundle "$BUNDLE" --bundle-sha256 "$BSHA" --protocol-sha256 "$PSHA" \
        --authorization-reference "$1" --ownership-reference "$2" --lease-valid-until "$3") || RC=$?
fi
echo "supervisor rc=${RC:-0}"
printf '%s\n' "$OUT"
REC=$(printf '%s\n' "$OUT" | "$PY" -c 'import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1])["receipt"])')
REC=$(in_scope "$REC")
RSHA=$(sha256sum -- "$REC" | cut -c1-64)
"$PY" "$SUP" --post-run-probe --supervisor-receipt "$REC" --sha256 "$RSHA" || PRC=$?
echo "probe rc=${PRC:-0}"
FILES=("$(dirname -- "$REC")"/*.json)
[[ ${#FILES[@]} -gt 0 ]] || { echo "no receipts found" >&2; exit 3; }
sha256sum -- "${FILES[@]}"
echo CHAIN_DONE
