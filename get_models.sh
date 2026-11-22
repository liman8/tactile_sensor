#!/bin/bash
# Put the trained networks the demo needs in place and check them against MODELS.sha256.
#
#     bash get_models.sh                  # the models prezentacija/demo.sh uses (default set)
#     bash get_models.sh --all            # every model listed in MODELS.sha256
#     bash get_models.sh --from DIR|URL   # where the files come from (default: $MODELS_URL, see below)
#     bash get_models.sh --check          # only verify what is already there, download nothing
#
# A model that is already present with the right SHA-256 is kept (the demo's model is versioned in git, so on a
# fresh clone it is normally already there). A missing or different file is fetched from the source as the flat
# asset name <path with "/" replaced by "__">, e.g. results_centre__mini_robust__model_100k_seed0.pt, checked, and
# only then moved into place. Exit code 1 if any requested model is missing or does not match MODELS.sha256.
#
# Source: a GitHub Release of this repository (tag models-v1) was planned for this. It does NOT exist: the
# repository is public, and the run that added this script was told not to upload anything to a public
# repository. --from takes any folder or URL that holds the flat asset names (e.g. a copy made with
# `bash get_models.sh --export DIR`).
#     bash get_models.sh --export DIR     # write the requested models into DIR under their asset names
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
SRC=${MODELS_URL:-https://github.com/liman8/tactile_sensor/releases/download/models-v1}
DEMO_MODELS=(results_centre/mini_robust/model_100k_seed0.pt)      # NET in prezentacija/demo.sh, FINAL_MODEL.md
MODE=get; ALL=0; EXPORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --all) ALL=1 ;;
    --from) SRC=$2; shift ;;
    --check) MODE=check ;;
    --export) MODE=export; EXPORT=$2; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -f MODELS.sha256 ] || { echo "MODELS.sha256 not found" >&2; exit 1; }
if [ $ALL = 1 ]; then
  mapfile -t LIST < <(awk '{print $2}' MODELS.sha256 | sort -u)
else
  LIST=("${DEMO_MODELS[@]}")
fi
want() { awk -v p="$1" '$2 == p {print $1; exit}' MODELS.sha256; }
have() { [ -f "$1" ] && sha256sum "$1" | awk '{print $1}'; }
fetch() {      # fetch ASSET DEST
  case "$SRC" in
    http://*|https://*) curl -fL --retry 3 -o "$2" "$SRC/$1" ;;
    file://*) cp "${SRC#file://}/$1" "$2" ;;
    *) cp "$SRC/$1" "$2" ;;
  esac
}
bad=0
for p in "${LIST[@]}"; do
  h=$(want "$p")
  [ -n "$h" ] || { echo "MISSING FROM MODELS.sha256: $p"; bad=1; continue; }
  asset=${p//\//__}
  if [ "$MODE" = export ]; then
    mkdir -p "$EXPORT"
    if [ "$(have "$p")" = "$h" ]; then cp "$(readlink -f "$p")" "$EXPORT/$asset"; echo "exported  $p -> $EXPORT/$asset"
    else echo "NOT EXPORTED (missing or wrong hash): $p"; bad=1; fi
    continue
  fi
  if [ "$(have "$p")" = "$h" ]; then echo "ok        $p"; continue; fi
  if [ "$MODE" = check ]; then echo "BAD       $p ($( [ -f "$p" ] && echo 'hash differs' || echo missing))"; bad=1; continue; fi
  mkdir -p "$(dirname "$p")"
  tmp="$p.download"
  if ! fetch "$asset" "$tmp"; then echo "FAILED    $p (could not fetch $asset from $SRC)"; rm -f "$tmp"; bad=1; continue; fi
  if [ "$(sha256sum "$tmp" | awk '{print $1}')" != "$h" ]; then echo "FAILED    $p (downloaded file has the wrong SHA-256)"; rm -f "$tmp"; bad=1; continue; fi
  mv "$tmp" "$p"; echo "fetched   $p"
done
exit $bad
