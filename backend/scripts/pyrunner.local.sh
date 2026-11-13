#!/usr/bin/env bash
set -Eeuo pipefail

log() { printf '[%s] [local] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
on_error() {
    local rc=$?
    log "ERROR: exit=${rc}; line=${BASH_LINENO[0]:-unknown}; command=${BASH_COMMAND}"
    exit "$rc"
}
trap on_error ERR

: "${APGCE_PYTHON_SCRIPT:?Set APGCE_PYTHON_SCRIPT}"
: "${APGCE_RESULT_PATH:?Set APGCE_RESULT_PATH}"

APGCE_EXECUTION_MODE="${APGCE_EXECUTION_MODE:-local}"
APGCE_PYTHON_EXECUTABLE="${APGCE_PYTHON_EXECUTABLE:-/workspace/shared_envs/apgce-torch/bin/python}"
APGCE_PYTHON_SCRIPT="$(realpath -- "${APGCE_PYTHON_SCRIPT}")"
APGCE_RESULT_PATH="$(realpath -m -- "${APGCE_RESULT_PATH}")"
APGCE_PROJECT_ROOT="${APGCE_PROJECT_ROOT:-$(cd -- "$(dirname -- "${APGCE_PYTHON_SCRIPT}")/.." && pwd)}"

log "================ APGCE Python job ================"
log "Mode            : ${APGCE_EXECUTION_MODE}"
log "Host            : $(hostname)"
log "User            : $(id -un) (uid=$(id -u), gid=$(id -g))"
log "Project root    : ${APGCE_PROJECT_ROOT}"
log "Python script   : ${APGCE_PYTHON_SCRIPT}"
log "Result path     : ${APGCE_RESULT_PATH}"
log "Processing job : ${APGCE_JOB_ID:-$(basename "$(dirname "${APGCE_RESULT_PATH}")")}"
log "Input manifest  : ${APGCE_INPUT_MANIFEST:-<none>}"
log "Python          : ${APGCE_PYTHON_EXECUTABLE}"
log "Started         : $(date --iso-8601=seconds)"
log "CUDA devices    : ${CUDA_VISIBLE_DEVICES:-<unset>}"

[[ -f "${APGCE_PYTHON_SCRIPT}" ]] || { log "ERROR: Python script not found"; exit 2; }
if [[ -n "${APGCE_INPUT_MANIFEST:-}" && ! -f "${APGCE_INPUT_MANIFEST}" ]]; then
    log "ERROR: Input manifest was specified but not found: ${APGCE_INPUT_MANIFEST}"
    exit 2
fi
[[ -x "${APGCE_PYTHON_EXECUTABLE}" ]] || { log "ERROR: Python interpreter is not executable"; exit 2; }
mkdir -p -- "$(dirname -- "${APGCE_RESULT_PATH}")"

log "Python version:"
"${APGCE_PYTHON_EXECUTABLE}" --version
"${APGCE_PYTHON_EXECUTABLE}" -c 'import os, sys; print("sys.executable =", sys.executable); print("sys.prefix =", sys.prefix); print("cwd =", os.getcwd())'
log "Script details: $(stat -c 'size=%s bytes; modified=%y' "${APGCE_PYTHON_SCRIPT}")"

cd -- "${APGCE_PROJECT_ROOT}"
if [[ -n "${APGCE_INPUT_MANIFEST:-}" ]]; then
    log "Running: ${APGCE_PYTHON_EXECUTABLE} ${APGCE_PYTHON_SCRIPT} --input-manifest ${APGCE_INPUT_MANIFEST}"
    set +e
    "${APGCE_PYTHON_EXECUTABLE}" "${APGCE_PYTHON_SCRIPT}" --input-manifest "${APGCE_INPUT_MANIFEST}"
    rc=$?
    set -e
else
    log "Running without an input manifest"
    set +e
    "${APGCE_PYTHON_EXECUTABLE}" "${APGCE_PYTHON_SCRIPT}"
    rc=$?
    set -e
fi
log "Finished       : $(date --iso-8601=seconds)"
log "Exit code      : ${rc}"
if [[ -f "${APGCE_RESULT_PATH}" ]]; then
    log "Result file    : $(stat -c 'size=%s bytes; modified=%y' "${APGCE_RESULT_PATH}")"
else
    log "WARNING: result file was not created: ${APGCE_RESULT_PATH}"
fi
exit "$rc"
