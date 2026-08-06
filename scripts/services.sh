#!/usr/bin/env bash
#
# Background process manager for the web interface and the HTTP API.
#
#   scripts/services.sh start   [frontend|backend]
#   scripts/services.sh stop    [frontend|backend]
#   scripts/services.sh restart [frontend|backend]
#   scripts/services.sh status
#   scripts/services.sh logs    [frontend|backend] [-f]
#
# With no service named, every command applies to both.
#
# The whole contract is a directory of PID files, so `kill $(cat .run/frontend.pid)`
# is always a valid escape hatch.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_DIR="${PROJECT_ROOT}/.run"
LOG_DIR="${PROJECT_ROOT}/logs"

# Prefer the project's virtualenv, activated or not.
PYTHON="${PYTHON:-$([ -x venv/bin/python ] && echo "${PROJECT_ROOT}/venv/bin/python" || echo python3)}"

# How long to wait for a service to start listening before calling it failed.
READY_TIMEOUT_SECONDS=45

SERVICES=(frontend backend)

# --- Service definitions -------------------------------------------------------
# A third service means a case in each function below plus a name in SERVICES.

service_port() {
    case "$1" in
        frontend) echo "${FRONTEND_PORT:-8501}" ;;
        backend)  echo "${API_PORT:-8000}" ;;
        *) return 1 ;;
    esac
}

service_command() {
    case "$1" in
        # --server.headless suppresses the first-run email prompt, which would
        # otherwise block a detached start.
        frontend) echo "$PYTHON -m streamlit run nl2sql/ui/app.py \
--server.port $(service_port frontend) --server.headless true \
--browser.gatherUsageStats false" ;;
        backend)  echo "$PYTHON -m uvicorn nl2sql.api:app --host 127.0.0.1 --port $(service_port backend)" ;;
        *) return 1 ;;
    esac
}

service_label() {
    case "$1" in
        frontend) echo "web interface" ;;
        backend)  echo "HTTP API" ;;
        *) echo "$1" ;;
    esac
}

service_url() {
    case "$1" in
        frontend) echo "http://127.0.0.1:$(service_port frontend)" ;;
        backend)  echo "http://127.0.0.1:$(service_port backend)/docs" ;;
        *) return 1 ;;
    esac
}

# --- Helpers -------------------------------------------------------------------

pid_file() { echo "${RUN_DIR}/$1.pid"; }
log_file() { echo "${LOG_DIR}/$1.log"; }

# Print the PID of a running service, or nothing. Stale PID files are removed.
running_pid() {
    local service="$1" file pid
    file="$(pid_file "$service")"

    [ -f "$file" ] || return 0
    pid="$(cat "$file" 2>/dev/null || true)"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "$pid"
    else
        rm -f "$file"
    fi
}

# -sTCP:LISTEN only: without it an open browser tab counts as a port conflict.
port_pid() { lsof -ti ":$1" -sTCP:LISTEN 2>/dev/null | head -1 || true; }

# Detach a command from this shell. `setsid` on Linux, `nohup` on macOS.
spawn_detached() {
    local command="$1" log="$2"

    if command -v setsid >/dev/null 2>&1; then
        setsid $command >>"$log" 2>&1 </dev/null &
    else
        nohup $command >>"$log" 2>&1 </dev/null &
    fi
}

# Children first, so a supervisor cannot restart one after its parent is gone.
signal_tree() {
    local signal="$1" pid="$2"

    pkill "-$signal" -P "$pid" 2>/dev/null || true
    kill "-$signal" "$pid" 2>/dev/null || true
}

wait_until_listening() {
    local port="$1" waited=0
    while [ "$waited" -lt "$READY_TIMEOUT_SECONDS" ]; do
        [ -n "$(port_pid "$port")" ] && return 0
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

# Wait for a stopped process to actually exit, so a restart never races the port.
wait_until_gone() {
    local pid="$1" waited=0
    while [ "$waited" -lt 10 ]; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

# Space-separated so callers can iterate unquoted; macOS bash 3.2 has no mapfile.
resolve_services() {
    if [ "$#" -eq 0 ]; then
        echo "${SERVICES[*]}"
        return
    fi
    for name in "$@"; do
        if ! service_port "$name" >/dev/null 2>&1; then
            echo "Unknown service '$name'. Choose one of: ${SERVICES[*]}" >&2
            exit 1
        fi
    done
    echo "$@"
}

# --- Commands ------------------------------------------------------------------

# Padded to a fixed column so both services line up.
report() {
    local service="$1"
    shift
    printf '  %-9s %s\n' "$service" "$*"
}

report_error() {
    local service="$1"
    shift
    printf '  %-9s %s\n' "$service" "$*" >&2
}

start_service() {
    local service="$1" port pid log
    port="$(service_port "$service")"
    log="$(log_file "$service")"

    if pid="$(running_pid "$service")" && [ -n "$pid" ]; then
        report "$service" "already running (pid $pid)"
        return 0
    fi

    # A foreign process on the port would satisfy the readiness check.
    if [ -n "$(port_pid "$port")" ]; then
        report_error "$service" "port $port is in use by pid $(port_pid "$port") —"
        report_error "" "stop that process, or set $([ "$service" = frontend ] && echo FRONTEND_PORT || echo API_PORT)"
        return 1
    fi

    mkdir -p "$RUN_DIR" "$LOG_DIR"
    spawn_detached "$(service_command "$service")" "$log"
    echo $! >"$(pid_file "$service")"

    if wait_until_listening "$port"; then
        report "$service" "started (pid $(cat "$(pid_file "$service")")) -> $(service_url "$service")"
    else
        report_error "$service" "failed to start within ${READY_TIMEOUT_SECONDS}s — last lines of $log:"
        tail -n 15 "$log" >&2 || true
        rm -f "$(pid_file "$service")"
        return 1
    fi
}

stop_service() {
    local service="$1" pid
    pid="$(running_pid "$service")"

    if [ -z "$pid" ]; then
        report "$service" "not running"
        return 0
    fi

    signal_tree TERM "$pid"

    if ! wait_until_gone "$pid"; then
        signal_tree KILL "$pid"
    fi

    rm -f "$(pid_file "$service")"
    report "$service" "stopped"
}

show_status() {
    printf '  %-9s %-8s %-6s %-8s %s\n' SERVICE STATE PORT PID URL
    for service in "${SERVICES[@]}"; do
        local pid port state
        pid="$(running_pid "$service")"
        port="$(service_port "$service")"

        if [ -n "$pid" ]; then
            state="running"
        elif [ -n "$(port_pid "$port")" ]; then
            # Someone else is on the port; this is why the next start will fail.
            state="foreign"
            pid="$(port_pid "$port")"
        else
            state="stopped"
            pid="-"
        fi

        printf '  %-9s %-8s %-6s %-8s %s\n' \
            "$service" "$state" "$port" "$pid" "$(service_url "$service")"
    done
}

show_logs() {
    local follow="" named="" argument service files=()

    for argument in "$@"; do
        case "$argument" in
            -f|--follow) follow="yes" ;;
            *) named="$named $argument" ;;
        esac
    done

    mkdir -p "$LOG_DIR"
    for service in $(resolve_services $named); do
        local file
        file="$(log_file "$service")"
        [ -f "$file" ] || : >"$file"
        files+=("$file")
    done

    if [ -n "$follow" ]; then
        tail -n 40 -f "${files[@]}"
    else
        tail -n 60 "${files[@]}"
    fi
}

# --- Entry point ---------------------------------------------------------------

main() {
    local command="${1:-status}"
    shift || true

    local targets service
    case "$command" in
        start)
            targets="$(resolve_services "$@")"
            for service in $targets; do start_service "$service"; done
            ;;
        stop)
            targets="$(resolve_services "$@")"
            for service in $targets; do stop_service "$service"; done
            ;;
        restart)
            # All stopped before any is started, so no port is held over.
            targets="$(resolve_services "$@")"
            for service in $targets; do stop_service "$service"; done
            for service in $targets; do start_service "$service"; done
            ;;
        status) show_status ;;
        logs) show_logs "$@" ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs} [frontend|backend]" >&2
            exit 1
            ;;
    esac
}

main "$@"
