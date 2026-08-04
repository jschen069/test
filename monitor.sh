#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
    find_container_creator.sh <pid>
    find_container_creator.sh --host-pid <host_pid>
    find_container_creator.sh --container-pid <container_pid>
    find_container_creator.sh --container <container_id_or_name>

Description:
    Finds the Docker container for a PID or container ID/name, prints container
    metadata, and searches host logs for the likely user that created the
    container.

    Bare <pid> tries host PID lookup first, then falls back to container-internal
    PID lookup using /proc/*/status NSpid mappings. This is useful for PIDs shown
    by npu-smi inside containers.

Notes:
    Docker does not reliably store the host username that created a container.
    This script can only infer the creator from audit, sudo, or auth logs if
    those logs exist on the host.
EOF
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command not found: $command_name" >&2
        exit 1
    fi
}

trim() {
    awk '{$1=$1; print}'
}

show_container_process_mapping() {
    local container_pid="$1"
    local host_pid="$2"

    cat <<EOF
Process mapping
---------------
Container-internal PID: $container_pid
Host PID: $host_pid
EOF
}

container_exists() {
    local container_ref="$1"
    docker inspect "$container_ref" >/dev/null 2>&1
}

find_container_by_pid() {
    local target_pid="$1"
    local container_id container_pid

    while IFS= read -r container_id; do
        [[ -n "$container_id" ]] || continue
        container_pid=$(docker inspect -f '{{.State.Pid}}' "$container_id" 2>/dev/null || true)
        if [[ "$container_pid" == "$target_pid" ]]; then
            echo "$container_id"
            return 0
        fi
    done < <(docker ps -aq)

    return 1
}

find_container_by_host_process_pid() {
    local host_pid="$1"
    local container_id
    local cgroup_file="/proc/$host_pid/cgroup"

    [[ -r "$cgroup_file" ]] || return 1

    while IFS= read -r container_id; do
        [[ -n "$container_id" ]] || continue
        if grep -q "$container_id" "$cgroup_file" 2>/dev/null; then
            echo "$container_id"
            return 0
        fi
    done < <(docker ps -q)

    return 1
}

find_host_processes_by_container_pid() {
    local target_pid="$1"
    local status_file host_pid last_nspid

    for status_file in /proc/[0-9]*/status; do
        [[ -r "$status_file" ]] || continue
        [[ -n "$(grep '^NSpid:' "$status_file" 2>/dev/null || true)" ]] || continue
        host_pid=$(basename "$(dirname "$status_file")")
        last_nspid=$(awk '/^NSpid:/ {print $NF}' "$status_file" 2>/dev/null || true)
        if [[ "$last_nspid" == "$target_pid" ]]; then
            echo "$host_pid"
        fi
    done
}

resolve_container_pid() {
    local target_pid="$1"
    local host_pid container_id

    while IFS= read -r host_pid; do
        [[ -n "$host_pid" ]] || continue
        if container_id=$(find_container_by_host_process_pid "$host_pid"); then
            echo "$container_id $host_pid"
            return 0
        fi
    done < <(find_host_processes_by_container_pid "$target_pid")

    return 1
}

to_local_time() {
    local timestamp="$1"
    date -d "$timestamp" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$timestamp"
}

shift_time() {
    local base_time="$1"
    local offset="$2"
    date -d "$base_time $offset" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$base_time"
}

show_container_metadata() {
    local container_ref="$1"
    local name image created_utc created_local status configured_user pid host_process_user

    name=$(docker inspect --format '{{.Name}}' "$container_ref" | sed 's#^/##')
    image=$(docker inspect --format '{{.Config.Image}}' "$container_ref")
    created_utc=$(docker inspect --format '{{.Created}}' "$container_ref")
    created_local=$(to_local_time "$created_utc")
    status=$(docker inspect --format '{{.State.Status}}' "$container_ref")
    configured_user=$(docker inspect --format '{{if .Config.User}}{{.Config.User}}{{else}}root (default){{end}}' "$container_ref")
    pid=$(docker inspect --format '{{.State.Pid}}' "$container_ref")
    host_process_user="N/A"

    if [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 0 )); then
        host_process_user=$(ps -o user= -p "$pid" 2>/dev/null | trim || true)
        host_process_user=${host_process_user:-unknown}
    fi

    cat <<EOF
Container metadata
------------------
Container: $name
Image: $image
Created (UTC): $created_utc
Created (local): $created_local
Status: $status
Configured container user: $configured_user
Host user owning container PID: $host_process_user
Host PID: $pid
EOF
}

search_journalctl() {
    local since_time="$1"
    local until_time="$2"

    if ! command -v journalctl >/dev/null 2>&1; then
        return 0
    fi

    echo
    echo "journalctl matches"
    echo "------------------"
    journalctl --since "$since_time" --until "$until_time" --no-pager 2>/dev/null \
        | grep -Ei 'docker|dockerd|containerd|sudo' || echo "No journalctl matches found."
}

search_auth_logs() {
    local since_time="$1"
    local until_time="$2"
    local log_file
    local found=0

    echo
    echo "Auth log matches"
    echo "----------------"

    for log_file in /var/log/auth.log /var/log/secure; do
        if [[ -r "$log_file" ]]; then
            found=1
            echo "Log file: $log_file"
            grep -Ei 'docker|dockerd|sudo' "$log_file" | tail -n 50 || true
            echo
        fi
    done

    if [[ $found -eq 0 ]]; then
        echo "No readable auth logs found."
    else
        echo "Auth logs are not time-filtered here; inspect entries around $since_time to $until_time."
    fi
}

search_auditd() {
    local since_time="$1"
    local until_time="$2"

    echo
    echo "auditd matches"
    echo "-------------"

    if ! command -v ausearch >/dev/null 2>&1; then
        echo "ausearch not available."
        return 0
    fi

    ausearch -m EXECVE -ts "$since_time" -te "$until_time" 2>/dev/null | grep -Ei 'docker|dockerd' \
        || echo "No auditd matches found."
}

main() {
    local container_ref=""
    local target_pid=""
    local target_host_pid=""
    local target_container_pid=""
    local resolved_host_pid=""
    local lookup_mode="auto"
    local created_utc created_local since_time until_time

    require_command docker
    require_command ps
    require_command date

    if [[ $# -lt 1 || $# -gt 2 ]]; then
        usage
        exit 1
    fi

    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --container)
            if [[ $# -ne 2 ]]; then
                usage
                exit 1
            fi
            container_ref="$2"
            if ! container_exists "$container_ref"; then
                echo "Error: container not found: $container_ref" >&2
                exit 1
            fi
            ;;
        --host-pid)
            if [[ $# -ne 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --host-pid requires a numeric PID" >&2
                exit 1
            fi
            lookup_mode="host"
            target_host_pid="$2"
            if ! container_ref=$(find_container_by_host_process_pid "$target_host_pid"); then
                echo "Error: no running Docker container found for host PID $target_host_pid" >&2
                exit 1
            fi
            resolved_host_pid="$target_host_pid"
            ;;
        --container-pid)
            if [[ $# -ne 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --container-pid requires a numeric PID" >&2
                exit 1
            fi
            lookup_mode="container"
            target_container_pid="$2"
            if ! read -r container_ref resolved_host_pid < <(resolve_container_pid "$target_container_pid"); then
                echo "Error: no running Docker container found for container-internal PID $target_container_pid" >&2
                exit 1
            fi
            ;;
        *)
            target_pid="$1"
            if [[ ! "$target_pid" =~ ^[0-9]+$ ]]; then
                echo "Error: pid must be numeric" >&2
                exit 1
            fi
            if container_ref=$(find_container_by_host_process_pid "$target_pid"); then
                lookup_mode="host"
                resolved_host_pid="$target_pid"
            elif read -r container_ref resolved_host_pid < <(resolve_container_pid "$target_pid"); then
                lookup_mode="container"
                target_container_pid="$target_pid"
            else
                echo "Error: no running Docker container found for PID $target_pid as either a host PID or a container-internal PID" >&2
                exit 1
            fi
            ;;
    esac

    if [[ "$lookup_mode" == "container" ]]; then
        show_container_process_mapping "$target_container_pid" "$resolved_host_pid"
        echo
    elif [[ "$lookup_mode" == "host" && -n "$resolved_host_pid" ]]; then
        show_container_process_mapping "N/A" "$resolved_host_pid"
        echo
    fi

    show_container_metadata "$container_ref"

    created_utc=$(docker inspect --format '{{.Created}}' "$container_ref")
    created_local=$(to_local_time "$created_utc")
    since_time=$(shift_time "$created_local" '-5 minutes')
    until_time=$(shift_time "$created_local" '+5 minutes')

    echo
    echo "Search window"
    echo "-------------"
    echo "Since: $since_time"
    echo "Until: $until_time"

    search_journalctl "$since_time" "$until_time"
    search_auth_logs "$since_time" "$until_time"
    search_auditd "$since_time" "$until_time"

    echo
    echo "Conclusion"
    echo "----------"
    echo "If the creator is not shown above, the host likely does not have enough audit or auth logging to recover it after the fact."
    echo "For future containers, add a creator label such as: --label creator=\"$(id -un)\""
}

main "$@"
