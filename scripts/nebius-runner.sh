#!/usr/bin/env bash
### LLM-generated ###
# Cargo target runner that executes binaries on the first running instance of
# the active Nebius profile. Invoked by cargo as: nebius-runner.sh <binary> [args...]
#
# Optional overrides:
#   NEBIUS_RUNNER_HOST  skip the Nebius lookup and use this host/IP
#   NEBIUS_RUNNER_USER  ssh user (default: ubuntu)
#   NEBIUS_RUNNER_SANITIZER       run under compute-sanitizer with this tool
#                                 (memcheck, racecheck, initcheck, synccheck)
#   NEBIUS_RUNNER_SANITIZER_ARGS  extra compute-sanitizer flags (word-split)
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <binary> [args...]" >&2
    exit 2
fi

bin=$(realpath "$1")
shift

host=${NEBIUS_RUNNER_HOST:-}
if [[ -z $host ]]; then
    host=$(nebius compute instance list --format json | jq -r '
        [.items[] | select(.status.state == "RUNNING")
                  | .status.network_interfaces[0].public_ip_address.address // empty]
        | first // empty | sub("/32$"; "")')
    if [[ -z $host ]]; then
        profile=$(nebius profile current 2>/dev/null || echo "?")
        echo "nebius-runner: no RUNNING instance with a public IP in profile '$profile'" >&2
        exit 1
    fi
fi
target="${NEBIUS_RUNNER_USER:-ubuntu}@$host"

# Share one ssh connection between rsync, the run, and subsequent test binaries.
ssh_opts=(
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=15
    -o ControlMaster=auto
    -o "ControlPath=${XDG_RUNTIME_DIR:-/tmp}/nebius-runner-%C"
    -o ControlPersist=120
)

# Key the remote path on the local path so reruns get rsync delta transfers,
# while concurrent invocations (e.g. doctests, all named rust_out) don't collide.
key=$(printf '%s' "$bin" | sha256sum | cut -c1-16)
remote_dir=".cache/nebius-runner/$key"
remote_bin="$remote_dir/$(basename "$bin")"
pidfile="$remote_dir/pid.$$"

ssh "${ssh_opts[@]}" "$target" "mkdir -p $(printf '%q' "$remote_dir")"
rsync --times --perms --compress --compress-choice=zstd \
    -e "ssh ${ssh_opts[*]}" "$bin" "$target:$remote_bin"
echo "nebius-runner: running $target:~/$remote_bin" >&2

# Without a tty, ssh doesn't propagate Ctrl-C to the remote process, so kill it
# explicitly via the process group of its parent shell (which traps TERM so it
# can clean up). The group includes children such as compute-sanitizer's target.
on_signal() {
    trap - INT TERM HUP
    ssh "${ssh_opts[@]}" -o ConnectTimeout=5 "$target" \
        "kill -TERM -- -\"\$(cat $(printf '%q' "$pidfile") 2>/dev/null)\"" >/dev/null 2>&1 || true
    exit 130
}
trap on_signal INT TERM HUP

prefix=
if [[ -n ${NEBIUS_RUNNER_SANITIZER:-} ]]; then
    read -ra sanitizer_args <<<"${NEBIUS_RUNNER_SANITIZER_ARGS:-}"
    printf -v prefix '%q ' "--tool=$NEBIUS_RUNNER_SANITIZER" --error-exitcode=86 "${sanitizer_args[@]}"
    prefix="\$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer) $prefix"
fi
remote_args=
[[ $# -gt 0 ]] && printf -v remote_args ' %q' "$@"
remote_cmd="trap : TERM
echo \$\$ > $(printf '%q' "$pidfile")
cd $(printf '%q' "$remote_dir")
RUST_LOG=debug RUST_BACKTRACE=1 ${prefix}./$(printf '%q' "$(basename "$bin")")${remote_args}
rc=\$?
rm -f $(printf '%q' "$(basename "$pidfile")")
exit \$rc"

# ssh exits with the remote command's status (255 on connection failure). Run
# it in the background and wait so the signal trap fires promptly.
set +e
ssh "${ssh_opts[@]}" "$target" "$remote_cmd" &
wait $!
