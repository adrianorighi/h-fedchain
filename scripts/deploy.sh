#!/usr/bin/env bash
set -euo pipefail

# H-FedChain deploy entrypoint (T14 rewrite).
#
# The previous deploy.sh was an interactive menu for a VM/SSH flow
# (config/topology.json, rsync to fog/cloud VMs, images h-fedchain-fog /
# h-fedchain-cloud, `push` to registry.example.com) that no longer matches
# the repository: those images and the topology flow were never part of the
# distributed stack. The deploy entrypoint is now the kind cloud deploy.
#
# All behavior lives in scripts/deploy-cloud.sh (single implementation);
# this wrapper keeps `./scripts/deploy.sh` working for existing docs
# (docs/07-deployment.md — drift to be fixed in T15).

exec "$(cd "$(dirname "$0")" && pwd)/deploy-cloud.sh" "$@"
