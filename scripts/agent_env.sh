#!/usr/bin/env bash
# Source in agent shells:  source scripts/agent_env.sh
# Enables ROADMAP strict gate checks (default on) and a stable agent id.

export ORBIT_WARS_IMPL_GATE="${ORBIT_WARS_IMPL_GATE:-1}"
export ORBIT_WARS_AGENT_ID="${ORBIT_WARS_AGENT_ID:-cursor-$(hostname)-$$}"
