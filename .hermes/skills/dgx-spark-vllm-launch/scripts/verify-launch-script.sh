#!/usr/bin/env bash
# verify-launch-script.sh — verify a vLLM launch script for syntax, dead code, and die()
# Usage: ./scripts/verify-launch-script.sh <path/to/start-*.sh>
set -euo pipefail

script_path="${1:-}"
if [ -z "$script_path" ] || [ ! -f "$script_path" ]; then
  echo "Usage: $0 <path/to/start-*.sh>"
  exit 1
fi

name="$(basename "$script_path")"
pass=0
fail=0

check() {
  if [ "$1" -eq 0 ]; then
    echo "  ✓ $2"
    pass=$((pass + 1))
  else
    echo "  ✗ $2"
    fail=$((fail + 1))
  fi
}

echo "=== Verifying: $name ==="

# 1. Syntax check
bash -n "$script_path" 2>/dev/null
check $? "bash -n syntax"

# 2. die() is defined
grep -q "^die()" "$script_path"
check $? "die() defined"

# 3. ok() is NOT defined (dead code)
if grep -q "^ok()" "$script_path"; then
  echo "  ✗ ok() should be removed (dead code)"
  fail=$((fail + 1))
else
  echo "  ✓ ok() removed"
  pass=$((pass + 1))
fi

# 4. info() is NOT defined (dead code)
if grep -q "^info()" "$script_path"; then
  echo "  ✗ info() should be removed (dead code)"
  fail=$((fail + 1))
else
  echo "  ✓ info() removed"
  pass=$((pass + 1))
fi

# 5. die() is actually used (not just defined)
grep -q 'die "' "$script_path"
check $? "die() is invoked"

echo "=== Results: $pass passed, $fail failed ==="
exit $fail
