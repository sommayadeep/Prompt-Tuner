#!/bin/bash
# Pre-submission validation script
# Usage: bash pre_validation.sh https://your-space.hf.space ~/path/to/env

set -e

SPACE_URL="${1:-https://sommayadeep-prompt-optimizer.hf.space}"
ENV_DIR="${2:-.}"

echo "🔍 Phase 2 Pre-validation Script"
echo "================================"
echo "Space URL: $SPACE_URL"
echo "Env Dir: $ENV_DIR"
echo ""

# Check 1: openenv.yaml exists and has tasks
echo "[CHECK 1] openenv.yaml has 3+ tasks with inline graders..."
if [ ! -f "$ENV_DIR/openenv.yaml" ]; then
  echo "❌ FAIL: openenv.yaml not found"
  exit 1
fi

TASK_COUNT=$(python3 -c "
import yaml
try:
  d = yaml.safe_load(open('$ENV_DIR/openenv.yaml'))
  tasks = d.get('tasks', [])
  ok = sum(1 for t in tasks if t.get('grader') is not None)
  print(ok)
except Exception as e:
  print(0)
" 2>/dev/null || echo 0)

if [ "$TASK_COUNT" -lt 3 ]; then
  echo "❌ FAIL: Only $TASK_COUNT tasks with graders (need 3+)"
  exit 1
else
  echo "✅ PASS: $TASK_COUNT tasks with inline graders"
fi
echo ""

# Check 2: inference.py runs and emits correct format
echo "[CHECK 2] inference.py output format..."
cd "$ENV_DIR"

INFERENCE_OUTPUT=$(python3 inference.py 2>&1 || true)

if ! echo "$INFERENCE_OUTPUT" | grep -q "\[START\]"; then
  echo "❌ FAIL: Missing [START] marker"
  exit 1
fi

if ! echo "$INFERENCE_OUTPUT" | grep -q "\[END\]"; then
  echo "❌ FAIL: Missing [END] marker"
  exit 1
fi

STEP_COUNT=$(echo "$INFERENCE_OUTPUT" | grep -c "\[STEP\]" || echo 0)
if [ "$STEP_COUNT" -lt 3 ]; then
  echo "❌ FAIL: Only $STEP_COUNT [STEP] blocks (need 3+)"
  exit 1
fi

echo "✅ PASS: Correct output format ([START], 3+ [STEP], [END])"
echo ""

# Check 3: All scores are strictly between 0 and 1
echo "[CHECK 3] Score bounds (0 < score < 1)..."
INVALID_SCORES=$(echo "$INFERENCE_OUTPUT" | grep "score:" | grep -E "^\s*score:\s*(0\.0+|1\.0+)\s*$" || echo "")

if [ ! -z "$INVALID_SCORES" ]; then
  echo "❌ FAIL: Found scores at bounds (0.0 or 1.0)"
  echo "$INVALID_SCORES"
  exit 1
fi

echo "✅ PASS: All scores strictly in (0, 1)"
echo ""

# Check 4: App endpoints respond
echo "[CHECK 4] API endpoints (/reset, /step)..."
if ! python3 -c "
import requests
try:
  resp = requests.post('$SPACE_URL/reset', json={'task_id': 'task1_keywords'}, timeout=5)
  if resp.status_code == 200:
    print('✅ /reset OK')
  else:
    print(f'❌ /reset failed: {resp.status_code}')
    exit(1)
except Exception as e:
  print(f'⚠️  /reset warning: {e}')
" 2>&1; then
  echo "⚠️  WARN: Could not reach Space URL (may be sleeping)"
fi

echo ""
echo "================================"
echo "✅ ALL LOCAL CHECKS PASSED"
echo "Ready to submit Phase 2!"
echo "================================"
