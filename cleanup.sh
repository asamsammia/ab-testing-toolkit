#!/usr/bin/env bash
set -e
if [ -d "CUPED/guardrail/uplift example" ]; then
  git rm -r "CUPED/guardrail/uplift example" || true
fi
if [ -f "install_ab_demo_patch.py" ]; then
  git rm "install_ab_demo_patch.py" || true
fi
if [ -f "install_ab_demo_patch_v2.py" ]; then
  git rm "install_ab_demo_patch_v2.py" || true
fi
echo "Cleanup staged. Next: git commit -m 'chore: remove old patch files' && git push"
