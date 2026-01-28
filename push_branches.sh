#!/bin/bash
# push_branch.sh
# Usage: ./push_branch.sh branch_name

BRANCH=$1

if [ -z "$BRANCH" ]; then
  echo "❌ Please provide a branch name, e.g.: ./push_branch.sh main"
  exit 1
fi

echo "➡️  Pushing branch $BRANCH to origin..."
git push origin "$BRANCH"

if [ $? -ne 0 ]; then
  echo "❌ Failed to push to origin"
  exit 1
fi

echo "➡️  Pushing branch $BRANCH to secondary..."
git push secondary "$BRANCH"

if [ $? -ne 0 ]; then
  echo "❌ Failed to push to secondary"
  exit 1
fi

echo "✅ Branch $BRANCH has been successfully pushed to both origin and secondary!"
