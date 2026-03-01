#!/usr/bin/env bash
set -e

# Ensure we're inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: Not inside a git repository."
  exit 1
fi

echo "Generating repository tree (respecting .gitignore)..."

git ls-files | awk -F/ '
{
  path=""
  for (i=1; i<=NF; i++) {
    path = path $i
    if (!seen[path]++) {
      indent=""
      for (j=1; j<i; j++) indent=indent "│   "
      if (i == NF)
        print indent "├── " $i
      else
        print indent "├── " $i "/"
    }
    path = path "/"
  }
}' > tree.txt

echo "Tree written to tree.txt"