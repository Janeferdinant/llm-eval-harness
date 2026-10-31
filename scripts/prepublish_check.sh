#!/usr/bin/env bash
# Pre-publish check. Run from anywhere inside the repo: bash scripts/prepublish_check.sh
# Exit code 1 when anything is found. Findings are printed as "FINDING: ...".
#
# Lines containing the marker  prepublish-allow  are skipped, and so are the paths listed
# (one per line, exact match) in .prepublish-ignore.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 1

fail=0
finding() { echo "FINDING: $*"; fail=1; }

# Files that would be published: tracked plus untracked-but-not-ignored.
files=$(mktemp); trap 'rm -f "$files"; rm -rf "${scan_dir:-}"' EXIT
git ls-files -z -co --exclude-standard > "$files"

ignored_path() { [ -f .prepublish-ignore ] && grep -Fxq -- "$1" .prepublish-ignore; }

# 1. gitleaks over history and over the current files (secrets redacted in the output).
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks git . --no-banner --redact -v >/tmp/gitleaks-git.$$ 2>&1 || { finding "gitleaks found something in git history"; cat /tmp/gitleaks-git.$$; }
  rm -f /tmp/gitleaks-git.$$
  scan_dir=$(mktemp -d)
  tar --null -T "$files" -cf - 2>/dev/null | tar -xf - -C "$scan_dir"
  gitleaks dir "$scan_dir" --no-banner --redact -v >/tmp/gitleaks-dir.$$ 2>&1 || { finding "gitleaks found something in the working files"; cat /tmp/gitleaks-dir.$$; }
  rm -f /tmp/gitleaks-dir.$$
else
  finding "gitleaks is not installed"
fi

# 2. Personal or key-like patterns.
scan_pattern() {  # $1 = label, remaining args = grep options and pattern
  local label=$1; shift
  xargs -0 grep -nI "$@" -- < "$files" 2>/dev/null | while IFS= read -r hit; do
    path=${hit%%:*}
    case $hit in *prepublish-allow*) continue ;; esac
    ignored_path "$path" && continue
    echo "FINDING: $label in $hit"
  done
}
scan_email() {  # any address except GitHub noreply, noreply@ and example.com/.org ones
  xargs -0 grep -noI -E -e '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+' -- < "$files" 2>/dev/null |
  while IFS= read -r hit; do
    path=${hit%%:*}; rest=${hit#*:}; lineno=${rest%%:*}; match=${rest#*:}
    case $match in *@users.noreply.github.com|noreply@*|*@example.com|*@example.org) continue ;; esac
    ignored_path "$path" && continue
    sed -n "${lineno}p" "$path" | grep -q 'prepublish-allow' && continue
    echo "FINDING: email address in $path:$lineno"
  done
}
# Names that must never be published (a company, a user name, a private address) live in .prepublish-private,
# one per line, case-insensitive. That file is git-ignored, so the names themselves are never in the repo.
scan_private() {
  # the repo's own file, or one shared by every repo in the folder above (that folder is not a git repo)
  local private=.prepublish-private
  [ -f "$private" ] || private=../.prepublish-private
  [ -f "$private" ] || return 0
  while IFS= read -r pat; do
    pat=${pat%$'\r'}  # the file may have been saved with Windows line endings
    case $pat in ''|'#'*) continue ;; esac
    esc=$(printf '%s' "$pat" | sed 's/[][\.*^$/]/\\&/g')  # grep -iF aborts in Git for Windows: escape instead
    xargs -0 grep -nIi -e "$esc" -- < "$files" 2>/dev/null | while IFS= read -r hit; do
      path=${hit%%:*}; rest=${hit#*:}; lineno=${rest%%:*}
      ignored_path "$path" && continue
      echo "FINDING: private pattern in $path:$lineno"
    done
    n=$(git log --all -i -F -S"$pat" --oneline 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && echo "FINDING: private pattern appears in git history ($n commits)"
  done < "$private"
}
out=$( { scan_email
         scan_private
         scan_pattern "Windows user path" -F -e 'C:\Users\'
         scan_pattern "Linux home path" -F -e '/home/'
         scan_pattern "key-like token" -E -e '(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}'; } )
if [ -n "$out" ]; then echo "$out"; fail=1; fi

# 3. Files over 50 MB.
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  size=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$size" -gt 52428800 ]; then finding "file over 50 MB: $f ($size bytes)"; fi
done < "$files"

# 4. Licence files.
[ -f LICENSE ] || finding "LICENSE is missing"
[ -f NOTICE.md ] || finding "NOTICE.md is missing"

if [ "$fail" -eq 0 ]; then echo "prepublish check: clean"; fi
exit "$fail"
