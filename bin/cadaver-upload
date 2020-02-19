#!/bin/sh

usage () { echo "$0 <src> <cadaver-args>*" >/dev/stderr; }
error () { echo "$1" >/dev/stderr; usage; exit 1; }

test $# '<' 3 || \
    error "Source and cadaver arguments expected!";

src="$1"; shift;
test -r "$src" || \
    error "Source argument should be a readable file or directory!";

cd "$(dirname "$src")";
src="$(basename "$src")";
root="$(pwd)";
rc="$(mktemp)";
{
    find "$src" '(' -type d -a -readable ')' \
    -printf 'mkcol "%p"\n';
    find "$src" '(' -type f -a -readable ')' \
    -printf 'cd "%h"\nlcd "%h"\n'            \
    -printf 'mput "%f"\n'                    \
    -printf 'cd -\nlcd "'"$root"'"\n';
    echo "quit";
} > "$rc";

cadaver -r "$rc" "$@";
rm -f "$rc";
