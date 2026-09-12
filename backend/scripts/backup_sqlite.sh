#!/bin/sh
# Takes a consistent snapshot of the live SQLite DB using sqlite3's .backup
# command (safe to run against a database that's actively being written to —
# unlike `cp`, which can copy a half-written page and corrupt the copy).
# Keeps the last 14 daily backups and deletes older ones.
set -eu

DB_PATH="${DB_PATH:-/app/data/trace.db}"
BACKUP_DIR="${BACKUP_DIR:-/app/backups}"
KEEP_DAYS="${KEEP_DAYS:-14}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$BACKUP_DIR"

if [ ! -f "$DB_PATH" ]; then
  echo "No database found at $DB_PATH — nothing to back up yet."
  exit 0
fi

sqlite3 "$DB_PATH" ".backup '$BACKUP_DIR/trace-$STAMP.db'"
echo "Backed up $DB_PATH -> $BACKUP_DIR/trace-$STAMP.db"

find "$BACKUP_DIR" -name 'trace-*.db' -mtime "+$KEEP_DAYS" -delete