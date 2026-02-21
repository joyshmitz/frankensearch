#!/usr/bin/env bash
#
# fsfs installer (frankensearch standalone CLI)
#
# One-liner install:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash
#
# With cache buster:
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh?$(date +%s)" | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --from-source      Build from source instead of downloading binary
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-frankensearch}"
BINARY_NAME="fsfs"
VERSION="${VERSION:-}"
DEST_DEFAULT="$HOME/.local/bin"
DEST="${DEST:-$DEST_DEFAULT}"
EASY=0
QUIET=0
VERIFY=0
FROM_SOURCE=0
CHECKSUM="${CHECKSUM:-}"
CHECKSUM_URL="${CHECKSUM_URL:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
LOCK_FILE="/tmp/fsfs-install.lock"
SYSTEM=0
NO_GUM=0

# Detect gum for fancy output (https://github.com/charmbracelet/gum)
HAS_GUM=0
if command -v gum &> /dev/null && [ -t 1 ]; then
  HAS_GUM=1
fi

log() { [ "$QUIET" -eq 1 ] && return 0; echo -e "$@"; }

info() {
  [ "$QUIET" -eq 1 ] && return 0
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 39 "→ $*"
  else
    echo -e "\033[0;34m→\033[0m $*"
  fi
}

ok() {
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 42 "✓ $*"
  else
    echo -e "\033[0;32m✓\033[0m $*"
  fi
}

warn() {
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 214 "⚠ $*"
  else
    echo -e "\033[1;33m⚠\033[0m $*"
  fi
}

err() {
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style --foreground 196 "✗ $*"
  else
    echo -e "\033[0;31m✗\033[0m $*"
  fi
}

run_with_spinner() {
  local title="$1"
  shift
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ] && [ "$QUIET" -eq 0 ]; then
    gum spin --spinner dot --title "$title" -- "$@"
  else
    info "$title"
    "$@"
  fi
}

resolve_version() {
  if [ -n "$VERSION" ]; then return 0; fi

  info "Resolving latest version..."
  local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
  local tag
  if ! tag=$(curl -fsSL -H "Accept: application/vnd.github.v3+json" "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/'); then
    tag=""
  fi

  if [ -n "$tag" ]; then
    VERSION="$tag"
    info "Resolved latest version: $VERSION"
  else
    # Try redirect-based resolution as fallback
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if tag=$(curl -fsSL -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||'); then
      if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *"/"* ]]; then
        VERSION="$tag"
        info "Resolved latest version via redirect: $VERSION"
        return 0
      fi
    fi
    err "Could not resolve latest version. Use --version vX.Y.Z"
    exit 1
  fi
}

maybe_add_path() {
  case ":$PATH:" in
    *:"$DEST":*) return 0;;
    *)
      if [ "$EASY" -eq 1 ]; then
        local UPDATED=0
        for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
          if [ -e "$rc" ] && [ -w "$rc" ]; then
            if ! grep -qF "$DEST" "$rc" 2>/dev/null; then
              printf '\nexport PATH="%s:$PATH"\n' "$DEST" >> "$rc"
            fi
            UPDATED=1
          fi
        done
        if [ "$UPDATED" -eq 1 ]; then
          warn "PATH updated in shell config; restart your shell to use ${BINARY_NAME}"
        else
          warn "Add $DEST to PATH to use ${BINARY_NAME}"
        fi
      else
        warn "Add $DEST to PATH to use ${BINARY_NAME}"
      fi
    ;;
  esac
}

ensure_rust() {
  if [ "${RUSTUP_INIT_SKIP:-0}" != "0" ]; then
    info "Skipping rustup install (RUSTUP_INIT_SKIP set)"
    return 0
  fi
  if command -v cargo >/dev/null 2>&1 && rustc --version 2>/dev/null | grep -q nightly; then return 0; fi
  if [ "$EASY" -ne 1 ]; then
    if [ -t 0 ]; then
      printf "Install Rust nightly via rustup? (y/N): "
      read -r ans
      case "$ans" in y|Y) :;; *) warn "Skipping rustup install"; return 0;; esac
    fi
  fi
  info "Installing rustup (nightly)"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  rustup component add rustfmt clippy || true
}

usage() {
  cat <<EOFU
Usage: install.sh [--version vX.Y.Z] [--dest DIR] [--system] [--easy-mode] [--verify] \\
                  [--artifact-url URL] [--checksum HEX] [--checksum-url URL] [--quiet] [--no-gum]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --from-source      Build from source instead of downloading binary
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
EOFU
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --system) SYSTEM=1; DEST="/usr/local/bin"; shift;;
    --easy-mode) EASY=1; shift;;
    --verify) VERIFY=1; shift;;
    --artifact-url) ARTIFACT_URL="$2"; shift 2;;
    --checksum) CHECKSUM="$2"; shift 2;;
    --checksum-url) CHECKSUM_URL="$2"; shift 2;;
    --from-source) FROM_SOURCE=1; shift;;
    --quiet|-q) QUIET=1; shift;;
    --no-gum) NO_GUM=1; shift;;
    -h|--help) usage; exit 0;;
    *) shift;;
  esac
done

# Show header
if [ "$QUIET" -eq 0 ]; then
  if [ "$HAS_GUM" -eq 1 ] && [ "$NO_GUM" -eq 0 ]; then
    gum style \
      --border normal \
      --border-foreground 39 \
      --padding "0 1" \
      --margin "1 0" \
      "$(gum style --foreground 42 --bold 'fsfs installer')" \
      "$(gum style --foreground 245 'Two-tier hybrid local search')"
  else
    echo ""
    echo -e "\033[1;32mfsfs installer\033[0m"
    echo -e "\033[0;90mTwo-tier hybrid local search (frankensearch)\033[0m"
    echo ""
  fi
fi

resolve_version

mkdir -p "$DEST"
OS=$(uname -s | tr 'A-Z' 'a-z')
ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *) warn "Unknown arch $ARCH, using as-is" ;;
esac

TARGET=""
EXT=""
case "${OS}-${ARCH}" in
  linux-x86_64)   TARGET="x86_64-unknown-linux-musl"; EXT="tar.xz" ;;
  linux-aarch64)  TARGET="aarch64-unknown-linux-musl"; EXT="tar.xz" ;;
  darwin-x86_64)  TARGET="x86_64-apple-darwin"; EXT="tar.xz" ;;
  darwin-aarch64) TARGET="aarch64-apple-darwin"; EXT="tar.xz" ;;
  *) :;;
esac

# Build artifact filename and download URL.
# dsr artifact naming: fsfs-${version_bare}-${target_triple}.${ext}
# Also try versionless: fsfs-${target_triple}.${ext}
VERSION_BARE="${VERSION#v}"  # strip leading v for artifact naming
TAR=""
URL=""
if [ "$FROM_SOURCE" -eq 0 ]; then
  if [ -n "$ARTIFACT_URL" ]; then
    TAR=$(basename "$ARTIFACT_URL")
    URL="$ARTIFACT_URL"
  elif [ -n "$TARGET" ]; then
    TAR="${BINARY_NAME}-${VERSION_BARE}-${TARGET}.${EXT}"
    URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${TAR}"
  else
    warn "No prebuilt artifact for ${OS}/${ARCH}; falling back to build-from-source"
    FROM_SOURCE=1
  fi
fi

# Cross-platform locking using mkdir (atomic on all POSIX systems including macOS)
LOCK_DIR="${LOCK_FILE}.d"
LOCKED=0
if mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCKED=1
  echo $$ > "$LOCK_DIR/pid"
else
  if [ -f "$LOCK_DIR/pid" ]; then
    OLD_PID=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && ! kill -0 "$OLD_PID" 2>/dev/null; then
      rm -rf "$LOCK_DIR"
      if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCKED=1
        echo $$ > "$LOCK_DIR/pid"
      fi
    fi
  fi
  if [ "$LOCKED" -eq 0 ]; then
    err "Another installer is running (lock $LOCK_DIR)"
    exit 1
  fi
fi

cleanup() {
  rm -rf "$TMP"
  if [ "$LOCKED" -eq 1 ]; then rm -rf "$LOCK_DIR"; fi
}

TMP=$(mktemp -d)
trap cleanup EXIT

if [ "$FROM_SOURCE" -eq 0 ]; then
  info "Downloading $URL"
  if ! curl -fsSL "$URL" -o "$TMP/$TAR"; then
    # Try versionless artifact name as fallback
    FALLBACK_TAR="${BINARY_NAME}-${TARGET}.${EXT}"
    FALLBACK_URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${FALLBACK_TAR}"
    info "Trying fallback: $FALLBACK_URL"
    if ! curl -fsSL "$FALLBACK_URL" -o "$TMP/$FALLBACK_TAR"; then
      warn "Artifact download failed; falling back to build-from-source"
      FROM_SOURCE=1
    else
      TAR="$FALLBACK_TAR"
    fi
  fi
fi

if [ "$FROM_SOURCE" -eq 1 ]; then
  info "Building from source (requires git, rust nightly)"
  ensure_rust
  git clone --depth 1 "https://github.com/${OWNER}/${REPO}.git" "$TMP/src"
  (cd "$TMP/src" && cargo build --release -p frankensearch-fsfs)
  BIN="$TMP/src/target/release/${BINARY_NAME}"
  [ -x "$BIN" ] || { err "Build failed"; exit 1; }
  if [ "$SYSTEM" -eq 1 ]; then
    sudo install -m 0755 "$BIN" "$DEST/${BINARY_NAME}"
  else
    install -m 0755 "$BIN" "$DEST/${BINARY_NAME}"
  fi
  ok "Installed to $DEST/${BINARY_NAME} (source build)"
  maybe_add_path
  if [ "$VERIFY" -eq 1 ]; then
    "$DEST/${BINARY_NAME}" --version || true
    ok "Self-test complete"
  fi
  ok "Done. Binary at: $DEST/${BINARY_NAME}"
  exit 0
fi

# Verify checksum
if [ -z "$CHECKSUM" ]; then
  if [ -z "$CHECKSUM_URL" ]; then
    CHECKSUM_URL="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/SHA256SUMS"
  fi
  info "Fetching checksum from ${CHECKSUM_URL}"
  CHECKSUM_FILE="$TMP/SHA256SUMS"
  if ! curl -fsSL "$CHECKSUM_URL" -o "$CHECKSUM_FILE"; then
    warn "Checksum not available; skipping verification"
    CHECKSUM="SKIP"
  else
    CHECKSUM=$(grep "  ${TAR}\$" "$CHECKSUM_FILE" 2>/dev/null | awk '{print $1}')
    if [ -z "$CHECKSUM" ]; then
      CHECKSUM=$(grep " ${TAR}\$" "$CHECKSUM_FILE" 2>/dev/null | awk '{print $1}')
    fi
    if [ -z "$CHECKSUM" ]; then warn "Checksum for ${TAR} not found; skipping verification"; CHECKSUM="SKIP"; fi
  fi
fi

if [ "$CHECKSUM" != "SKIP" ]; then
  if command -v sha256sum >/dev/null 2>&1; then
    echo "$CHECKSUM  $TMP/$TAR" | sha256sum -c - || { err "Checksum mismatch"; exit 1; }
  elif command -v shasum >/dev/null 2>&1; then
    echo "$CHECKSUM  $TMP/$TAR" | shasum -a 256 -c - || { err "Checksum mismatch"; exit 1; }
  else
    warn "No sha256sum or shasum found; skipping checksum verification"
  fi
  ok "Checksum verified"
fi

# Extract
info "Extracting"
case "$TAR" in
  *.tar.xz)  tar -xJf "$TMP/$TAR" -C "$TMP" ;;
  *.tar.gz)  tar -xzf "$TMP/$TAR" -C "$TMP" ;;
  *.zip)     unzip -qo "$TMP/$TAR" -d "$TMP" ;;
  *)         err "Unknown archive format: $TAR"; exit 1 ;;
esac

# Find the binary in extracted files
BIN="$TMP/${BINARY_NAME}"
if [ ! -x "$BIN" ]; then
  BIN=$(find "$TMP" -maxdepth 3 -type f -name "${BINARY_NAME}" -perm -111 2>/dev/null | head -n 1)
fi
[ -x "$BIN" ] || { err "Binary not found in archive"; exit 1; }

if [ "$SYSTEM" -eq 1 ]; then
  sudo install -m 0755 "$BIN" "$DEST/${BINARY_NAME}"
else
  install -m 0755 "$BIN" "$DEST/${BINARY_NAME}"
fi
ok "Installed to $DEST/${BINARY_NAME}"
maybe_add_path

if [ "$VERIFY" -eq 1 ]; then
  "$DEST/${BINARY_NAME}" --version || true
  ok "Self-test complete"
fi

ok "Done. Binary at: $DEST/${BINARY_NAME}"
echo ""
info "Quick start:"
echo "  1. Index a directory:  fsfs index /path/to/your/files"
echo "  2. Search:             fsfs search \"your query\""
echo "  3. Interactive TUI:    fsfs"
echo ""
