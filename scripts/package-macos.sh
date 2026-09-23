#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/dist/Keel-macOS.zip}"
binary="${2:-$root/target/release/keel}"
laya_worker_bundle="${KEEL_LAYA_WORKER_BUNDLE:-}"
laya_model="${KEEL_LAYA_MODEL:-}"
if [[ ! -x "$binary" ]]; then
  echo "Keel binary not found: $binary" >&2
  exit 1
fi
if [[ "$out" != /* ]]; then
  out="$root/$out"
fi
if [[ "$out" != *.zip ]]; then
  echo "Package output must be a .zip file" >&2
  exit 1
fi
if [[ -z "$laya_worker_bundle" || ! -x "$laya_worker_bundle/laya-worker" ]]; then
  echo "Set KEEL_LAYA_WORKER_BUNDLE to the standalone Laya worker directory" >&2
  exit 1
fi
if ! "$laya_worker_bundle/laya-worker" --capabilities 2>/dev/null | grep -q 'download-progress-v1'; then
  echo "The Laya worker must support structured on-demand download progress" >&2
  exit 1
fi
if [[ -n "$laya_model" ]]; then
  # The full package must contain the exact immutable checkpoint accepted by
  # the install path; a lone config file cannot prove inference will work.
  while IFS=' ' read -r expected relative; do
    [[ -n "$expected" ]] || continue
    file="$laya_model/$relative"
    if [[ ! -f "$file" || -L "$file" ]]; then
      echo "Laya checkpoint file missing or not regular: $relative" >&2
      exit 1
    fi
    actual="$(shasum -a 256 "$file")"
    actual="${actual%% *}"
    if [[ "$actual" != "$expected" ]]; then
      echo "Laya checkpoint hash mismatch: $relative" >&2
      exit 1
    fi
  done <<'SHA256'
9f5ae62247c9be221c7ced17157b1b5ec25ef70bdd3cb809d45ba698da579418 model.mlpackage/Manifest.json
6b906ab7b6b8bbc0f11608425f6091b508bbe968788f27d8e81df25399c281ba model.mlpackage/Data/com.apple.CoreML/weights/weight.bin
d6ea5688f45c6afa3c041698d33337ac9c62de7b3f1077c9cb39be7c965fe0de model.mlpackage/Data/com.apple.CoreML/model.mlmodel
83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4 encoder/config.json
6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1 tokenizer/tokenizer_config.json
609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f tokenizer/tokenizer.json
8131967fdb403243f2817d7eeb09721d1e822bd7301c244d25afb44bcc258352 coreml_config.json
25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6 rl_agent_config.json
SHA256
fi

staging="$(mktemp -d "${TMPDIR:-/tmp}/keel-package.XXXXXX")"
trap 'rm -r "$staging"' EXIT
bundle="$staging/Keel.app"
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
cp "$binary" "$bundle/Contents/MacOS/keel"
ditto "$laya_worker_bundle" "$bundle/Contents/Resources"
if [[ -n "$laya_model" ]]; then
  ditto "$laya_model" "$bundle/Contents/Resources/laya-model"
  rm -rf "$bundle/Contents/Resources/laya-model/.cache"
fi
# Keep the same layout as the source notices so their relative links resolve.
cp "$root/LICENSE" "$bundle/Contents/Resources/LICENSE"
cp "$root/THIRD_PARTY_NOTICES.md" "$bundle/Contents/Resources/THIRD_PARTY_NOTICES.md"
ditto "$root/licenses" "$bundle/Contents/Resources/licenses"
cat > "$bundle/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Keel</string>
  <key>CFBundleDisplayName</key><string>Keel</string>
  <key>CFBundleIdentifier</key><string>app.keel.local</string>
  <key>CFBundleExecutable</key><string>keel</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.2.0</string>
  <key>CFBundleVersion</key><string>2</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleIconFile</key><string>Keel</string>
</dict></plist>
PLIST
if [[ -f "$root/assets/Keel.icns" ]]; then
  cp "$root/assets/Keel.icns" "$bundle/Contents/Resources/Keel.icns"
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
  xattr -cr "$bundle"
  codesign --force --deep --sign - --timestamp=none "$bundle"
  codesign --verify --deep --strict "$bundle"
fi
mkdir -p "$(dirname "$out")"
ditto -c -k --keepParent "$bundle" "$out"
echo "$out"
