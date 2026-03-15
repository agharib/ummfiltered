#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESKTOP_DIR="$ROOT_DIR/desktop"
APP_NAME="UmmFiltered"
APP_BUNDLE="$DESKTOP_DIR/src-tauri/target/release/bundle/macos/$APP_NAME.app"
ARTIFACTS_DIR="$DESKTOP_DIR/release-artifacts"
DMG_PATH="$ARTIFACTS_DIR/$APP_NAME.dmg"
ZIP_PATH="$ARTIFACTS_DIR/$APP_NAME.app.zip"
STAGING_DIR="$ARTIFACTS_DIR/dmg-staging"
ENTITLEMENTS_PATH="$DESKTOP_DIR/src-tauri/Entitlements.plist"

required_vars=(
  APPLE_SIGNING_IDENTITY
  APPLE_ID
  APPLE_TEAM_ID
  APPLE_APP_PASSWORD
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
done

mkdir -p "$ARTIFACTS_DIR"
rm -rf "$ZIP_PATH" "$DMG_PATH" "$STAGING_DIR"

cd "$DESKTOP_DIR"

python3 ./scripts/build_macos_worker.py
npm run build
npm run tauri build -- --bundles app

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "Expected app bundle at $APP_BUNDLE" >&2
  exit 1
fi

codesign \
  --force \
  --deep \
  --sign "$APPLE_SIGNING_IDENTITY" \
  --options runtime \
  --entitlements "$ENTITLEMENTS_PATH" \
  "$APP_BUNDLE"

ditto -c -k --keepParent "$APP_BUNDLE" "$ZIP_PATH"
xcrun notarytool submit "$ZIP_PATH" \
  --apple-id "$APPLE_ID" \
  --team-id "$APPLE_TEAM_ID" \
  --password "$APPLE_APP_PASSWORD" \
  --wait
xcrun stapler staple "$APP_BUNDLE"

mkdir -p "$STAGING_DIR"
cp -R "$APP_BUNDLE" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

codesign --force --sign "$APPLE_SIGNING_IDENTITY" "$DMG_PATH"
xcrun notarytool submit "$DMG_PATH" \
  --apple-id "$APPLE_ID" \
  --team-id "$APPLE_TEAM_ID" \
  --password "$APPLE_APP_PASSWORD" \
  --wait
xcrun stapler staple "$DMG_PATH"

echo "Signed and notarized DMG created at $DMG_PATH"
