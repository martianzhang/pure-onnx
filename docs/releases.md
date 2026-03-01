# Release Runbook (R2 + Cosign)

This repository publishes signed release artifacts to:

- `https://releases.amikos.tech/pure-onnx/<version>/...`
- `https://releases.amikos.tech/pure-onnx/latest.json`
- `https://github.com/amikos-tech/pure-onnx/releases`

The workflow is implemented in `.github/workflows/release.yml` and runs on tag pushes matching `v*`.

## Release Artifacts

`make release` builds cross-platform bundles into `dist/`:

- `pure-onnx-<version>-linux-amd64.tar.gz`
- `pure-onnx-<version>-linux-arm64.tar.gz`
- `pure-onnx-<version>-darwin-amd64.tar.gz`
- `pure-onnx-<version>-darwin-arm64.tar.gz`
- `pure-onnx-<version>-windows-amd64.tar.gz`

Each archive contains:

- `bin/basic` or `bin/basic.exe`
- `bin/inference` or `bin/inference.exe`
- `README.md`
- `ARTIFACTS.txt`

The release workflow also generates:

- `SHA256SUMS`
- `*.sig` and `*.pem` for each archive
- `SHA256SUMS.sig` and `SHA256SUMS.pem`

## Required GitHub Secrets

- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ENDPOINT` (`https://<account_id>.r2.cloudflarestorage.com`)

## Optional Variables/Secrets

- `RELEASE_INDEX_ENABLED=1` to publish signed `releases.json` index
- `RELEASES_DOMAIN` (defaults to `releases.amikos.tech`)
- `CF_ZONE_ID` + `CLOUDFLARE_API_TOKEN` for metadata cache purges

## R2 Token Scope

Use a dedicated Cloudflare R2 token scoped to:

- bucket: `releases`
- prefix: `pure-onnx/`
- permissions: Object Read + Write

## Publishing a Release

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Verifying Downloaded Artifacts

```bash
TAG=v0.1.0
PROJECT=pure-onnx
BASE_URL="https://releases.amikos.tech/${PROJECT}/${TAG}"

curl -LO "${BASE_URL}/${PROJECT}-${TAG}-linux-amd64.tar.gz"
curl -LO "${BASE_URL}/${PROJECT}-${TAG}-linux-amd64.tar.gz.sig"
curl -LO "${BASE_URL}/${PROJECT}-${TAG}-linux-amd64.tar.gz.pem"
curl -LO "${BASE_URL}/SHA256SUMS"
curl -LO "${BASE_URL}/SHA256SUMS.sig"
curl -LO "${BASE_URL}/SHA256SUMS.pem"

# Verify only the artifact you downloaded (Linux example).
grep " ${PROJECT}-${TAG}-linux-amd64.tar.gz$" SHA256SUMS | sha256sum -c -

cosign verify-blob \
  --signature "${PROJECT}-${TAG}-linux-amd64.tar.gz.sig" \
  --certificate "${PROJECT}-${TAG}-linux-amd64.tar.gz.pem" \
  --certificate-identity "https://github.com/amikos-tech/pure-onnx/.github/workflows/release.yml@refs/tags/${TAG}" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "${PROJECT}-${TAG}-linux-amd64.tar.gz"

cosign verify-blob \
  --signature SHA256SUMS.sig \
  --certificate SHA256SUMS.pem \
  --certificate-identity "https://github.com/amikos-tech/pure-onnx/.github/workflows/release.yml@refs/tags/${TAG}" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  SHA256SUMS
```

On macOS, replace `sha256sum` with `shasum -a 256`:

```bash
grep " ${PROJECT}-${TAG}-darwin-arm64.tar.gz$" SHA256SUMS | shasum -a 256 -c -
```
