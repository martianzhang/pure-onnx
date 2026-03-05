package openclip

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEnsureModelAssetsDownloadsAndCaches(t *testing.T) {
	repoID := "unit/test-repo"
	revision := "main"
	assetData := map[string][]byte{
		textModelFileName:    []byte("text-onnx"),
		visionModelFileName:  []byte("vision-onnx"),
		tokenizerFileName:    []byte(`{"type":"tokenizer"}`),
		preprocessorFileName: []byte(`{"size":224,"crop_size":224}`),
	}

	var mu sync.Mutex
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requestCount++
		mu.Unlock()

		prefix := "/" + repoID + "/resolve/" + revision + "/"
		if len(r.URL.Path) <= len(prefix) || r.URL.Path[:len(prefix)] != prefix {
			http.NotFound(w, r)
			return
		}
		fileName := r.URL.Path[len(prefix):]
		payload, ok := assetData[fileName]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	shaByFile := map[string]string{}
	sizeByFile := map[string]int64{}
	for fileName, data := range assetData {
		shaByFile[fileName] = sha256Hex(data)
		sizeByFile[fileName] = int64(len(data))
	}

	cfg := bootstrapConfig{
		repoID:             repoID,
		revision:           revision,
		baseURL:            server.URL,
		cacheDir:           tempDir,
		verifySHA:          true,
		shaByFile:          shaByFile,
		expectedSizeByFile: sizeByFile,
		maxDownloadBytes:   1024 * 1024,
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}

	assets, err := ensureModelAssets(cfg)
	if err != nil {
		t.Fatalf("ensureModelAssets failed: %v", err)
	}
	assertFileContains(t, assets.TextModelPath, assetData[textModelFileName])
	assertFileContains(t, assets.VisionModelPath, assetData[visionModelFileName])
	assertFileContains(t, assets.TokenizerPath, assetData[tokenizerFileName])
	assertFileContains(t, assets.PreprocessorConfigPath, assetData[preprocessorFileName])

	mu.Lock()
	firstRequestCount := requestCount
	mu.Unlock()
	if firstRequestCount != 4 {
		t.Fatalf("unexpected request count after first download: got %d, want 4", firstRequestCount)
	}

	_, err = ensureModelAssets(cfg)
	if err != nil {
		t.Fatalf("second ensureModelAssets failed: %v", err)
	}
	mu.Lock()
	secondRequestCount := requestCount
	mu.Unlock()
	if secondRequestCount != firstRequestCount {
		t.Fatalf("expected cached second call to avoid network requests, got first=%d second=%d", firstRequestCount, secondRequestCount)
	}
}

func TestEnsureAssetFileReplacesCorruptFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("good-content"))
	}))
	defer server.Close()

	tempDir := t.TempDir()
	destination := filepath.Join(tempDir, textModelFileName)
	if err := os.WriteFile(destination, []byte("corrupt"), 0o644); err != nil {
		t.Fatalf("failed to write corrupt fixture: %v", err)
	}

	expected := sha256Hex([]byte("good-content"))
	cfg := bootstrapConfig{
		repoID:             "repo/test",
		revision:           "main",
		baseURL:            server.URL,
		cacheDir:           tempDir,
		verifySHA:          true,
		shaByFile:          map[string]string{textModelFileName: expected},
		expectedSizeByFile: map[string]int64{textModelFileName: int64(len("good-content"))},
		maxDownloadBytes:   1024 * 1024,
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}

	if err := ensureAssetFile(cfg, destination, textModelFileName, expected, int64(len("good-content"))); err != nil {
		t.Fatalf("ensureAssetFile failed: %v", err)
	}
	assertFileContains(t, destination, []byte("good-content"))
}

func TestDownloadFileOnceRejectsOversizeByContentLength(t *testing.T) {
	payload := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "64")
		_, _ = w.Write([]byte(payload))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileOnce(&http.Client{}, server.URL+"/asset.bin", destination, "", 32, 0)
	if err == nil || !strings.Contains(err.Error(), "content-length") {
		t.Fatalf("expected content-length oversize error, got: %v", err)
	}
}

func TestDownloadFileOnceRejectsOversizeWithoutContentLength(t *testing.T) {
	payload := strings.Repeat("x", 64)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("response writer must support flush")
		}
		for i := 0; i < len(payload); i++ {
			_, _ = w.Write([]byte{payload[i]})
			flusher.Flush()
		}
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileOnce(&http.Client{}, server.URL+"/asset.bin", destination, "", 32, 0)
	if err == nil || !strings.Contains(err.Error(), "download exceeded max allowed size") {
		t.Fatalf("expected streamed max-bytes error, got: %v", err)
	}
}

func TestDownloadFileWithRetryRetryableHTTPStatus(t *testing.T) {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "temporary error", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileWithRetry(
		&http.Client{},
		server.URL+"/asset.bin",
		destination,
		"",
		16,
		2,
	)
	if err != nil {
		t.Fatalf("downloadFileWithRetry failed: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts for retryable status, got %d", attempts)
	}
	assertFileContains(t, destination, []byte("ok"))
}

func TestDownloadFileWithRetryDoesNotRetryClientError(t *testing.T) {
	var attempts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "asset.bin")
	err := downloadFileWithRetry(
		&http.Client{},
		server.URL+"/asset.bin",
		destination,
		"",
		16,
		0,
	)
	if err == nil {
		t.Fatalf("expected download to fail for 400")
	}
	if attempts != 1 {
		t.Fatalf("expected no retry for non-retryable status, got %d", attempts)
	}
}

func TestAcquireFileLockTimeout(t *testing.T) {
	lockPath := filepath.Join(t.TempDir(), "asset.lock")
	first, err := acquireFileLock(lockPath, time.Second, time.Hour)
	if err != nil {
		t.Fatalf("failed to acquire first lock: %v", err)
	}
	defer func() {
		_ = first.Release()
	}()

	_, err = acquireFileLock(lockPath, 200*time.Millisecond, time.Hour)
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected timed out lock acquisition, got: %v", err)
	}
}

func TestAcquireFileLockStaleCleanup(t *testing.T) {
	tempDir := t.TempDir()
	lockPath := filepath.Join(tempDir, "asset.lock")
	if err := os.WriteFile(lockPath, []byte("stale"), 0o600); err != nil {
		t.Fatalf("failed to seed stale lock: %v", err)
	}
	oldTime := time.Now().Add(-time.Hour)
	if err := os.Chtimes(lockPath, oldTime, oldTime); err != nil {
		t.Fatalf("failed to set stale lock mtime: %v", err)
	}

	lock, err := acquireFileLock(lockPath, time.Second, time.Millisecond*10)
	if err != nil {
		t.Fatalf("acquireFileLock stale cleanup failed: %v", err)
	}
	defer func() {
		_ = lock.Release()
	}()
}

func TestIsRetryableDownloadError(t *testing.T) {
	if !isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusRequestTimeout}) {
		t.Fatalf("expected HTTP 408 to be retryable")
	}
	if !isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusTooManyRequests}) {
		t.Fatalf("expected HTTP 429 to be retryable")
	}
	if !isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusInternalServerError}) {
		t.Fatalf("expected HTTP 500 to be retryable")
	}
	if isRetryableDownloadError(&downloadStatusError{StatusCode: http.StatusBadRequest}) {
		t.Fatalf("expected HTTP 400 to be non-retryable")
	}
}

func TestIsRetryableDownloadErrorNetworkTransient(t *testing.T) {
	if !isRetryableDownloadError(&url.Error{
		Op:  "Get",
		URL: "http://example.com",
		Err: errors.New("connection refused"),
	}) {
		t.Fatalf("expected transient connection error to be retryable")
	}

	if isRetryableDownloadError(&url.Error{
		Op:  "Get",
		URL: "http://example.com",
		Err: errors.New("no such host"),
	}) {
		t.Fatalf("expected DNS resolution error to be non-retryable")
	}

	if isRetryableDownloadError(&url.Error{
		Op:  "Get",
		URL: "http://example.com",
		Err: errors.New("permission denied"),
	}) {
		t.Fatalf("expected unrelated URL error to be non-retryable")
	}
}

func TestRedirectPolicyRejectsHTTPSDowngrade(t *testing.T) {
	policy := makeRedirectPolicy("huggingface.co")
	req := &http.Request{URL: mustParseURL(t, "http://example.com/file")}
	via := []*http.Request{
		{URL: mustParseURL(t, "https://huggingface.co/repo/resolve/main/file")},
	}
	err := policy(req, via)
	if err == nil || !strings.Contains(err.Error(), "downgrade") {
		t.Fatalf("expected HTTPS downgrade rejection, got: %v", err)
	}
}

func TestIsAllowedRedirectHost(t *testing.T) {
	if !isAllowedRedirectHost("huggingface.co", "cas-bridge.xethub.hf.co") {
		t.Fatalf("expected hf.co subdomain to be allowed")
	}
	if isAllowedRedirectHost("huggingface.co", "example.com") {
		t.Fatalf("expected unrelated host to be rejected")
	}
	if !isAllowedRedirectHost("my.mirror.local", "cdn.my.mirror.local") {
		t.Fatalf("expected subdomain of custom host to be allowed")
	}
}

func TestEnsureDefaultAssetsValidation(t *testing.T) {
	_, err := EnsureDefaultAssets(WithBootstrapCacheDir(""))
	if err == nil {
		t.Fatalf("expected validation error for empty cache dir")
	}
}

func TestEnsureDefaultAssetsCustomRepoRequiresChecksums(t *testing.T) {
	_, err := EnsureDefaultAssets(
		WithBootstrapRepoID("custom/repo"),
		WithBootstrapRevision("main"),
	)
	if err == nil || !strings.Contains(err.Error(), "missing checksum") {
		t.Fatalf("expected missing checksum error for custom repo without explicit checksums, got: %v", err)
	}
}

func assertFileContains(t *testing.T, path string, expected []byte) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read %q: %v", path, err)
	}
	if string(got) != string(expected) {
		t.Fatalf("unexpected file content at %q: got %q, want %q", path, string(got), string(expected))
	}
}

func mustParseURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("failed to parse URL %q: %v", raw, err)
	}
	return parsed
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
