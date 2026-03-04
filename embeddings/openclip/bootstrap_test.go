package openclip

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
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
	for fileName, data := range assetData {
		shaByFile[fileName] = sha256Hex(data)
	}

	cfg := bootstrapConfig{
		repoID:    repoID,
		revision:  revision,
		baseURL:   server.URL,
		cacheDir:  tempDir,
		verifySHA: true,
		shaByFile: shaByFile,
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
		repoID:    "repo/test",
		revision:  "main",
		baseURL:   server.URL,
		cacheDir:  tempDir,
		verifySHA: true,
		shaByFile: map[string]string{textModelFileName: expected},
		httpClient: &http.Client{
			Transport: http.DefaultTransport,
		},
	}

	if err := ensureAssetFile(cfg, destination, textModelFileName, expected); err != nil {
		t.Fatalf("ensureAssetFile failed: %v", err)
	}
	assertFileContains(t, destination, []byte("good-content"))
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

func TestEnsureDefaultAssetsValidation(t *testing.T) {
	_, err := EnsureDefaultAssets(WithBootstrapCacheDir(""))
	if err == nil {
		t.Fatalf("expected validation error for empty cache dir")
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

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
