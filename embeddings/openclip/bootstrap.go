package openclip

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// DefaultBootstrapRepoID is the default Hugging Face repository for OpenCLIP ONNX artifacts.
	DefaultBootstrapRepoID = "amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx"
	// DefaultBootstrapRevision is the default repository revision used by the bootstrapper.
	DefaultBootstrapRevision = "248a2ed76a7189fc080e654e36930171331ef085"
	// DefaultBootstrapBaseURL is the default Hugging Face host used by the bootstrapper.
	DefaultBootstrapBaseURL = "https://huggingface.co"
)

const (
	defaultTextModelSHA256 = "252b86e0ef1fc95b22cfd52fbf647142727fdbecc152556ffe0fba0b10a80370"
	defaultVisionSHA256    = "7e14f76233d0c840c0621b1ef68f5877efe9357850782b1bbaf0c01693f73b43"
	defaultTokenizerSHA256 = "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842"
	defaultPreprocSHA256   = "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd"
)

const (
	textModelFileName    = "text_model.onnx"
	visionModelFileName  = "vision_model.onnx"
	tokenizerFileName    = "tokenizer.json"
	preprocessorFileName = "preprocessor_config.json"
)

// ModelAssets describes a local OpenCLIP artifact bundle.
type ModelAssets struct {
	TextModelPath          string
	VisionModelPath        string
	TokenizerPath          string
	PreprocessorConfigPath string
}

// BootstrapOption customizes EnsureDefaultAssets behavior.
type BootstrapOption func(*bootstrapConfig) error

type bootstrapConfig struct {
	repoID     string
	revision   string
	baseURL    string
	cacheDir   string
	hfToken    string
	verifySHA  bool
	shaByFile  map[string]string
	httpClient *http.Client
}

type bootstrapAssetSpec struct {
	fileName string
}

// WithBootstrapCacheDir sets the local cache directory used for downloaded assets.
func WithBootstrapCacheDir(path string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if strings.TrimSpace(path) == "" {
			return fmt.Errorf("bootstrap cache directory cannot be empty")
		}
		cfg.cacheDir = path
		return nil
	}
}

// WithBootstrapRepoID sets the Hugging Face repo ID to fetch assets from.
func WithBootstrapRepoID(repoID string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if strings.TrimSpace(repoID) == "" {
			return fmt.Errorf("bootstrap repo ID cannot be empty")
		}
		cfg.repoID = repoID
		return nil
	}
}

// WithBootstrapRevision sets the Hugging Face revision to fetch assets from.
func WithBootstrapRevision(revision string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		if strings.TrimSpace(revision) == "" {
			return fmt.Errorf("bootstrap revision cannot be empty")
		}
		cfg.revision = revision
		return nil
	}
}

// WithBootstrapToken sets an optional Hugging Face access token for downloads.
func WithBootstrapToken(token string) BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.hfToken = strings.TrimSpace(token)
		return nil
	}
}

// WithoutBootstrapChecksumVerification disables checksum verification for downloaded assets.
func WithoutBootstrapChecksumVerification() BootstrapOption {
	return func(cfg *bootstrapConfig) error {
		cfg.verifySHA = false
		return nil
	}
}

// EnsureDefaultAssets ensures OpenCLIP assets are present locally and returns their paths.
//
// By default this fetches:
//   - repo: amikos/openclip-vit-b-32-laion2b-s34b-b79k-onnx
//   - revision: main
//   - files: text_model.onnx, vision_model.onnx, tokenizer.json, preprocessor_config.json
func EnsureDefaultAssets(opts ...BootstrapOption) (ModelAssets, error) {
	cfg, err := defaultBootstrapConfig()
	if err != nil {
		return ModelAssets{}, err
	}
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return ModelAssets{}, err
		}
	}
	applyDefaultChecksums(&cfg)
	return ensureModelAssets(cfg)
}

func defaultBootstrapConfig() (bootstrapConfig, error) {
	cacheDir := strings.TrimSpace(os.Getenv("ONNXRUNTIME_OPENCLIP_CACHE_DIR"))
	if cacheDir == "" {
		cacheDir = strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR"))
	}
	if cacheDir == "" {
		userCacheDir, err := os.UserCacheDir()
		if err != nil {
			return bootstrapConfig{}, fmt.Errorf("cannot determine user cache directory: %w", err)
		}
		cacheDir = filepath.Join(userCacheDir, "onnx-purego", "openclip")
	}

	return bootstrapConfig{
		repoID:    DefaultBootstrapRepoID,
		revision:  DefaultBootstrapRevision,
		baseURL:   DefaultBootstrapBaseURL,
		cacheDir:  cacheDir,
		hfToken:   strings.TrimSpace(os.Getenv("HF_TOKEN")),
		verifySHA: true,
		shaByFile: map[string]string{},
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}, nil
}

func applyDefaultChecksums(cfg *bootstrapConfig) {
	if !cfg.verifySHA {
		return
	}
	if cfg.repoID != DefaultBootstrapRepoID || cfg.revision != DefaultBootstrapRevision {
		return
	}
	cfg.shaByFile[textModelFileName] = defaultTextModelSHA256
	cfg.shaByFile[visionModelFileName] = defaultVisionSHA256
	cfg.shaByFile[tokenizerFileName] = defaultTokenizerSHA256
	cfg.shaByFile[preprocessorFileName] = defaultPreprocSHA256
}

func ensureModelAssets(cfg bootstrapConfig) (ModelAssets, error) {
	if strings.TrimSpace(cfg.repoID) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap repo ID cannot be empty")
	}
	if strings.TrimSpace(cfg.revision) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap revision cannot be empty")
	}
	if strings.TrimSpace(cfg.cacheDir) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap cache directory cannot be empty")
	}
	if strings.TrimSpace(cfg.baseURL) == "" {
		return ModelAssets{}, fmt.Errorf("bootstrap base URL cannot be empty")
	}
	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{Timeout: 60 * time.Second}
	}

	repoSlug := strings.ReplaceAll(cfg.repoID, "/", "--")
	revisionSlug := strings.ReplaceAll(cfg.revision, "/", "--")
	baseDir := filepath.Join(cfg.cacheDir, repoSlug, revisionSlug)
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return ModelAssets{}, fmt.Errorf("failed to create bootstrap cache directory %q: %w", baseDir, err)
	}

	specs := []bootstrapAssetSpec{
		{fileName: textModelFileName},
		{fileName: visionModelFileName},
		{fileName: tokenizerFileName},
		{fileName: preprocessorFileName},
	}

	paths := map[string]string{}
	for _, spec := range specs {
		targetPath := filepath.Join(baseDir, spec.fileName)
		expected := strings.ToLower(strings.TrimSpace(cfg.shaByFile[spec.fileName]))
		if expected != "" {
			if err := validateSHA256(expected); err != nil {
				return ModelAssets{}, fmt.Errorf("invalid checksum for %s: %w", spec.fileName, err)
			}
		}
		if err := ensureAssetFile(cfg, targetPath, spec.fileName, expected); err != nil {
			return ModelAssets{}, err
		}
		paths[spec.fileName] = targetPath
	}

	return ModelAssets{
		TextModelPath:          paths[textModelFileName],
		VisionModelPath:        paths[visionModelFileName],
		TokenizerPath:          paths[tokenizerFileName],
		PreprocessorConfigPath: paths[preprocessorFileName],
	}, nil
}

func ensureAssetFile(cfg bootstrapConfig, destinationPath string, fileName string, expectedSHA256 string) error {
	if _, err := os.Stat(destinationPath); err == nil {
		if cfg.verifySHA && expectedSHA256 != "" {
			if verifyErr := verifyFileSHA256(destinationPath, expectedSHA256); verifyErr == nil {
				return nil
			}
			if removeErr := os.Remove(destinationPath); removeErr != nil {
				return fmt.Errorf("failed to remove stale asset %q after checksum mismatch: %w", destinationPath, removeErr)
			}
		} else {
			return nil
		}
	}

	url := fmt.Sprintf("%s/%s/resolve/%s/%s",
		strings.TrimRight(cfg.baseURL, "/"),
		cfg.repoID,
		cfg.revision,
		fileName,
	)

	if err := downloadFileWithRetry(cfg.httpClient, url, destinationPath, cfg.hfToken); err != nil {
		return fmt.Errorf("failed to download %s: %w", fileName, err)
	}
	if cfg.verifySHA && expectedSHA256 != "" {
		if err := verifyFileSHA256(destinationPath, expectedSHA256); err != nil {
			_ = os.Remove(destinationPath)
			return fmt.Errorf("downloaded %s failed checksum verification: %w", fileName, err)
		}
	}
	return nil
}

func downloadFileWithRetry(client *http.Client, assetURL string, destinationPath string, hfToken string) error {
	var lastErr error
	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err := downloadFileOnce(client, assetURL, destinationPath, hfToken); err != nil {
			lastErr = err
			if !isRetryableDownloadError(err) || attempt == maxAttempts {
				break
			}
			time.Sleep(time.Duration(attempt) * 500 * time.Millisecond)
			continue
		}
		return nil
	}
	return fmt.Errorf("failed to download %s after %d attempts: %w", assetURL, maxAttempts, lastErr)
}

func downloadFileOnce(client *http.Client, assetURL string, destinationPath string, hfToken string) (err error) {
	req, err := http.NewRequest(http.MethodGet, assetURL, nil)
	if err != nil {
		return err
	}
	if strings.TrimSpace(hfToken) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(hfToken))
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return &downloadStatusError{
			StatusCode: resp.StatusCode,
			URL:        assetURL,
			Body:       strings.TrimSpace(string(body)),
		}
	}

	tempPath := destinationPath + ".tmp"
	if err := os.MkdirAll(filepath.Dir(destinationPath), 0o755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}
	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer func() {
		_ = file.Close()
		if err != nil {
			_ = os.Remove(tempPath)
		}
	}()

	if _, err = io.Copy(file, resp.Body); err != nil {
		return fmt.Errorf("failed to write response body: %w", err)
	}
	if err = file.Sync(); err != nil {
		return fmt.Errorf("failed to flush temp file: %w", err)
	}
	if err = file.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	if err = os.Rename(tempPath, destinationPath); err != nil {
		return fmt.Errorf("failed to move temp file into place: %w", err)
	}
	return nil
}

type downloadStatusError struct {
	StatusCode int
	URL        string
	Body       string
}

func (e *downloadStatusError) Error() string {
	if e == nil {
		return "download status error: <nil>"
	}
	if e.Body == "" {
		return fmt.Sprintf("HTTP %d for %s", e.StatusCode, e.URL)
	}
	return fmt.Sprintf("HTTP %d for %s: %s", e.StatusCode, e.URL, e.Body)
}

func isRetryableDownloadError(err error) bool {
	statusErr := (*downloadStatusError)(nil)
	if !errors.As(err, &statusErr) {
		return false
	}
	if statusErr.StatusCode == http.StatusRequestTimeout || statusErr.StatusCode == http.StatusTooManyRequests {
		return true
	}
	return statusErr.StatusCode >= 500 && statusErr.StatusCode <= 599
}

func verifyFileSHA256(path string, expectedSHA256 string) error {
	expected, err := parseSHA256(expectedSHA256)
	if err != nil {
		return err
	}
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file %q: %w", path, err)
	}
	defer func() {
		_ = file.Close()
	}()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return fmt.Errorf("failed to hash file %q: %w", path, err)
	}
	actual := hasher.Sum(nil)
	if !equalBytes(expected, actual) {
		return fmt.Errorf("checksum mismatch: got %s want %s", hex.EncodeToString(actual), strings.ToLower(expectedSHA256))
	}
	return nil
}

func validateSHA256(value string) error {
	_, err := parseSHA256(value)
	return err
}

func parseSHA256(value string) ([]byte, error) {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) != 64 {
		return nil, fmt.Errorf("checksum must be 64 hex chars, got %d", len(trimmed))
	}
	decoded, err := hex.DecodeString(trimmed)
	if err != nil {
		return nil, fmt.Errorf("checksum is not valid hex: %w", err)
	}
	return decoded, nil
}

func equalBytes(a []byte, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
