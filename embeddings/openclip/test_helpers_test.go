package openclip

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/amikos-tech/pure-onnx/ort"
)

const (
	testOpenCLIPTextModelPathEnv    = "ONNXRUNTIME_TEST_OPENCLIP_TEXT_MODEL_PATH"
	testOpenCLIPVisionModelPathEnv  = "ONNXRUNTIME_TEST_OPENCLIP_VISION_MODEL_PATH"
	testOpenCLIPTokenizerPathEnv    = "ONNXRUNTIME_TEST_OPENCLIP_TOKENIZER_PATH"
	testOpenCLIPPreprocessorPathEnv = "ONNXRUNTIME_TEST_OPENCLIP_PREPROCESSOR_PATH"

	testOpenCLIPTextModelSHAEnv    = "ONNXRUNTIME_TEST_OPENCLIP_TEXT_MODEL_SHA256"
	testOpenCLIPVisionModelSHAEnv  = "ONNXRUNTIME_TEST_OPENCLIP_VISION_MODEL_SHA256"
	testOpenCLIPTokenizerSHAEnv    = "ONNXRUNTIME_TEST_OPENCLIP_TOKENIZER_SHA256"
	testOpenCLIPPreprocessorSHAEnv = "ONNXRUNTIME_TEST_OPENCLIP_PREPROCESSOR_SHA256"
)

func setupORTTestEnvironment(tb testing.TB) func() {
	tb.Helper()

	libPath := strings.TrimSpace(os.Getenv("ONNXRUNTIME_LIB_PATH"))
	if libPath == "" {
		tb.Skip("ONNXRUNTIME_LIB_PATH not set, skipping integration test")
	}

	if err := ort.SetSharedLibraryPath(libPath); err != nil {
		tb.Fatalf("failed to set ONNX Runtime library path: %v", err)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		tb.Fatalf("failed to initialize ONNX Runtime: %v", err)
	}

	return func() {
		if err := ort.DestroyEnvironment(); err != nil {
			tb.Errorf("failed to destroy ONNX Runtime environment: %v", err)
		}
	}
}

func resolveOpenCLIPAssets(tb testing.TB) ModelAssets {
	tb.Helper()

	textModelPath, hasTextPath := resolveOpenCLIPAssetPath(tb, testOpenCLIPTextModelPathEnv, testOpenCLIPTextModelSHAEnv, "text model")
	visionModelPath, hasVisionPath := resolveOpenCLIPAssetPath(tb, testOpenCLIPVisionModelPathEnv, testOpenCLIPVisionModelSHAEnv, "vision model")
	tokenizerPath, hasTokenizerPath := resolveOpenCLIPAssetPath(tb, testOpenCLIPTokenizerPathEnv, testOpenCLIPTokenizerSHAEnv, "tokenizer")
	preprocessorPath, hasPreprocessorPath := resolveOpenCLIPAssetPath(tb, testOpenCLIPPreprocessorPathEnv, testOpenCLIPPreprocessorSHAEnv, "preprocessor config")

	explicitCount := 0
	for _, isSet := range []bool{hasTextPath, hasVisionPath, hasTokenizerPath, hasPreprocessorPath} {
		if isSet {
			explicitCount++
		}
	}

	switch explicitCount {
	case 0:
		tb.Logf(
			"using default OpenCLIP asset bootstrap (%s@%s); explicit test asset paths were not set",
			DefaultBootstrapRepoID,
			DefaultBootstrapRevision,
		)
		return resolveDefaultOpenCLIPAssets(tb)
	case 4:
		return ModelAssets{
			TextModelPath:          textModelPath,
			VisionModelPath:        visionModelPath,
			TokenizerPath:          tokenizerPath,
			PreprocessorConfigPath: preprocessorPath,
		}
	default:
		tb.Fatalf(
			"OpenCLIP explicit test asset paths must set all 4 env vars together (%s, %s, %s, %s); currently set %d",
			testOpenCLIPTextModelPathEnv,
			testOpenCLIPVisionModelPathEnv,
			testOpenCLIPTokenizerPathEnv,
			testOpenCLIPPreprocessorPathEnv,
			explicitCount,
		)
		return ModelAssets{}
	}
}

func resolveOpenCLIPAssetPath(tb testing.TB, envPathKey string, envSHAKey string, label string) (path string, isSet bool) {
	tb.Helper()

	path = strings.TrimSpace(os.Getenv(envPathKey))
	if path == "" {
		return "", false
	}

	if err := validateFilePath(label, path); err != nil {
		tb.Fatalf("%s failed validation: %v", envPathKey, err)
	}

	expectedSHA := strings.TrimSpace(os.Getenv(envSHAKey))
	if expectedSHA != "" {
		if err := verifyOpenCLIPFileSHA256(path, expectedSHA); err != nil {
			tb.Fatalf("%s failed checksum validation: %v", envPathKey, err)
		}
	} else {
		tb.Logf(
			"WARNING: %s is set but %s is unset; skipping SHA256 validation for %s",
			envPathKey,
			envSHAKey,
			label,
		)
	}

	return path, true
}

func resolveDefaultOpenCLIPAssets(tb testing.TB) ModelAssets {
	tb.Helper()

	opts := []BootstrapOption{}
	if cacheRoot := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR")); cacheRoot != "" {
		opts = append(opts, WithBootstrapCacheDir(filepath.Join(cacheRoot, "openclip")))
	}
	if token := strings.TrimSpace(os.Getenv("HF_TOKEN")); token != "" {
		opts = append(opts, WithBootstrapToken(token))
	}

	assets, err := EnsureDefaultAssets(opts...)
	if err != nil {
		tb.Fatalf(
			"failed to bootstrap default OpenCLIP assets (%s@%s): %v",
			DefaultBootstrapRepoID,
			DefaultBootstrapRevision,
			err,
		)
	}

	return assets
}

func verifyOpenCLIPFileSHA256(path string, expected string) error {
	expected = strings.ToLower(strings.TrimSpace(expected))
	if err := validateSHA256(expected); err != nil {
		return fmt.Errorf("invalid sha256 value %q: %w", expected, err)
	}

	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = file.Close()
	}()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return err
	}

	actual := hex.EncodeToString(hash.Sum(nil))
	if subtle.ConstantTimeCompare([]byte(actual), []byte(expected)) != 1 {
		return fmt.Errorf("sha256 mismatch for %s: got %s want %s", path, actual, expected)
	}

	return nil
}

func solidImage(width int, height int, c color.NRGBA) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}
