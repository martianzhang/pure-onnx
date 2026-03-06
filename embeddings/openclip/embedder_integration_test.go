package openclip

import (
	"image"
	"image/color"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/amikos-tech/pure-onnx/ort"
)

func TestEmbedTextsAndImagesWithOpenCLIPModel(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	textModelPath := os.Getenv("ONNXRUNTIME_TEST_OPENCLIP_TEXT_MODEL_PATH")
	visionModelPath := os.Getenv("ONNXRUNTIME_TEST_OPENCLIP_VISION_MODEL_PATH")
	tokenizerPath := os.Getenv("ONNXRUNTIME_TEST_OPENCLIP_TOKENIZER_PATH")
	preprocessorPath := os.Getenv("ONNXRUNTIME_TEST_OPENCLIP_PREPROCESSOR_PATH")
	if textModelPath == "" || visionModelPath == "" || tokenizerPath == "" || preprocessorPath == "" {
		opts := []BootstrapOption{}
		if cacheDir := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_MODEL_CACHE_DIR")); cacheDir != "" {
			opts = append(opts, WithBootstrapCacheDir(cacheDir))
		}
		if token := strings.TrimSpace(os.Getenv("HF_TOKEN")); token != "" {
			opts = append(opts, WithBootstrapToken(token))
		}
		assets, err := EnsureDefaultAssets(opts...)
		if err != nil {
			t.Fatalf(
				"failed to bootstrap default OpenCLIP assets (%s@%s): %v",
				DefaultBootstrapRepoID,
				DefaultBootstrapRevision,
				err,
			)
		}
		textModelPath = assets.TextModelPath
		visionModelPath = assets.VisionModelPath
		tokenizerPath = assets.TokenizerPath
		preprocessorPath = assets.PreprocessorConfigPath
	}

	embedder, err := NewEmbedder(textModelPath, visionModelPath, tokenizerPath, preprocessorPath)
	if err != nil {
		t.Fatalf("failed to create openclip embedder: %v", err)
	}
	defer func() {
		if closeErr := embedder.Close(); closeErr != nil {
			t.Errorf("failed to close openclip embedder: %v", closeErr)
		}
	}()

	textEmbeddings, err := embedder.EmbedTexts([]string{"a photo of a cat", "a photo of a dog"})
	if err != nil {
		t.Fatalf("EmbedTexts failed: %v", err)
	}
	if len(textEmbeddings) != 2 {
		t.Fatalf("unexpected text embedding row count: got %d, want 2", len(textEmbeddings))
	}
	for i, row := range textEmbeddings {
		if len(row) != int(OutputEmbeddingDimension) {
			t.Fatalf("unexpected text embedding width at row %d: got %d, want %d", i, len(row), OutputEmbeddingDimension)
		}
		assertFiniteVector(t, "text embedding", row)
		assertApproxUnitNormIntegration(t, "text embedding", row, 1e-4)
	}
	if len(embedder.textSessionsByBatch) != 1 {
		t.Fatalf("expected one cached text session, got %d", len(embedder.textSessionsByBatch))
	}

	imageEmbeddings, err := embedder.EmbedImages([]image.Image{
		solidImage(224, 224, color.NRGBA{R: 220, G: 220, B: 220, A: 255}),
		solidImage(224, 224, color.NRGBA{R: 40, G: 40, B: 40, A: 255}),
	})
	if err != nil {
		t.Fatalf("EmbedImages failed: %v", err)
	}
	if len(imageEmbeddings) != 2 {
		t.Fatalf("unexpected image embedding row count: got %d, want 2", len(imageEmbeddings))
	}
	for i, row := range imageEmbeddings {
		if len(row) != int(OutputEmbeddingDimension) {
			t.Fatalf("unexpected image embedding width at row %d: got %d, want %d", i, len(row), OutputEmbeddingDimension)
		}
		assertFiniteVector(t, "image embedding", row)
		assertApproxUnitNormIntegration(t, "image embedding", row, 1e-4)
	}
	if len(embedder.visionSessionsByBatch) != 1 {
		t.Fatalf("expected one cached vision session, got %d", len(embedder.visionSessionsByBatch))
	}

	logits, err := CLIPSimilarityLogits(imageEmbeddings, textEmbeddings, DefaultCLIPLogitScale)
	if err != nil {
		t.Fatalf("CLIPSimilarityLogits failed: %v", err)
	}
	if len(logits) != 2 || len(logits[0]) != 2 || len(logits[1]) != 2 {
		t.Fatalf("unexpected similarity logits shape: %#v", logits)
	}
	for i := range logits {
		for j := range logits[i] {
			if math.IsNaN(float64(logits[i][j])) || math.IsInf(float64(logits[i][j]), 0) {
				t.Fatalf("unexpected non-finite logit value at [%d][%d]: %f", i, j, logits[i][j])
			}
		}
	}
}

func setupORTTestEnvironment(tb testing.TB) func() {
	tb.Helper()

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
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

func solidImage(width int, height int, c color.NRGBA) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

func assertApproxUnitNormIntegration(t *testing.T, label string, values []float32, epsilon float64) {
	t.Helper()
	var normSquared float64
	for _, value := range values {
		normSquared += float64(value) * float64(value)
	}
	norm := math.Sqrt(normSquared)
	if math.Abs(norm-1.0) > epsilon {
		t.Fatalf("%s norm mismatch: got %.8f, want ~1.0", label, norm)
	}
}

func assertFiniteVector(t *testing.T, label string, values []float32) {
	t.Helper()
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%s has non-finite value at index %d: %f", label, i, value)
		}
	}
}
