package openclip

import (
	"image"
	"image/color"
	"math"
	"strings"
	"testing"
)

func TestEmbedTextsAndImagesWithOpenCLIPModel(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	assets := resolveOpenCLIPAssets(t)

	embedder, err := NewEmbedder(
		assets.TextModelPath,
		assets.VisionModelPath,
		assets.TokenizerPath,
		assets.PreprocessorConfigPath,
	)
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

func TestOpenCLIPFailsWithWrongInputOutputNames(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	assets := resolveOpenCLIPAssets(t)
	embedder, err := NewEmbedder(
		assets.TextModelPath,
		assets.VisionModelPath,
		assets.TokenizerPath,
		assets.PreprocessorConfigPath,
		WithTextInputOutputNames("bad_input_ids", "bad_attention_mask", "bad_text_output"),
	)
	if err != nil {
		t.Fatalf("failed to create openclip embedder: %v", err)
	}
	defer func() {
		if closeErr := embedder.Close(); closeErr != nil {
			t.Errorf("failed to close openclip embedder: %v", closeErr)
		}
	}()

	_, err = embedder.EmbedTexts([]string{"a photo of a cat"})
	if err == nil {
		t.Fatalf("expected text inference to fail with incorrect text input/output names")
	}
	assertErrorContainsAll(t, err, []string{"text embedding inference failed"})
	assertErrorContainsAny(
		t,
		err,
		[]string{"bad_input_ids", "bad_attention_mask", "bad_text_output"},
	)
}

func TestOpenCLIPFailsWithWrongEmbeddingDimension(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	assets := resolveOpenCLIPAssets(t)
	embedder, err := NewEmbedder(
		assets.TextModelPath,
		assets.VisionModelPath,
		assets.TokenizerPath,
		assets.PreprocessorConfigPath,
		WithEmbeddingDimension(OutputEmbeddingDimension+1),
	)
	if err != nil {
		t.Fatalf("failed to create openclip embedder: %v", err)
	}
	defer func() {
		if closeErr := embedder.Close(); closeErr != nil {
			t.Errorf("failed to close openclip embedder: %v", closeErr)
		}
	}()

	_, err = embedder.EmbedTexts([]string{"a photo of a cat"})
	if err == nil {
		t.Fatalf("expected text inference to fail with incompatible embedding dimension")
	}
	assertErrorContainsAny(
		t,
		err,
		[]string{"text output length mismatch", "invalid dimensions for output"},
	)
}

func TestOpenCLIPFailsWithImageSizeMismatch(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	assets := resolveOpenCLIPAssets(t)
	embedder, err := NewEmbedder(
		assets.TextModelPath,
		assets.VisionModelPath,
		assets.TokenizerPath,
		assets.PreprocessorConfigPath,
		WithImageSize(256),
	)
	if err != nil {
		t.Fatalf("failed to create openclip embedder: %v", err)
	}
	defer func() {
		if closeErr := embedder.Close(); closeErr != nil {
			t.Errorf("failed to close openclip embedder: %v", closeErr)
		}
	}()

	_, err = embedder.EmbedImages([]image.Image{
		solidImage(256, 256, color.NRGBA{R: 64, G: 64, B: 64, A: 255}),
	})
	if err == nil {
		t.Fatalf("expected image inference to fail with forced non-default image size")
	}
	assertErrorContainsAll(t, err, []string{"vision embedding inference failed"})
	assertErrorContainsAny(
		t,
		err,
		[]string{"pixel_values", "invalid dimensions", "dimension"},
	)
}

func TestOpenCLIPErrorsAfterClose(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	assets := resolveOpenCLIPAssets(t)
	embedder, err := NewEmbedder(
		assets.TextModelPath,
		assets.VisionModelPath,
		assets.TokenizerPath,
		assets.PreprocessorConfigPath,
	)
	if err != nil {
		t.Fatalf("failed to create openclip embedder: %v", err)
	}
	if err := embedder.Close(); err != nil {
		t.Fatalf("failed to close openclip embedder: %v", err)
	}

	_, err = embedder.EmbedTexts([]string{"a photo of a cat"})
	assertErrorContainsAll(t, err, []string{"embedder has been closed"})

	_, err = embedder.EmbedImages([]image.Image{
		solidImage(224, 224, color.NRGBA{R: 128, G: 128, B: 128, A: 255}),
	})
	assertErrorContainsAll(t, err, []string{"embedder has been closed"})
}

func TestOpenCLIPCloseIsIdempotent(t *testing.T) {
	cleanup := setupORTTestEnvironment(t)
	defer cleanup()

	assets := resolveOpenCLIPAssets(t)
	embedder, err := NewEmbedder(
		assets.TextModelPath,
		assets.VisionModelPath,
		assets.TokenizerPath,
		assets.PreprocessorConfigPath,
	)
	if err != nil {
		t.Fatalf("failed to create openclip embedder: %v", err)
	}

	if err := embedder.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}
	if err := embedder.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
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

func assertErrorContainsAny(t *testing.T, err error, fragments []string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	got := strings.ToLower(err.Error())
	for _, fragment := range fragments {
		if strings.Contains(got, strings.ToLower(fragment)) {
			return
		}
	}
	t.Fatalf("error %q did not contain any of expected fragments: %v", err.Error(), fragments)
}

func assertErrorContainsAll(t *testing.T, err error, fragments []string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	got := strings.ToLower(err.Error())
	for _, fragment := range fragments {
		if !strings.Contains(got, strings.ToLower(fragment)) {
			t.Fatalf("error %q did not contain required fragment %q", err.Error(), fragment)
		}
	}
}
