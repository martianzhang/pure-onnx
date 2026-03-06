package openclip

import (
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWithMaxCachedBatchSessionsValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithMaxCachedBatchSessions(0)(&cfg); err == nil {
		t.Fatalf("expected validation error for zero max cached sessions")
	}
	if err := WithMaxCachedBatchSessions(3)(&cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	if cfg.maxCachedBatchCount != 3 {
		t.Fatalf("unexpected maxCachedBatchCount: got %d, want 3", cfg.maxCachedBatchCount)
	}
}

func TestWithEmbeddingDimensionValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithEmbeddingDimension(0)(&cfg); err == nil {
		t.Fatalf("expected validation error for zero embedding dimension")
	}
	if err := WithEmbeddingDimension(768)(&cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
	if cfg.embeddingDimension != 768 {
		t.Fatalf("unexpected embeddingDimension: got %d, want 768", cfg.embeddingDimension)
	}
}

func TestWithInputOutputNamesValidation(t *testing.T) {
	cfg := defaultConfig()
	if err := WithTextInputOutputNames("", "mask", "output")(&cfg); err == nil {
		t.Fatalf("expected validation error for empty text input name")
	}
	if err := WithVisionInputOutputNames("", "output")(&cfg); err == nil {
		t.Fatalf("expected validation error for empty vision input name")
	}

	if err := WithTextInputOutputNames("ids", "mask", "text_embedding")(&cfg); err != nil {
		t.Fatalf("unexpected text names validation error: %v", err)
	}
	if err := WithVisionInputOutputNames("pixels", "image_embedding")(&cfg); err != nil {
		t.Fatalf("unexpected vision names validation error: %v", err)
	}
	if cfg.textInputIDsName != "ids" || cfg.textAttentionMask != "mask" || cfg.textOutputName != "text_embedding" {
		t.Fatalf("unexpected text names after override: %+v", cfg)
	}
	if cfg.visionInputName != "pixels" || cfg.visionOutputName != "image_embedding" {
		t.Fatalf("unexpected vision names after override: %+v", cfg)
	}
}

func TestNormalizationOptions(t *testing.T) {
	cfg := defaultConfig()

	if err := WithoutL2Normalization()(&cfg); err != nil {
		t.Fatalf("WithoutL2Normalization failed: %v", err)
	}
	if cfg.l2Normalize {
		t.Fatalf("expected l2Normalize=false after WithoutL2Normalization")
	}

	if err := WithL2Normalization()(&cfg); err != nil {
		t.Fatalf("WithL2Normalization failed: %v", err)
	}
	if !cfg.l2Normalize {
		t.Fatalf("expected l2Normalize=true after WithL2Normalization")
	}
}

func TestCosineSimilarity(t *testing.T) {
	got, err := CosineSimilarity([]float32{1, 0}, []float32{0, 1})
	if err != nil {
		t.Fatalf("CosineSimilarity failed: %v", err)
	}
	if !float32Near(got, 0, 1e-6) {
		t.Fatalf("unexpected cosine value: got %.6f, want 0", got)
	}

	got, err = CosineSimilarity([]float32{1, 2, 3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("CosineSimilarity failed: %v", err)
	}
	if !float32Near(got, 1, 1e-6) {
		t.Fatalf("unexpected cosine value: got %.6f, want 1", got)
	}
}

func TestCosineSimilarityValidation(t *testing.T) {
	_, err := CosineSimilarity(nil, []float32{1})
	if err == nil || !strings.Contains(err.Error(), "non-empty") {
		t.Fatalf("expected non-empty validation error, got: %v", err)
	}
	_, err = CosineSimilarity([]float32{1}, []float32{1, 2})
	if err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("expected dimension mismatch validation error, got: %v", err)
	}
}

func TestCosineSimilarityZeroNorm(t *testing.T) {
	got, err := CosineSimilarity([]float32{0, 0}, []float32{1, 2})
	if err != nil {
		t.Fatalf("CosineSimilarity failed: %v", err)
	}
	if got != 0 {
		t.Fatalf("expected zero cosine similarity for zero norm vector, got %f", got)
	}
}

func TestCLIPSimilarityLogits(t *testing.T) {
	imageEmbeddings := [][]float32{
		{1, 0},
		{0, 1},
	}
	textEmbeddings := [][]float32{
		{1, 0},
		{0, 1},
	}
	logits, err := CLIPSimilarityLogits(imageEmbeddings, textEmbeddings, 2)
	if err != nil {
		t.Fatalf("CLIPSimilarityLogits failed: %v", err)
	}
	if len(logits) != 2 || len(logits[0]) != 2 || len(logits[1]) != 2 {
		t.Fatalf("unexpected logits shape: %#v", logits)
	}

	expected := [][]float32{
		{2, 0},
		{0, 2},
	}
	for i := range expected {
		for j := range expected[i] {
			if !float32Near(logits[i][j], expected[i][j], 1e-6) {
				t.Fatalf("unexpected logits[%d][%d]: got %.6f, want %.6f", i, j, logits[i][j], expected[i][j])
			}
		}
	}
}

func TestCLIPSimilarityLogitsValidation(t *testing.T) {
	_, err := CLIPSimilarityLogits([][]float32{{1, 2}}, [][]float32{{1}}, 1)
	if err == nil || !strings.Contains(err.Error(), "dimension mismatch") {
		t.Fatalf("expected embedding dimension mismatch error, got: %v", err)
	}

	_, err = CLIPSimilarityLogits([][]float32{{1}}, [][]float32{{1}}, float32(math.Inf(1)))
	if err == nil || !strings.Contains(err.Error(), "finite") {
		t.Fatalf("expected finite logit scale validation error, got: %v", err)
	}
}

func TestPostProcessDenseEmbeddings(t *testing.T) {
	embeddings, err := postProcessDenseEmbeddings(
		[]float32{
			3, 4,
			5, 12,
		},
		2,
		2,
		true,
		"text output",
	)
	if err != nil {
		t.Fatalf("postProcessDenseEmbeddings failed: %v", err)
	}
	if len(embeddings) != 2 {
		t.Fatalf("unexpected embedding row count: got %d, want 2", len(embeddings))
	}

	assertApproxUnitNorm(t, "row 0", embeddings[0], 1e-6)
	assertApproxUnitNorm(t, "row 1", embeddings[1], 1e-6)
}

func TestPostProcessDenseEmbeddingsValidation(t *testing.T) {
	_, err := postProcessDenseEmbeddings([]float32{1, 2, 3}, 2, 2, true, "text output")
	if err == nil || !strings.Contains(err.Error(), "length mismatch") {
		t.Fatalf("expected length mismatch error, got: %v", err)
	}
}

func TestParseSizeField(t *testing.T) {
	size, err := parseSizeField([]byte(`224`))
	if err != nil {
		t.Fatalf("parseSizeField integer failed: %v", err)
	}
	if size != 224 {
		t.Fatalf("unexpected size: got %d, want 224", size)
	}

	size, err = parseSizeField([]byte(`{"shortest_edge":256}`))
	if err != nil {
		t.Fatalf("parseSizeField shortest_edge failed: %v", err)
	}
	if size != 256 {
		t.Fatalf("unexpected size: got %d, want 256", size)
	}

	size, err = parseSizeField([]byte(`{"height":224,"width":224}`))
	if err != nil {
		t.Fatalf("parseSizeField crop object failed: %v", err)
	}
	if size != 224 {
		t.Fatalf("unexpected size: got %d, want 224", size)
	}

	if _, err := parseSizeField([]byte(`{"height":224,"width":256}`)); err == nil {
		t.Fatalf("expected non-square crop error")
	}
}

func TestLoadPreprocessorConfig(t *testing.T) {
	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "preprocessor_config.json")
	payload := `{
		"do_resize": true,
		"size": {"shortest_edge": 300},
		"do_center_crop": true,
		"crop_size": {"height": 256, "width": 256},
		"do_normalize": true,
		"do_rescale": true,
		"rescale_factor": 0.0039215686,
		"do_convert_rgb": true,
		"image_mean": [0.1, 0.2, 0.3],
		"image_std": [0.9, 0.8, 0.7]
	}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("failed to write config fixture: %v", err)
	}

	cfg, err := loadPreprocessorConfig(path)
	if err != nil {
		t.Fatalf("loadPreprocessorConfig failed: %v", err)
	}
	if !cfg.doResize || cfg.resizeShortestEdge != 300 {
		t.Fatalf("unexpected resize config: %+v", cfg)
	}
	if !cfg.doCenterCrop || cfg.cropSize != 256 {
		t.Fatalf("unexpected crop config: %+v", cfg)
	}
	if !cfg.doNormalize || !cfg.doRescale || !cfg.doConvertRGB {
		t.Fatalf("unexpected bool flags: %+v", cfg)
	}
	if !float32Near(cfg.rescaleFactor, 0.0039215686, 1e-7) {
		t.Fatalf("unexpected rescaleFactor: got %.10f", cfg.rescaleFactor)
	}
	if !float32Near(cfg.imageMean[0], 0.1, 1e-7) || !float32Near(cfg.imageStd[2], 0.7, 1e-7) {
		t.Fatalf("unexpected mean/std config: %+v", cfg)
	}
}

func TestPreprocessImagesIntoCHW(t *testing.T) {
	embedder := &Embedder{
		imageSize: 2,
		preprocessor: clipPreprocessorConfig{
			doResize:      false,
			doCenterCrop:  false,
			doNormalize:   false,
			doRescale:     false,
			doConvertRGB:  true,
			imageMean:     [3]float32{0, 0, 0},
			imageStd:      [3]float32{1, 1, 1},
			rescaleFactor: 1,
		},
	}

	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 10, G: 20, B: 30, A: 255})
	img.SetNRGBA(1, 0, color.NRGBA{R: 40, G: 50, B: 60, A: 255})
	img.SetNRGBA(0, 1, color.NRGBA{R: 70, G: 80, B: 90, A: 255})
	img.SetNRGBA(1, 1, color.NRGBA{R: 100, G: 110, B: 120, A: 255})

	pixelValues := make([]float32, 3*2*2)
	if err := embedder.preprocessImagesInto([]image.Image{img}, pixelValues); err != nil {
		t.Fatalf("preprocessImagesInto failed: %v", err)
	}

	expected := []float32{
		10, 40, 70, 100,
		20, 50, 80, 110,
		30, 60, 90, 120,
	}
	if len(pixelValues) != len(expected) {
		t.Fatalf("unexpected pixel buffer length: got %d, want %d", len(pixelValues), len(expected))
	}
	for i := range expected {
		if !float32Near(pixelValues[i], expected[i], 1e-6) {
			t.Fatalf("unexpected pixelValues[%d]: got %.6f, want %.6f", i, pixelValues[i], expected[i])
		}
	}
}

func TestPreprocessImagesIntoValidation(t *testing.T) {
	embedder := &Embedder{
		imageSize: 2,
		preprocessor: clipPreprocessorConfig{
			doConvertRGB: true,
			doRescale:    false,
			doNormalize:  false,
		},
	}
	pixelValues := make([]float32, 12)

	err := embedder.preprocessImagesInto([]image.Image{nil}, pixelValues)
	if err == nil || !strings.Contains(err.Error(), "is nil") {
		t.Fatalf("expected nil image validation error, got: %v", err)
	}
}

func TestGetRGBConvertsAccordingToConvertRGBFlag(t *testing.T) {
	colorValue := color.NRGBA{R: 10, G: 20, B: 30, A: 255}
	r, g, b := getRGB(colorValue, true)
	if !float32Near(r, 10, 1e-7) || !float32Near(g, 20, 1e-7) || !float32Near(b, 30, 1e-7) {
		t.Fatalf("expected RGB conversion path to preserve channels, got %.2f,%.2f,%.2f", r, g, b)
	}
	r, g, b = getRGB(colorValue, false)
	expectedGray := color.GrayModel.Convert(colorValue).(color.Gray).Y
	if !float32Near(r, float32(expectedGray), 1e-7) ||
		!float32Near(g, float32(expectedGray), 1e-7) ||
		!float32Near(b, float32(expectedGray), 1e-7) {
		t.Fatalf("expected non-convert path to map to grayscale, got %.2f,%.2f,%.2f", r, g, b)
	}
}

func TestEmbedTextValidation(t *testing.T) {
	var embedder *Embedder
	_, err := embedder.EmbedText("test")
	if err == nil || !strings.Contains(err.Error(), "embedder is nil") {
		t.Fatalf("expected nil embedder error, got: %v", err)
	}
}

func float32Near(a float32, b float32, epsilon float32) bool {
	diff := float32(math.Abs(float64(a - b)))
	return diff <= epsilon
}

func assertApproxUnitNorm(t *testing.T, label string, values []float32, epsilon float64) {
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
