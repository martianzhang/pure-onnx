package openclip

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	openCLIPGoldenRowsRequestTimeout = 60 * time.Second
	openCLIPGoldenDefaultDatasetPath = "openclip_endpoint_golden/v1/openclip_vit_b_32_laion2b_s34b_b79k_prefix64_v1.jsonl"
)

type openCLIPGoldenDatasetRow struct {
	ID          string                    `json:"id"`
	Text        string                    `json:"text"`
	Image       openCLIPGoldenImageRecipe `json:"image"`
	TextPrefix  []float32                 `json:"text_prefix"`
	ImagePrefix []float32                 `json:"image_prefix"`
	LogitsRow   []float32                 `json:"logits_row"`
}

type openCLIPGoldenImageRecipe struct {
	Kind      string `json:"kind"`
	Width     int    `json:"width"`
	Height    int    `json:"height"`
	PNGBase64 string `json:"png_base64"`
	Color     []int  `json:"color"`
	ColorA    []int  `json:"color_a"`
	ColorB    []int  `json:"color_b"`
	BlockSize int    `json:"block_size"`
}

func TestOpenCLIPGoldenDatasetParity(t *testing.T) {
	jsonlURL := strings.TrimSpace(os.Getenv("ONNXRUNTIME_TEST_OPENCLIP_GOLDEN_JSONL_URL"))
	if jsonlURL == "" {
		repo := strings.TrimSpace(os.Getenv("HF_DATASET_REPO"))
		if repo == "" {
			t.Skip("set ONNXRUNTIME_TEST_OPENCLIP_GOLDEN_JSONL_URL or HF_DATASET_REPO to run OpenCLIP golden dataset parity test")
		}
		jsonlURL = fmt.Sprintf("https://huggingface.co/datasets/%s/resolve/main/%s", repo, openCLIPGoldenDefaultDatasetPath)
	}

	rows, err := downloadOpenCLIPGoldenRows(jsonlURL, strings.TrimSpace(os.Getenv("HF_TOKEN")))
	if err != nil {
		t.Skipf("unable to load OpenCLIP golden dataset from %s: %v", jsonlURL, err)
	}
	if len(rows) == 0 {
		t.Fatalf("OpenCLIP golden dataset is empty")
	}

	tolerance, err := parseOpenCLIPFloat32Env("ONNXRUNTIME_TEST_OPENCLIP_GOLDEN_TOLERANCE", 1e-4)
	if err != nil {
		t.Fatalf("invalid ONNXRUNTIME_TEST_OPENCLIP_GOLDEN_TOLERANCE: %v", err)
	}

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
		t.Fatalf("failed to create OpenCLIP embedder: %v", err)
	}
	defer func() {
		if closeErr := embedder.Close(); closeErr != nil {
			t.Errorf("failed to close OpenCLIP embedder: %v", closeErr)
		}
	}()

	texts := make([]string, len(rows))
	images := make([]image.Image, len(rows))
	for i := range rows {
		row := rows[i]
		if strings.TrimSpace(row.Text) == "" {
			t.Fatalf("row %d has empty text", i)
		}
		if len(row.TextPrefix) == 0 {
			t.Fatalf("row %d has empty text_prefix", i)
		}
		if len(row.ImagePrefix) == 0 {
			t.Fatalf("row %d has empty image_prefix", i)
		}
		if len(row.LogitsRow) == 0 {
			t.Fatalf("row %d has empty logits_row", i)
		}

		imageValue, renderErr := renderOpenCLIPGoldenImage(row.Image)
		if renderErr != nil {
			t.Fatalf("row %d image recipe is invalid: %v", i, renderErr)
		}
		texts[i] = row.Text
		images[i] = imageValue
	}

	textEmbeddings, err := embedder.EmbedTexts(texts)
	if err != nil {
		t.Fatalf("EmbedTexts failed: %v", err)
	}
	imageEmbeddings, err := embedder.EmbedImages(images)
	if err != nil {
		t.Fatalf("EmbedImages failed: %v", err)
	}
	if len(textEmbeddings) != len(rows) {
		t.Fatalf("text embedding row count mismatch: got %d want %d", len(textEmbeddings), len(rows))
	}
	if len(imageEmbeddings) != len(rows) {
		t.Fatalf("image embedding row count mismatch: got %d want %d", len(imageEmbeddings), len(rows))
	}

	logits, err := CLIPSimilarityLogits(imageEmbeddings, textEmbeddings, DefaultCLIPLogitScale)
	if err != nil {
		t.Fatalf("CLIPSimilarityLogits failed: %v", err)
	}
	if len(logits) != len(rows) {
		t.Fatalf("logits row count mismatch: got %d want %d", len(logits), len(rows))
	}

	for rowIndex := range rows {
		row := rows[rowIndex]
		assertOpenCLIPPrefixNear(
			t,
			fmt.Sprintf("row %d text prefix", rowIndex),
			textEmbeddings[rowIndex],
			row.TextPrefix,
			tolerance,
		)
		assertOpenCLIPPrefixNear(
			t,
			fmt.Sprintf("row %d image prefix", rowIndex),
			imageEmbeddings[rowIndex],
			row.ImagePrefix,
			tolerance,
		)

		if len(row.LogitsRow) != len(rows) {
			t.Fatalf("row %d logits_row length mismatch: got %d want %d", rowIndex, len(row.LogitsRow), len(rows))
		}
		if len(logits[rowIndex]) != len(rows) {
			t.Fatalf("row %d computed logits length mismatch: got %d want %d", rowIndex, len(logits[rowIndex]), len(rows))
		}

		for col := range row.LogitsRow {
			diff := math.Abs(float64(logits[rowIndex][col] - row.LogitsRow[col]))
			if diff > float64(tolerance) {
				t.Fatalf(
					"row %d logits[%d] mismatch: got %.8f want %.8f tolerance %.8f",
					rowIndex,
					col,
					logits[rowIndex][col],
					row.LogitsRow[col],
					tolerance,
				)
			}
		}
	}
}

func downloadOpenCLIPGoldenRows(url string, token string) ([]openCLIPGoldenDatasetRow, error) {
	request, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{Timeout: openCLIPGoldenRowsRequestTimeout}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = response.Body.Close()
	}()

	if response.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(response.Body, 512))
		return nil, fmt.Errorf("unexpected HTTP status %d: %s", response.StatusCode, string(snippet))
	}

	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
	rows := make([]openCLIPGoldenDatasetRow, 0, 64)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row openCLIPGoldenDatasetRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("invalid jsonl row: %w", err)
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return rows, nil
}

func parseOpenCLIPFloat32Env(key string, fallback float32) (float32, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(raw, 32)
	if err != nil {
		return fallback, fmt.Errorf("%s=%q is not a valid float32: %w", key, raw, err)
	}
	return float32(parsed), nil
}

func renderOpenCLIPGoldenImage(recipe openCLIPGoldenImageRecipe) (image.Image, error) {
	kind := strings.ToLower(strings.TrimSpace(recipe.Kind))
	if kind == "" {
		kind = "solid"
	}

	switch kind {
	case "solid":
		if recipe.Width <= 0 || recipe.Height <= 0 {
			return nil, fmt.Errorf("width and height must be > 0, got %d x %d", recipe.Width, recipe.Height)
		}
		rgb, err := parseRGB(recipe.Color, "color")
		if err != nil {
			return nil, err
		}
		return solidImage(
			recipe.Width,
			recipe.Height,
			color.NRGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255},
		), nil
	case "checkerboard":
		if recipe.Width <= 0 || recipe.Height <= 0 {
			return nil, fmt.Errorf("width and height must be > 0, got %d x %d", recipe.Width, recipe.Height)
		}
		if recipe.BlockSize <= 0 {
			return nil, fmt.Errorf("block_size must be > 0 for checkerboard, got %d", recipe.BlockSize)
		}
		a, err := parseRGB(recipe.ColorA, "color_a")
		if err != nil {
			return nil, err
		}
		b, err := parseRGB(recipe.ColorB, "color_b")
		if err != nil {
			return nil, err
		}
		return checkerboardImage(
			recipe.Width,
			recipe.Height,
			recipe.BlockSize,
			color.NRGBA{R: a[0], G: a[1], B: a[2], A: 255},
			color.NRGBA{R: b[0], G: b[1], B: b[2], A: 255},
		), nil
	case "png_base64":
		if strings.TrimSpace(recipe.PNGBase64) == "" {
			return nil, fmt.Errorf("png_base64 payload is empty")
		}
		raw, err := base64.StdEncoding.DecodeString(recipe.PNGBase64)
		if err != nil {
			return nil, fmt.Errorf("invalid png_base64 payload: %w", err)
		}
		img, err := png.Decode(bytes.NewReader(raw))
		if err != nil {
			return nil, fmt.Errorf("invalid PNG payload: %w", err)
		}
		return img, nil
	default:
		return nil, fmt.Errorf("unsupported image kind %q", recipe.Kind)
	}
}

func checkerboardImage(width int, height int, blockSize int, colorA color.NRGBA, colorB color.NRGBA) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			useA := ((x / blockSize) + (y / blockSize)) % 2
			if useA == 0 {
				img.SetNRGBA(x, y, colorA)
				continue
			}
			img.SetNRGBA(x, y, colorB)
		}
	}
	return img
}

func parseRGB(values []int, label string) ([3]uint8, error) {
	if len(values) != 3 {
		return [3]uint8{}, fmt.Errorf("%s must contain exactly 3 values, got %d", label, len(values))
	}
	result := [3]uint8{}
	for i := range values {
		if values[i] < 0 || values[i] > 255 {
			return [3]uint8{}, fmt.Errorf("%s[%d] must be between 0 and 255, got %d", label, i, values[i])
		}
		result[i] = uint8(values[i])
	}
	return result, nil
}

func assertOpenCLIPPrefixNear(t *testing.T, label string, got []float32, want []float32, tolerance float32) {
	t.Helper()

	if len(got) < len(want) {
		t.Fatalf("%s length mismatch: got %d want at least %d", label, len(got), len(want))
	}
	for i := range want {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Fatalf("%s[%d] mismatch: got %.8f want %.8f tolerance %.8f", label, i, got[i], want[i], tolerance)
		}
	}
}
