package ort

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
)

// NewSessionOptions creates a new SessionOptions with default settings.
// Call Destroy() to release resources when done.
// SessionOptions is NOT safe for concurrent use. Do not share across goroutines.
// The returned SessionOptions can be reused to create multiple sessions,
// but you must not destroy it until all sessions created with it are destroyed.
func NewSessionOptions() (*SessionOptions, error) {
	ortCallMu.RLock()
	defer ortCallMu.RUnlock()

	mu.Lock()
	createSessionOptions := createSessionOptionsFunc
	mu.Unlock()

	if createSessionOptions == nil {
		return nil, fmt.Errorf("ONNX Runtime not initialized")
	}

	var handle uintptr
	status := createSessionOptions(&handle)
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return nil, fmt.Errorf("failed to create session options: %s", errMsg)
	}

	return &SessionOptions{
		handle: handle,
	}, nil
}

// Destroy releases the session options resources.
// Must be called after all sessions created with this options object are destroyed.
// Safe to call multiple times.
func (s *SessionOptions) Destroy() {
	if s == nil || s.handle == 0 {
		return
	}

	ortCallMu.RLock()
	mu.Lock()
	releaseFn := releaseSessionOptionsFunc
	mu.Unlock()

	if releaseFn != nil {
		releaseFn(s.handle)
	}
	s.handle = 0
	ortCallMu.RUnlock()
}

// cudaRuntimeAvailable checks whether the CUDA runtime library is available on this system.
func cudaRuntimeAvailable() bool {
	var libName string
	switch runtime.GOOS {
	case "linux":
		libName = "libcuda.so.1"
	case "windows":
		libName = "nvcuda.dll"
	default:
		return false
	}
	handle, err := loadLibrary(libName)
	if err != nil || handle == 0 {
		return false
	}
	closeLibrary(handle)
	return true
}

// AppendCoreML adds the CoreML execution provider (macOS Apple Silicon only).
// This uses the ANE (Apple Neural Engine) and GPU for inference acceleration.
func (s *SessionOptions) AppendCoreML() error {
	if s == nil || s.handle == 0 {
		return fmt.Errorf("session options not initialized")
	}
	return s.appendProviderByName("CoreML")
}

// appendProviderByName appends an execution provider by name using the generic
// SessionOptionsAppendExecutionProvider API.
func (s *SessionOptions) appendProviderByName(name string) error {
	ortCallMu.RLock()
	mu.Lock()
	appendFn := appendExecutionProviderByNameFunc
	mu.Unlock()

	if appendFn == nil {
		ortCallMu.RUnlock()
		return fmt.Errorf("AppendExecutionProvider API not available in the loaded ONNX Runtime library")
	}

	nameBytes, namePtr := GoToCstring(name)
	status := appendFn(s.handle, namePtr)
	runtime.KeepAlive(nameBytes)
	ortCallMu.RUnlock()
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return fmt.Errorf("failed to append execution provider %q: %s", name, errMsg)
	}
	return nil
}

// AppendCUDA adds the CUDA execution provider with the given device ID.
// deviceID=0 uses the default GPU.
// Returns an error if CUDA is not available in the loaded ONNX Runtime library.
func (s *SessionOptions) AppendCUDA(deviceID int) error {
	if s == nil || s.handle == 0 {
		return fmt.Errorf("session options not initialized")
	}

	ortCallMu.RLock()
	mu.Lock()
	appendCUDA := appendExecutionProviderCUDAFunc
	mu.Unlock()

	if appendCUDA == nil {
		ortCallMu.RUnlock()
		return fmt.Errorf("CUDA execution provider not available — use a GPU build of ONNX Runtime (onnxruntime-gpu)")
	}

	status := appendCUDA(s.handle, int32(deviceID))
	ortCallMu.RUnlock()
	if status != 0 {
		errMsg := getErrorMessage(status)
		releaseStatus(status)
		return fmt.Errorf("failed to append CUDA execution provider: %s", errMsg)
	}
	return nil
}

// NewCUDASessionOptions creates session options with an accelerator execution provider.
// Auto-detects available hardware:
//  1. macOS: tries CoreML EP (ANE + GPU) — no CUDA on Apple Silicon
//  2. Linux/Windows: tries CUDA EP if CUDA runtime is installed
//  3. Falls back to nil (CPU) if no accelerator is available
//
// Returns nil (CPU) if no accelerator is available — callers fall back to CPU.
// The caller must call Destroy() on the returned options when done.
//
// Override: set AIGC_CUDA_DEVICE=N to force a specific GPU device ID on CUDA systems.
// Set AIGC_CUDA_DEVICE=-1 to force CPU mode even when an accelerator is available.
func NewCUDASessionOptions() *SessionOptions {
	envVal := os.Getenv("AIGC_CUDA_DEVICE")
	if envVal == "-1" {
		return nil
	}

	opts, err := NewSessionOptions()
	if err != nil {
		return nil
	}

	switch runtime.GOOS {
	case "darwin":
		// macOS: CoreML EP only works with some ONNX models (ANE
		// operator limitations). Require explicit opt-in via AIGC_CUDA_DEVICE
		// to avoid silent inference failures on unsupported models.
		if envVal == "" {
			opts.Destroy()
			return nil
		}
		if err := opts.AppendCoreML(); err != nil {
			opts.Destroy()
			return nil
		}
		return opts

	case "linux", "windows":
		// Linux/Windows: try CUDA EP if CUDA runtime is available
		mu.Lock()
		hasCUDAFn := appendExecutionProviderCUDAFunc != nil
		mu.Unlock()
		if !hasCUDAFn || !cudaRuntimeAvailable() {
			opts.Destroy()
			return nil
		}
		deviceID := 0
		if envVal != "" {
			id, err := strconv.Atoi(envVal)
			if err != nil || id < 0 {
				opts.Destroy()
				return nil
			}
			deviceID = id
		}
		if err := opts.AppendCUDA(deviceID); err != nil {
			opts.Destroy()
			return nil
		}
		return opts

	default:
		opts.Destroy()
		return nil
	}
}