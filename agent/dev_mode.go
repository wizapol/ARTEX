package agent

import (
	"os"
	"strings"
)

// devModeEnabled reports whether ARTEX is running in development mode.
// When enabled, authorization/scope verification is skipped and default scope is auto-created.
// Set ARTEX_DEV_MODE to "1", "true", "on", or "yes" (case- and space-insensitive) to enable.
func devModeEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ARTEX_DEV_MODE"))) {
	case "1", "true", "on", "yes":
		return true
	}
	return false
}

// devModeDefaultScope returns a default scope for development mode.
// This allows testing without requiring explicit authorization setup.
func devModeDefaultScope() string {
	if !devModeEnabled() {
		return ""
	}
	return "*"  // Wildcard: allows testing all targets
}
