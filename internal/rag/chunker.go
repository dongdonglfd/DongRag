package rag

import (
	"reflect"
	"strings"
	"unicode/utf8"
)

type SourceBlock struct {
	Text     string
	Metadata map[string]any
}

type ChunkPart struct {
	Text     string
	Metadata map[string]any
}

func ChunkText(text string, size, overlap int) []string {
	return chunkText(text, size, overlap)
}

func ParseMarkdownBlocks(text string) []SourceBlock {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	headingPath := make([]string, 0, 6)
	var current []string
	inFence := false
	fenceMarker := ""
	fenceLanguage := ""
	blocks := make([]SourceBlock, 0)

	flush := func(blockType, language string) {
		content := strings.TrimSpace(strings.Join(current, "\n"))
		if content == "" {
			current = nil
			return
		}
		metadata := map[string]any{"block_type": blockType}
		if len(headingPath) > 0 {
			metadata["heading_path"] = append([]string(nil), headingPath...)
		}
		if language != "" {
			metadata["language"] = language
		}
		blocks = append(blocks, SourceBlock{Text: content, Metadata: metadata})
		current = nil
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if inFence {
			current = append(current, line)
			if strings.HasPrefix(trimmed, fenceMarker) {
				flush("code", fenceLanguage)
				inFence = false
				fenceMarker = ""
				fenceLanguage = ""
			}
			continue
		}
		if marker, language, ok := markdownFence(trimmed); ok {
			flush("text", "")
			inFence = true
			fenceMarker = marker
			fenceLanguage = language
			current = []string{line}
			continue
		}
		if level, heading, ok := markdownHeading(trimmed); ok {
			flush("text", "")
			if level <= len(headingPath) {
				headingPath = headingPath[:level-1]
			}
			for len(headingPath) < level-1 {
				headingPath = append(headingPath, "")
			}
			headingPath = append(headingPath, heading)
			continue
		}
		if trimmed == "" {
			flush("text", "")
			continue
		}
		current = append(current, line)
	}
	if inFence {
		flush("code", fenceLanguage)
	} else {
		flush("text", "")
	}
	return blocks
}

func ParsePlainBlocks(text string) []SourceBlock {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	var current []string
	blocks := make([]SourceBlock, 0)
	flush := func() {
		content := strings.TrimSpace(strings.Join(current, "\n"))
		if content != "" {
			blocks = append(blocks, SourceBlock{Text: content, Metadata: map[string]any{"block_type": "text"}})
		}
		current = nil
	}
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			flush()
			continue
		}
		current = append(current, line)
	}
	flush()
	return blocks
}

func ChunkBlocks(blocks []SourceBlock, size, overlap int) []ChunkPart {
	parts := make([]ChunkPart, 0)
	for start := 0; start < len(blocks); {
		end := start + 1
		group := []string{blocks[start].Text}
		for end < len(blocks) && reflect.DeepEqual(blocks[start].Metadata, blocks[end].Metadata) {
			group = append(group, blocks[end].Text)
			end++
		}

		for _, text := range chunkText(strings.Join(group, "\n\n"), size, overlap) {
			metadata := cloneMetadata(blocks[start].Metadata)
			metadata["block_index"] = start
			if end-start > 1 {
				metadata["block_end_index"] = end - 1
			}
			metadata["chunk_index"] = len(parts)
			parts = append(parts, ChunkPart{Text: addHeadingContext(text, metadata), Metadata: metadata})
		}
		start = end
	}
	return parts
}

func addHeadingContext(text string, metadata map[string]any) string {
	headingPath, ok := metadata["heading_path"].([]string)
	if !ok || len(headingPath) == 0 {
		return text
	}
	return strings.Join(headingPath, " > ") + "\n\n" + text
}

func chunkText(text string, size, overlap int) []string {
	text = strings.TrimSpace(strings.ReplaceAll(text, "\r\n", "\n"))
	if text == "" || size <= 0 {
		return nil
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= size {
		overlap = size / 5
	}
	runes := []rune(text)
	chunks := make([]string, 0, (len(runes)/size)+1)
	for start := 0; start < len(runes); {
		end := start + size
		if end > len(runes) {
			end = len(runes)
		}
		if end < len(runes) {
			if boundary := findBoundary(runes, start, end); boundary > start+size/2 {
				end = boundary
			}
		}
		chunk := strings.TrimSpace(string(runes[start:end]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		if end == len(runes) {
			break
		}
		next := end - overlap
		if next <= start {
			next = end
		}
		start = next
	}
	return chunks
}

func markdownHeading(line string) (int, string, bool) {
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level > 6 || len(line) <= level || line[level] != ' ' {
		return 0, "", false
	}
	heading := strings.TrimSpace(line[level:])
	return level, heading, heading != ""
}

func markdownFence(line string) (string, string, bool) {
	if strings.HasPrefix(line, "```") {
		return "```", strings.TrimSpace(strings.TrimPrefix(line, "```")), true
	}
	if strings.HasPrefix(line, "~~~") {
		return "~~~", strings.TrimSpace(strings.TrimPrefix(line, "~~~")), true
	}
	return "", "", false
}

func cloneMetadata(source map[string]any) map[string]any {
	result := make(map[string]any, len(source)+2)
	for key, value := range source {
		if path, ok := value.([]string); ok {
			result[key] = append([]string(nil), path...)
			continue
		}
		result[key] = value
	}
	return result
}

func findBoundary(runes []rune, start, end int) int {
	for i := end; i > start+len(runes)/100 && i > start; i-- {
		if runes[i-1] == '\n' || runes[i-1] == '。' || runes[i-1] == '！' || runes[i-1] == '？' || runes[i-1] == '.' || runes[i-1] == '!' || runes[i-1] == '?' {
			return i
		}
	}
	return end
}

func approximateTokens(text string) int {
	return (utf8.RuneCountInString(text) + 3) / 4
}
