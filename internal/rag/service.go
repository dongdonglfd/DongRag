package rag

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/lfd/minirag/internal/observability"
	"github.com/lfd/minirag/internal/store"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type Embedder interface {
	Embed(context.Context, []string) ([][]float32, error)
}

type Chatger interface {
	Complete(context.Context, string, string) (string, error)
}

type StreamingChatger interface {
	Stream(context.Context, string, string, func(string) error) error
}

type RerankResult struct {
	Index int
	Score float64
}
type Reranker interface {
	Rerank(context.Context, string, []string) ([]RerankResult, error)
}

type Service struct {
	Store           *store.Store
	Embedder        Embedder
	Chatger         Chatger
	ChunkSize       int
	ChunkOverlap    int
	EmbeddingDim    int
	CandidateK      int
	Reranker        Reranker
	RerankMaxPerDoc int
	RerankTimeout   time.Duration
	Metrics         *observability.Metrics
}

type QueryResult struct {
	Answer         string      `json:"answer"`
	Citations      []store.Hit `json:"citations"`
	RetrievalMS    int64       `json:"retrieval_ms"`
	TotalMS        int64       `json:"total_ms"`
	RetrievalMode  string      `json:"retrieval_mode"`
	CandidateK     int         `json:"candidate_k"`
	Reranked       bool        `json:"reranked"`
	RerankFallback bool        `json:"rerank_fallback"`
}

type queryPreparation struct {
	hits           []store.Hit
	systemPrompt   string
	userPrompt     string
	retrievalMS    int64
	candidateK     int
	reranked       bool
	rerankFallback bool
}

func (s *Service) IndexDocument(ctx context.Context, name, contentType string, data []byte) (store.Document, error) {
	ctx, span := observability.Start(ctx, "Document Ingestion")
	var resultErr error
	defer func() { observability.End(span, resultErr) }()
	checksumBytes := sha256.Sum256(data)
	checksum := hex.EncodeToString(checksumBytes[:])
	if existing, err := s.Store.FindDocumentByChecksum(ctx, checksum); err != nil {
		resultErr = err
		return store.Document{}, err
	} else if existing != nil {
		return *existing, nil
	}
	document := store.Document{
		ID:          newID("doc"),
		Name:        filepath.Base(name),
		ContentType: contentType,
		SizeBytes:   int64(len(data)),
		Checksum:    checksum,
		CreatedAt:   time.Now().UTC(),
	}
	chunks, err := s.buildChunks(ctx, document.ID, document.Name, data)
	if err != nil {
		resultErr = err
		return store.Document{}, err
	}
	if err := s.Store.CreateDocument(ctx, document, chunks); err != nil {
		resultErr = err
		return store.Document{}, err
	}
	return document, nil
}

func (s *Service) ReindexDocument(ctx context.Context, documentID, name string, data []byte) error {
	ctx, span := observability.Start(ctx, "Document Reindex")
	var resultErr error
	defer func() { observability.End(span, resultErr) }()
	if strings.TrimSpace(documentID) == "" {
		resultErr = fmt.Errorf("document id is required for reindex")
		return resultErr
	}
	chunks, err := s.buildChunks(ctx, documentID, filepath.Base(name), data)
	if err != nil {
		resultErr = err
		return err
	}
	resultErr = s.Store.ReplaceDocumentChunks(ctx, documentID, chunks)
	return resultErr
}

func (s *Service) ReindexStoredChunks(ctx context.Context, documentID string) error {
	ctx, span := observability.Start(ctx, "Document Reindex")
	var resultErr error
	defer func() { observability.End(span, resultErr) }()
	chunks, err := s.Store.ListDocumentChunks(ctx, documentID)
	if err != nil {
		resultErr = err
		return err
	}
	rebuilt, err := s.rebuildStoredChunks(ctx, documentID, chunks)
	if err != nil {
		resultErr = err
		return err
	}
	resultErr = s.Store.ReplaceDocumentChunks(ctx, documentID, rebuilt)
	return resultErr
}

func (s *Service) rebuildStoredChunks(ctx context.Context, documentID string, current []store.Chunk) ([]store.Chunk, error) {
	if len(current) == 0 {
		return nil, fmt.Errorf("document contains no stored chunks")
	}
	texts := make([]string, len(current))
	for i, chunk := range current {
		texts[i] = chunk.Content
	}
	embeddings, err := s.embedTexts(ctx, texts)
	if err != nil {
		return nil, err
	}
	rebuilt := make([]store.Chunk, 0, len(current))
	for i, chunk := range current {
		metadata := cloneMetadata(chunk.Metadata)
		metadata["reindex_source"] = "stored_chunks"
		metadata["approx_tokens"] = approximateTokens(chunk.Content)
		rebuilt = append(rebuilt, store.Chunk{
			ID:         newID("chunk"),
			DocumentID: documentID,
			Ordinal:    chunk.Ordinal,
			Content:    chunk.Content,
			Metadata:   metadata,
			Embedding:  embeddings[i],
		})
	}
	return rebuilt, nil
}

func (s *Service) buildChunks(ctx context.Context, documentID, name string, data []byte) ([]store.Chunk, error) {
	blocks, err := extractBlocks(ctx, name, data)
	if err != nil {
		return nil, err
	}
	parts := ChunkBlocks(blocks, s.ChunkSize, s.ChunkOverlap)
	if len(parts) == 0 {
		return nil, fmt.Errorf("document contains no readable text")
	}
	texts := make([]string, len(parts))
	for i, part := range parts {
		texts[i] = part.Text
	}
	embeddings, err := s.embedTexts(ctx, texts)
	if err != nil {
		return nil, err
	}
	chunks := make([]store.Chunk, 0, len(parts))
	for i, part := range parts {
		metadata := cloneMetadata(part.Metadata)
		metadata["source"] = filepath.Base(name)
		metadata["approx_tokens"] = approximateTokens(part.Text)
		chunks = append(chunks, store.Chunk{
			ID:         newID("chunk"),
			DocumentID: documentID,
			Ordinal:    i,
			Content:    part.Text,
			Metadata:   metadata,
			Embedding:  embeddings[i],
		})
	}
	return chunks, nil
}

func (s *Service) embedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	ctx, span := observability.Start(ctx, "Embedding")
	started := time.Now()
	embeddings, err := s.Embedder.Embed(ctx, texts)
	observability.End(span, err)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("ingestion", "embedding", started)
	}
	if err != nil {
		return nil, err
	}
	if len(embeddings) != len(texts) {
		return nil, fmt.Errorf("embedding count mismatch: got %d, want %d", len(embeddings), len(texts))
	}
	for _, embedding := range embeddings {
		if len(embedding) != s.EmbeddingDim {
			return nil, fmt.Errorf("embedding dimension is %d, expected %d; update EMBEDDING_DIM and database schema if the model differs", len(embedding), s.EmbeddingDim)
		}
	}
	return embeddings, nil
}

func (s *Service) Query(ctx context.Context, question string, topK int) (result QueryResult, resultErr error) {
	started := time.Now()
	ctx, querySpan := observability.Start(ctx, "Query Pipeline")
	defer func() {
		observability.End(querySpan, resultErr)
		if s.Metrics != nil {
			s.Metrics.QueryDuration.Observe(time.Since(started).Seconds())
		}
	}()
	prep, err := s.prepareQuery(ctx, question, topK)
	if err != nil {
		return QueryResult{}, err
	}
	llmCtx, llmSpan := observability.Start(ctx, "LLM")
	llmStarted := time.Now()
	answer, err := s.Chatger.Complete(llmCtx, prep.systemPrompt, prep.userPrompt)
	observability.End(llmSpan, err)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "llm", llmStarted)
	}
	if err != nil {
		return QueryResult{}, err
	}
	_, responseSpan := observability.Start(ctx, "Response")
	result = prep.result(answer, time.Since(started).Milliseconds())
	observability.End(responseSpan, nil)
	return result, nil
}

func (s *Service) QueryStream(ctx context.Context, question string, topK int, onToken func(string) error) (result QueryResult, resultErr error) {
	started := time.Now()
	ctx, querySpan := observability.Start(ctx, "Query Pipeline")
	querySpan.SetAttributes(attribute.Bool("query.streaming", true))
	defer func() {
		querySpan.SetAttributes(attribute.Int64("query.total_duration_ms", time.Since(started).Milliseconds()))
		observability.End(querySpan, resultErr)
		if s.Metrics != nil {
			s.Metrics.QueryDuration.Observe(time.Since(started).Seconds())
		}
	}()
	prep, err := s.prepareQuery(ctx, question, topK)
	if err != nil {
		return QueryResult{}, err
	}
	streamer, ok := s.Chatger.(StreamingChatger)
	if !ok {
		return QueryResult{}, fmt.Errorf("chat provider does not support streaming")
	}
	streamCtx, streamingSpan := observability.Start(ctx, "Streaming")
	streamStarted := time.Now()
	firstToken := false
	var firstTokenAt time.Time
	var answer strings.Builder
	err = streamer.Stream(streamCtx, prep.systemPrompt, prep.userPrompt, func(token string) error {
		if !firstToken {
			firstToken = true
			firstTokenAt = time.Now()
			_, firstSpan := observability.Start(streamCtx, "First Token", trace.WithTimestamp(firstTokenAt))
			observability.End(firstSpan, nil)
			streamingSpan.SetAttributes(attribute.Int64("stream.first_token_ms", firstTokenAt.Sub(started).Milliseconds()))
			if s.Metrics != nil {
				s.Metrics.ObserveStage("query", "first_token", started)
			}
		}
		answer.WriteString(token)
		if onToken == nil {
			return nil
		}
		return onToken(token)
	})
	streamingSpan.SetAttributes(
		attribute.Int64("stream.generation_duration_ms", time.Since(streamStarted).Milliseconds()),
		attribute.Int64("stream.total_duration_ms", time.Since(started).Milliseconds()),
	)
	if firstToken {
		streamingSpan.SetAttributes(attribute.Bool("stream.first_token_received", true))
	}
	observability.End(streamingSpan, err)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "streaming", streamStarted)
	}
	if err != nil {
		return QueryResult{}, err
	}
	_, responseSpan := observability.Start(ctx, "Response")
	result = prep.result(answer.String(), time.Since(started).Milliseconds())
	observability.End(responseSpan, nil)
	return result, nil
}

func (s *Service) prepareQuery(ctx context.Context, question string, topK int) (queryPreparation, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return queryPreparation{}, fmt.Errorf("question is required")
	}
	embedCtx, embedSpan := observability.Start(ctx, "Embedding")
	embedStarted := time.Now()
	embeddings, err := s.Embedder.Embed(embedCtx, []string{question})
	observability.End(embedSpan, err)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "embedding", embedStarted)
	}
	if err != nil {
		return queryPreparation{}, err
	}
	if len(embeddings) != 1 || len(embeddings[0]) != s.EmbeddingDim {
		return queryPreparation{}, fmt.Errorf("query embedding dimension mismatch")
	}
	retrievalStarted := time.Now()
	candidateK := s.CandidateK
	if candidateK < topK {
		candidateK = topK
	}
	vectorCtx, vectorSpan := observability.Start(ctx, "Vector Search")
	vectorStarted := time.Now()
	vectorHits, err := s.Store.Search(vectorCtx, question, embeddings[0], store.SearchModeVector, candidateK, candidateK)
	observability.End(vectorSpan, err)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "vector_search", vectorStarted)
	}
	if err != nil {
		return queryPreparation{}, err
	}
	lexicalCtx, lexicalSpan := observability.Start(ctx, "Lexical Search")
	lexicalStarted := time.Now()
	lexicalHits, err := s.Store.Search(lexicalCtx, question, embeddings[0], store.SearchModeLexical, candidateK, candidateK)
	observability.End(lexicalSpan, err)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "lexical_search", lexicalStarted)
	}
	if err != nil {
		return queryPreparation{}, err
	}
	_, rrfSpan := observability.Start(ctx, "RRF")
	rrfStarted := time.Now()
	hits := fuseRRF(vectorHits, lexicalHits, candidateK)
	observability.End(rrfSpan, nil)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "rrf", rrfStarted)
	}
	reranked := false
	fallback := false
	rerankCtx, rerankSpan := observability.Start(ctx, "Rerank")
	rerankStarted := time.Now()
	if s.Reranker != nil {
		callCtx := rerankCtx
		if s.RerankTimeout > 0 {
			var cancel context.CancelFunc
			callCtx, cancel = context.WithTimeout(rerankCtx, s.RerankTimeout)
			defer cancel()
		}
		selected, rerankErr := rerankHits(callCtx, s.Reranker, question, hits, topK, s.RerankMaxPerDoc)
		if rerankErr == nil {
			hits = selected
			reranked = true
		} else {
			rerankSpan.RecordError(rerankErr)
			fallback = true
			hits = selectDiverse(hits, topK, s.RerankMaxPerDoc)
		}
	} else {
		hits = selectDiverse(hits, topK, s.RerankMaxPerDoc)
	}
	observability.End(rerankSpan, nil)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "rerank", rerankStarted)
	}
	retrievalMS := time.Since(retrievalStarted).Milliseconds()
	_, contextSpan := observability.Start(ctx, "Context Build")
	contextStarted := time.Now()
	hits, contextText := PrepareGenerationContext(hits, 6000, 2)
	observability.End(contextSpan, nil)
	if s.Metrics != nil {
		s.Metrics.ObserveStage("query", "context_build", contextStarted)
	}
	systemPrompt, prompt := GenerationPrompts(question, contextText)
	return queryPreparation{hits: hits, systemPrompt: systemPrompt, userPrompt: prompt, retrievalMS: retrievalMS, candidateK: candidateK, reranked: reranked, rerankFallback: fallback}, nil
}

func (p queryPreparation) result(answer string, totalMS int64) QueryResult {
	return QueryResult{Answer: answer, Citations: p.hits, RetrievalMS: p.retrievalMS, TotalMS: totalMS, RetrievalMode: string(store.SearchModeHybrid), CandidateK: p.candidateK, Reranked: p.reranked, RerankFallback: p.rerankFallback}
}

func fuseRRF(vectorHits, lexicalHits []store.Hit, topK int) []store.Hit {
	combined := make(map[string]store.Hit, len(vectorHits)+len(lexicalHits))
	for i, hit := range vectorHits {
		hit.VectorRank = i + 1
		hit.RRFScore = 1.0 / float64(61+i)
		combined[hit.ID] = hit
	}
	for i, hit := range lexicalHits {
		if existing, ok := combined[hit.ID]; ok {
			existing.LexicalRank = i + 1
			existing.LexicalScore = hit.LexicalScore
			existing.RRFScore += 1.0 / float64(61+i)
			combined[hit.ID] = existing
			continue
		}
		hit.LexicalRank = i + 1
		hit.RRFScore = 1.0 / float64(61+i)
		combined[hit.ID] = hit
	}
	result := make([]store.Hit, 0, len(combined))
	for _, hit := range combined {
		hit.Score = hit.RRFScore
		result = append(result, hit)
	}
	sort.SliceStable(result, func(i, j int) bool {
		if result[i].RRFScore != result[j].RRFScore {
			return result[i].RRFScore > result[j].RRFScore
		}
		left := math.Max(result[i].VectorScore, result[i].LexicalScore)
		right := math.Max(result[j].VectorScore, result[j].LexicalScore)
		if left != right {
			return left > right
		}
		return result[i].ID < result[j].ID
	})
	if topK < len(result) {
		result = result[:topK]
	}
	return result
}

func rerankHits(ctx context.Context, reranker Reranker, query string, candidates []store.Hit, topK, maxPerDoc int) ([]store.Hit, error) {
	texts := make([]string, len(candidates))
	for i := range candidates {
		texts[i] = candidates[i].Content
	}
	results, err := reranker.Rerank(ctx, query, texts)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("reranker returned no results")
	}
	ordered := make([]store.Hit, 0, len(results))
	seen := make(map[int]struct{}, len(results))
	for rank, result := range results {
		if result.Index < 0 || result.Index >= len(candidates) {
			return nil, fmt.Errorf("invalid reranker index %d", result.Index)
		}
		if _, ok := seen[result.Index]; ok {
			continue
		}
		seen[result.Index] = struct{}{}
		hit := candidates[result.Index]
		hit.RerankRank = rank + 1
		hit.RerankScore = result.Score
		ordered = append(ordered, hit)
	}
	selected := selectDiverse(ordered, topK, maxPerDoc)
	for i := range selected {
		selected[i].FinalRank = i + 1
	}
	return selected, nil
}

func RerankHits(ctx context.Context, reranker Reranker, query string, candidates []store.Hit, topK, maxPerDoc int) ([]store.Hit, error) {
	return rerankHits(ctx, reranker, query, candidates, topK, maxPerDoc)
}

func selectDiverse(hits []store.Hit, topK, maxPerDoc int) []store.Hit {
	if topK < 1 {
		return nil
	}
	if maxPerDoc < 1 {
		maxPerDoc = 2
	}
	selected := make([]store.Hit, 0, topK)
	counts := make(map[string]int)
	for _, hit := range hits {
		if len(selected) >= topK {
			break
		}
		if counts[hit.DocumentID] >= maxPerDoc {
			continue
		}
		counts[hit.DocumentID]++
		selected = append(selected, hit)
	}
	return selected
}

func PrepareGenerationContext(hits []store.Hit, maxChars, maxPerDocument int) ([]store.Hit, string) {
	ordered := append([]store.Hit(nil), hits...)
	sort.SliceStable(ordered, func(i, j int) bool { return generationScore(ordered[i]) > generationScore(ordered[j]) })
	if maxPerDocument < 1 {
		maxPerDocument = 2
	}
	seenContent := make(map[string]struct{}, len(ordered))
	documentCounts := make(map[string]int)
	selected := make([]store.Hit, 0, len(ordered))
	var builder strings.Builder
	for _, hit := range ordered {
		contentKey := strings.Join(strings.Fields(strings.ToLower(hit.Content)), " ")
		if contentKey == "" {
			continue
		}
		if _, duplicate := seenContent[contentKey]; duplicate {
			continue
		}
		if documentCounts[hit.DocumentID] >= maxPerDocument {
			continue
		}
		block := contextBlock(len(selected)+1, hit)
		if builder.Len()+len(block) > maxChars {
			continue
		}
		seenContent[contentKey] = struct{}{}
		documentCounts[hit.DocumentID]++
		selected = append(selected, hit)
		builder.WriteString(block)
	}
	if builder.Len() == 0 {
		return nil, "（没有可用参考资料）"
	}
	return selected, builder.String()
}

func GenerationPrompts(question, contextText string) (string, string) {
	system := "你是严谨的知识库问答助手。只能使用给定 Context 中明确陈述的信息，不得使用外部知识或自行推断。"
	user := fmt.Sprintf(`请根据 Context 直接回答 Question。

规则：
1. 先完整覆盖问题要求的关键点，再删除重复解释和无关背景；不要复述问题。
2. 每个事实必须能由 Context 支持，并在对应句末标注来源编号，如 [1]；编号只能使用 Context 中已有编号。
3. 不得加入 Context 未陈述的原因、结论、例子或常识。
4. 如果 Context 没有足够信息回答，必须只回答“资料不足，我不知道。”，且不要添加引用。

Context:
%s
Question: %s
Answer:`, contextText, question)
	return system, user
}

func generationScore(hit store.Hit) float64 {
	if hit.RerankRank > 0 {
		return hit.RerankScore
	}
	if hit.RRFScore != 0 {
		return hit.RRFScore
	}
	return hit.Score
}

func contextBlock(index int, hit store.Hit) string {
	metadata := []string{"文档：" + hit.DocumentName}
	if heading := metadataHeading(hit.Metadata["heading_path"]); heading != "" {
		metadata = append(metadata, "章节："+heading)
	}
	if page, ok := hit.Metadata["page"]; ok {
		metadata = append(metadata, fmt.Sprintf("页码：%v", page))
	}
	return fmt.Sprintf("[%d] %s\n正文：%s\n\n", index, strings.Join(metadata, " | "), strings.TrimSpace(hit.Content))
}

func metadataHeading(value any) string {
	switch path := value.(type) {
	case []string:
		return strings.Join(path, " > ")
	case []any:
		parts := make([]string, 0, len(path))
		for _, item := range path {
			if text, ok := item.(string); ok && text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, " > ")
	default:
		return ""
	}
}

func extractBlocks(ctx context.Context, name string, data []byte) ([]SourceBlock, error) {
	ext := strings.ToLower(filepath.Ext(name))
	if ext != ".pdf" {
		if ext == ".md" || ext == ".markdown" {
			return ParseMarkdownBlocks(string(data)), nil
		}
		return ParsePlainBlocks(string(data)), nil
	}
	path, err := os.CreateTemp("", "minirag-*.pdf")
	if err != nil {
		return nil, err
	}
	pathName := path.Name()
	defer os.Remove(pathName)
	if _, err := path.Write(data); err != nil {
		path.Close()
		return nil, err
	}
	if err := path.Close(); err != nil {
		return nil, err
	}
	command := exec.CommandContext(ctx, "pdftotext", "-layout", pathName, "-")
	output, err := command.Output()
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("PDF 解析需要安装 pdftotext（Ubuntu: sudo apt install poppler-utils）: %w", err)
	}
	pages := strings.Split(string(output), "\f")
	blocks := make([]SourceBlock, 0)
	for pageIndex, page := range pages {
		for _, block := range ParsePlainBlocks(page) {
			block.Metadata["page"] = pageIndex + 1
			blocks = append(blocks, block)
		}
	}
	return blocks, nil
}

func newID(prefix string) string {
	var random [12]byte
	if _, err := rand.Read(random[:]); err != nil {
		return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
	}
	return fmt.Sprintf("%s-%x", prefix, random)
}
