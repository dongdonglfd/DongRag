package evaluation

import (
	"testing"

	"github.com/lfd/minirag/internal/store"
)

func TestRuleJudgeScoresCitationsAndEvidence(t *testing.T) {
	item := Case{ReferenceAnswer: "RAG 先检索资料，再生成答案", RequiredCitations: []string{"guide.md"}}
	hits := []store.Hit{{DocumentName: "guide.md", Content: "RAG 先从知识库检索资料，再把资料放入上下文生成答案。"}}
	scores := (RuleJudge{}).Evaluate(item, "RAG 先检索资料，再生成答案。[1]", hits)
	if scores.AnswerCorrectness < .8 || scores.CitationPrecision != 1 || scores.CitationRecall != 1 || scores.Faithfulness != 1 {
		t.Fatalf("unexpected scores: %#v", scores)
	}
}

func TestRuleJudgeDetectsUnsupportedClaimsAndRefusal(t *testing.T) {
	item := Case{ShouldRefuse: true}
	scores := (RuleJudge{}).Evaluate(item, "资料不足，我不知道。", nil)
	if scores.RefusalAccuracy != 1 {
		t.Fatalf("expected correct refusal: %#v", scores)
	}
	if scores.AnswerCorrectness != 1 || scores.CitationPrecision != 1 || scores.CitationRecall != 1 || scores.Faithfulness != 1 {
		t.Fatalf("expected perfect refusal scores: %#v", scores)
	}
	item = Case{ReferenceAnswer: "Go 使用 context 传递取消信号"}
	hits := []store.Hit{{DocumentName: "go.md", Content: "Go 使用 context 传递取消信号。"}}
	scores = (RuleJudge{}).Evaluate(item, "Go 使用 context 传递取消信号。[1] 上海是中国最大的城市。", hits)
	if scores.UnsupportedClaimRate <= 0 || scores.Faithfulness >= 1 {
		t.Fatalf("expected unsupported claim: %#v", scores)
	}
}

func TestValidateRAGDataset(t *testing.T) {
	if err := ValidateRAGDataset(Dataset{Cases: []Case{{ID: "x", ShouldRefuse: true}}}); err != nil {
		t.Fatal(err)
	}
	if err := ValidateRAGDataset(Dataset{Cases: []Case{{ID: "x"}}}); err == nil {
		t.Fatal("expected missing answer error")
	}
}
