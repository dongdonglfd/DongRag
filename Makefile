.PHONY: run fmt test build seed eval eval-baseline

run:
	go run ./cmd/minirag

fmt:
	gofmt -w cmd internal

test:
	go test ./...

build:
	go build -o bin/minirag ./cmd/minirag

seed:
	@for file in seeddata/*.md; do curl -fsS -F "file=@$$file" http://localhost:8080/v1/documents; echo; done

eval:
	go run ./cmd/minirag eval --dataset testdata/eval.json --top-k 5

eval-baseline:
	go run ./cmd/minirag eval --dataset testdata/eval.json --top-k 5 --candidate-k 50 --baseline artifacts/retrieval-baseline.json --update-baseline
