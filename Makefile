GO  ?= go
BIN := gitproxy

.PHONY: all build test vet cross ci clean

all: build

build:
	$(GO) build -o $(BIN) .

test:
	$(GO) test -v .

vet:
	$(GO) vet .

cross:
	GOOS=linux   GOARCH=amd64 $(GO) build -o /dev/null .
	GOOS=windows GOARCH=amd64 $(GO) build -o /dev/null .
	GOOS=darwin  GOARCH=arm64 $(GO) build -o /dev/null .

ci: vet test cross build

clean:
	rm -f $(BIN)
