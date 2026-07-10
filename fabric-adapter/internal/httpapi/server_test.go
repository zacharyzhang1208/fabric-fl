package httpapi

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fakeClient struct {
	method      string
	transaction string
	args        []string
	result      []byte
	err         error
}

func (f *fakeClient) Evaluate(transaction string, args ...string) ([]byte, error) {
	f.record("evaluate", transaction, args)
	return f.result, f.err
}

func (f *fakeClient) Submit(transaction string, args ...string) ([]byte, error) {
	f.record("submit", transaction, args)
	return f.result, f.err
}

func (f *fakeClient) record(method string, transaction string, args []string) {
	f.method = method
	f.transaction = transaction
	f.args = args
}

func TestHealth(t *testing.T) {
	response := request(New(&fakeClient{}), http.MethodGet, "/healthz", nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if got := response.Body.String(); got != "{\"status\":\"ok\"}\n" {
		t.Fatalf("body = %q", got)
	}
}

func TestEvaluate(t *testing.T) {
	client := &fakeClient{result: []byte(`{"accuracy":0.91}`)}
	body := []byte(`{"transaction":"Get","args":["round:1"]}`)
	response := request(New(client), http.MethodPost, "/evaluate", body)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if client.method != "evaluate" || client.transaction != "Get" {
		t.Fatalf("call = %s %s", client.method, client.transaction)
	}
	if len(client.args) != 1 || client.args[0] != "round:1" {
		t.Fatalf("args = %#v", client.args)
	}
	if got := response.Body.String(); got != "{\"result\":{\"accuracy\":0.91}}\n" {
		t.Fatalf("body = %q", got)
	}
}

func TestSubmit(t *testing.T) {
	client := &fakeClient{}
	body := []byte(`{"transaction":"Set","args":["hello","world"]}`)
	response := request(New(client), http.MethodPost, "/submit", body)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if client.method != "submit" || client.transaction != "Set" {
		t.Fatalf("call = %s %s", client.method, client.transaction)
	}
	if got := response.Body.String(); got != "{\"result\":null}\n" {
		t.Fatalf("body = %q", got)
	}
}

func TestInvalidRequest(t *testing.T) {
	response := request(New(&fakeClient{}), http.MethodPost, "/evaluate", []byte(`{"args":[]}`))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusBadRequest)
	}
}

func TestFabricError(t *testing.T) {
	client := &fakeClient{err: errors.New("endorsement failed")}
	body := []byte(`{"transaction":"Set","args":[]}`)
	response := request(New(client), http.MethodPost, "/submit", body)
	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusBadGateway)
	}
}

func TestMethodNotAllowed(t *testing.T) {
	response := request(New(&fakeClient{}), http.MethodGet, "/submit", nil)
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusMethodNotAllowed)
	}
	if got := response.Header().Get("Allow"); got != http.MethodPost {
		t.Fatalf("Allow = %q", got)
	}
}

func request(handler http.Handler, method string, path string, body []byte) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(method, path, bytes.NewReader(body)))
	return recorder
}
