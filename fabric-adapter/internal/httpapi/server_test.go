package httpapi

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"fabric-fl/fabric-adapter/internal/traffic"
)

type fakeClient struct {
	method      string
	transaction string
	args        []string
	result      []byte
	err         error
	calls       int
	traffic     traffic.Snapshot
}

func (f *fakeClient) TrafficSnapshot() traffic.Snapshot {
	return f.traffic
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
	f.calls++
	f.method = method
	f.transaction = transaction
	f.args = args
}

func TestPrototypeBatchCollectsClientsAndSubmitsOneFabricTransaction(t *testing.T) {
	client := &fakeClient{result: processRoundResult()}
	handler := New(client)

	opened := request(
		handler,
		http.MethodPost,
		"/prototype-batches/open",
		openBatchBody(42, 2),
	)
	if opened.Code != http.StatusOK {
		t.Fatalf("open status = %d, body = %s", opened.Code, opened.Body.String())
	}

	first := request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(42, 0, 100))
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d, body = %s", first.Code, first.Body.String())
	}
	if client.calls != 0 {
		t.Fatalf("Fabric calls after first client = %d, want 0", client.calls)
	}

	second := request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(42, 1, 200))
	if second.Code != http.StatusOK {
		t.Fatalf("second status = %d, body = %s", second.Code, second.Body.String())
	}
	if client.calls != 1 || client.method != "submit" {
		t.Fatalf("Fabric calls = %d, method = %q", client.calls, client.method)
	}
	if client.transaction != "ProcessRound" || len(client.args) != 8 {
		t.Fatalf("Fabric transaction = %q, args = %#v", client.transaction, client.args)
	}
	if client.args[0] != "42" {
		t.Fatalf("round arg = %q", client.args[0])
	}
	var payloads []prototypeSubmission
	if err := json.Unmarshal([]byte(client.args[7]), &payloads); err != nil {
		t.Fatalf("decode Fabric batch: %v", err)
	}
	if len(payloads) != 2 || payloads[0].ClientID != 0 || payloads[1].ClientID != 1 {
		t.Fatalf("Fabric payloads = %#v", payloads)
	}
	var response struct {
		Result prototypeBatchStatusResponse `json:"result"`
	}
	if err := json.Unmarshal(second.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Result.Status != "PROCESSED" || len(response.Result.RoundResult) == 0 {
		t.Fatalf("second result = %#v", response.Result)
	}
}

func TestPrototypeBatchRejectsConflictingDuplicate(t *testing.T) {
	handler := New(&fakeClient{})
	request(
		handler,
		http.MethodPost,
		"/prototype-batches/open",
		openBatchBody(7, 2),
	)
	request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(7, 0, 100))
	response := request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(7, 0, 999))
	if response.Code != http.StatusConflict {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusConflict)
	}
}

func TestPrototypeBatchRejectsTamperedClientSignature(t *testing.T) {
	handler := New(&fakeClient{})
	request(handler, http.MethodPost, "/prototype-batches/open", openBatchBody(17, 1))
	body := prototypeBody(17, 0, 100)
	var payload prototypeSubmission
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	payload.Values[0] = 999
	tampered, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	response := request(handler, http.MethodPost, "/prototype-batches/submit", tampered)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d; body=%s", response.Code, http.StatusUnauthorized, response.Body)
	}
}

func TestPrototypeBatchRetriesAfterFabricSubmissionFailure(t *testing.T) {
	client := &fakeClient{err: errors.New("commit timeout"), result: processRoundResult()}
	handler := New(client)
	request(
		handler,
		http.MethodPost,
		"/prototype-batches/open",
		openBatchBody(8, 2),
	)
	request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(8, 0, 100))
	failed := request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(8, 1, 200))
	if failed.Code != http.StatusBadGateway {
		t.Fatalf("failed status = %d, want %d", failed.Code, http.StatusBadGateway)
	}

	client.err = nil
	retried := request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(8, 1, 200))
	if retried.Code != http.StatusOK {
		t.Fatalf("retry status = %d, body = %s", retried.Code, retried.Body.String())
	}
	if client.calls != 2 {
		t.Fatalf("Fabric calls = %d, want 2", client.calls)
	}
}

func TestPrototypeBatchStatus(t *testing.T) {
	handler := New(&fakeClient{})
	request(
		handler,
		http.MethodPost,
		"/prototype-batches/open",
		openBatchBody(9, 2),
	)
	request(handler, http.MethodPost, "/prototype-batches/submit", prototypeBody(9, 0, 100))
	response := request(handler, http.MethodGet, "/prototype-batches/status?round_id=9", nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := response.Body.String(); got != "{\"result\":{\"round_id\":9,\"expected_clients\":2,\"received_clients\":1,\"status\":\"COLLECTING\"}}\n" {
		t.Fatalf("body = %q", got)
	}
}

func openBatchBody(roundID int, expectedClients int) []byte {
	publicKeys := make([]string, expectedClients)
	for clientID := range publicKeys {
		publicKeys[clientID] = testClientPublicKey(clientID)
	}
	body, err := json.Marshal(openPrototypeBatchRequest{
		RoundID: roundID, ExperimentID: 1000, Sequence: roundID,
		ExpectedClients: expectedClients, NumClasses: 1, Dimension: 1, Scale: 100,
		ClientPublicKeys: publicKeys,
	})
	if err != nil {
		panic(err)
	}
	return body
}

func processRoundResult() []byte {
	return []byte(`{"round_id":42,"status":"FINALIZED"}`)
}

func prototypeBody(roundID int, clientID int, value int64) []byte {
	submission := prototypeSubmission{
		Encoding: prototypeEncoding,
		RoundID:  roundID,
		ClientID: clientID,
		Shape:    []int{1, 1},
		Scale:    100,
		Values:   []int64{value},
		Counts:   []int64{1},
	}
	privateKey := testClientPrivateKey(clientID)
	submission.PublicKey = base64.StdEncoding.EncodeToString(privateKey.Public().(ed25519.PublicKey))
	unsigned := unsignedPrototypeSubmission{
		Encoding: submission.Encoding, RoundID: submission.RoundID, ClientID: submission.ClientID,
		Shape: submission.Shape, Scale: submission.Scale, Values: submission.Values, Counts: submission.Counts,
	}
	message, err := json.Marshal(unsigned)
	if err != nil {
		panic(err)
	}
	submission.Signature = base64.StdEncoding.EncodeToString(
		ed25519.Sign(privateKey, append([]byte(prototypeSignatureDomain), message...)),
	)
	payload, err := json.Marshal(submission)
	if err != nil {
		panic(err)
	}
	return payload
}

func testClientPrivateKey(clientID int) ed25519.PrivateKey {
	seed := bytes.Repeat([]byte{byte(clientID + 1)}, ed25519.SeedSize)
	return ed25519.NewKeyFromSeed(seed)
}

func testClientPublicKey(clientID int) string {
	privateKey := testClientPrivateKey(clientID)
	return base64.StdEncoding.EncodeToString(privateKey.Public().(ed25519.PublicKey))
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

func TestTrafficReportsHTTPAndGRPCBytesWithoutCountingSnapshot(t *testing.T) {
	client := &fakeClient{
		result:  []byte(`{"ok":true}`),
		traffic: traffic.Snapshot{RXBytes: 300, TXBytes: 700},
	}
	handler := New(client)
	body := []byte(`{"transaction":"Get","args":[]}`)
	response := request(handler, http.MethodPost, "/evaluate", body)
	if response.Code != http.StatusOK {
		t.Fatalf("evaluate status = %d", response.Code)
	}

	first := request(handler, http.MethodGet, "/traffic", nil)
	second := request(handler, http.MethodGet, "/traffic", nil)
	if first.Body.String() != second.Body.String() {
		t.Fatalf("traffic endpoint counted itself: first=%s second=%s", first.Body, second.Body)
	}
	var decoded struct {
		Result map[string]uint64 `json:"result"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("decode traffic response: %v", err)
	}
	if decoded.Result["http_rx_bytes"] != uint64(len(body)) {
		t.Fatalf("http rx = %d, want %d", decoded.Result["http_rx_bytes"], len(body))
	}
	if decoded.Result["http_tx_bytes"] != uint64(response.Body.Len()) {
		t.Fatalf("http tx = %d, want %d", decoded.Result["http_tx_bytes"], response.Body.Len())
	}
	if decoded.Result["grpc_rx_bytes"] != 300 || decoded.Result["grpc_tx_bytes"] != 700 {
		t.Fatalf("grpc traffic = %#v", decoded.Result)
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
