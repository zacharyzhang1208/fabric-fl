package httpapi

import (
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"time"

	"fabric-fl/fabric-adapter/internal/traffic"
)

const maxRequestBody = 1 << 20

type FabricClient interface {
	Evaluate(transaction string, args ...string) ([]byte, error)
	Submit(transaction string, args ...string) ([]byte, error)
	TrafficSnapshot() traffic.Snapshot
}

type Server struct {
	client  FabricClient
	mux     *http.ServeMux
	batches *prototypeBatchCollector
	traffic *traffic.Counters
}

type transactionRequest struct {
	Transaction string   `json:"transaction"`
	Args        []string `json:"args"`
}

func New(client FabricClient) http.Handler {
	server := &Server{
		client:  client,
		mux:     http.NewServeMux(),
		batches: newPrototypeBatchCollector(),
		traffic: &traffic.Counters{},
	}
	server.mux.HandleFunc("/healthz", server.health)
	server.mux.HandleFunc("/evaluate", server.evaluate)
	server.mux.HandleFunc("/submit", server.submit)
	server.mux.HandleFunc("/prototype-batches/open", server.openPrototypeBatch)
	server.mux.HandleFunc("/prototype-batches/submit", server.submitPrototype)
	server.mux.HandleFunc("/prototype-batches/status", server.prototypeBatchStatus)
	server.mux.HandleFunc("/traffic", server.trafficSnapshot)
	return server
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	if r.URL.Path != "/healthz" && r.URL.Path != "/traffic" {
		r.Body = &countingReadCloser{ReadCloser: r.Body, counters: s.traffic}
		w = &countingResponseWriter{ResponseWriter: w, counters: s.traffic}
	}
	s.mux.ServeHTTP(w, r)
	log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
}

func (s *Server) trafficSnapshot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	httpTraffic := s.traffic.Snapshot()
	grpcTraffic := s.client.TrafficSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{
		"result": map[string]uint64{
			"http_rx_bytes": httpTraffic.RXBytes,
			"http_tx_bytes": httpTraffic.TXBytes,
			"grpc_rx_bytes": grpcTraffic.RXBytes,
			"grpc_tx_bytes": grpcTraffic.TXBytes,
		},
	})
}

type countingReadCloser struct {
	io.ReadCloser
	counters *traffic.Counters
}

func (r *countingReadCloser) Read(buffer []byte) (int, error) {
	count, err := r.ReadCloser.Read(buffer)
	if count > 0 {
		r.counters.AddRX(uint64(count))
	}
	return count, err
}

type countingResponseWriter struct {
	http.ResponseWriter
	counters *traffic.Counters
}

func (w *countingResponseWriter) Write(buffer []byte) (int, error) {
	count, err := w.ResponseWriter.Write(buffer)
	if count > 0 {
		w.counters.AddTX(uint64(count))
	}
	return count, err
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) evaluate(w http.ResponseWriter, r *http.Request) {
	s.handleTransaction(w, r, s.client.Evaluate)
}

func (s *Server) submit(w http.ResponseWriter, r *http.Request) {
	s.handleTransaction(w, r, s.client.Submit)
}

func (s *Server) handleTransaction(
	w http.ResponseWriter,
	r *http.Request,
	invoke func(string, ...string) ([]byte, error),
) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}

	request, err := decodeRequest(w, r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	result, err := invoke(request.Transaction, request.Args...)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"result": responseValue(result)})
}

func decodeRequest(w http.ResponseWriter, r *http.Request) (transactionRequest, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBody)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	var request transactionRequest
	if err := decoder.Decode(&request); err != nil {
		return transactionRequest{}, errors.New("request body must be valid JSON")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return transactionRequest{}, errors.New("request body must contain one JSON object")
	}
	if request.Transaction == "" {
		return transactionRequest{}, errors.New("transaction is required")
	}
	return request, nil
}

func responseValue(result []byte) any {
	if len(result) == 0 {
		return nil
	}

	var value any
	if json.Unmarshal(result, &value) == nil {
		return value
	}
	return string(result)
}

func methodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeError(w, http.StatusMethodNotAllowed, "method not allowed")
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
