package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strconv"
	"sync"
)

const prototypeEncoding = "fixed-point-int64"

type prototypeBatchCollector struct {
	mu     sync.Mutex
	rounds map[int]*prototypeBatchState
}

type prototypeBatchState struct {
	expected   int
	payloads   map[int]prototypeSubmission
	submitting bool
	submitted  bool
}

type openPrototypeBatchRequest struct {
	RoundID         int `json:"round_id"`
	ExpectedClients int `json:"expected_clients"`
}

type prototypeSubmission struct {
	Encoding string  `json:"encoding"`
	RoundID  int     `json:"round_id"`
	ClientID int     `json:"client_id"`
	Shape    []int   `json:"shape"`
	Scale    int64   `json:"scale"`
	Values   []int64 `json:"values"`
	Counts   []int64 `json:"counts"`
}

type prototypeBatchStatusResponse struct {
	RoundID         int    `json:"round_id"`
	ExpectedClients int    `json:"expected_clients"`
	ReceivedClients int    `json:"received_clients"`
	Status          string `json:"status"`
}

func newPrototypeBatchCollector() *prototypeBatchCollector {
	return &prototypeBatchCollector{rounds: make(map[int]*prototypeBatchState)}
}

func (s *Server) openPrototypeBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}

	var request openPrototypeBatchRequest
	if err := decodeJSONRequest(w, r, &request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if request.RoundID < 1 || request.ExpectedClients < 1 {
		writeError(w, http.StatusBadRequest, "round_id and expected_clients must be positive")
		return
	}

	s.batches.mu.Lock()
	state, exists := s.batches.rounds[request.RoundID]
	if exists && state.expected != request.ExpectedClients {
		s.batches.mu.Unlock()
		writeError(w, http.StatusConflict, "prototype batch already exists with different expected_clients")
		return
	}
	if !exists {
		state = &prototypeBatchState{
			expected: request.ExpectedClients,
			payloads: make(map[int]prototypeSubmission, request.ExpectedClients),
		}
		s.batches.rounds[request.RoundID] = state
	}
	status := prototypeBatchStatus(request.RoundID, state)
	s.batches.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]any{"result": status})
}

func (s *Server) submitPrototype(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}

	var submission prototypeSubmission
	if err := decodeJSONRequest(w, r, &submission); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validatePrototypeSubmission(submission); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.batches.mu.Lock()
	state, exists := s.batches.rounds[submission.RoundID]
	if !exists {
		s.batches.mu.Unlock()
		writeError(w, http.StatusConflict, "prototype batch has not been opened")
		return
	}
	if submission.ClientID >= state.expected {
		s.batches.mu.Unlock()
		writeError(w, http.StatusBadRequest, fmt.Sprintf(
			"client_id %d is outside [0, %d]",
			submission.ClientID,
			state.expected-1,
		))
		return
	}
	if existing, duplicate := state.payloads[submission.ClientID]; duplicate {
		if !reflect.DeepEqual(existing, submission) {
			s.batches.mu.Unlock()
			writeError(w, http.StatusConflict, "client already submitted a different prototype")
			return
		}
	} else {
		state.payloads[submission.ClientID] = submission
	}

	if state.submitted || state.submitting || len(state.payloads) < state.expected {
		status := prototypeBatchStatus(submission.RoundID, state)
		s.batches.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{"result": status})
		return
	}

	ordered := make([]prototypeSubmission, state.expected)
	for clientID := 0; clientID < state.expected; clientID++ {
		payload, ok := state.payloads[clientID]
		if !ok {
			s.batches.mu.Unlock()
			writeError(w, http.StatusConflict, fmt.Sprintf("prototype batch is missing client %d", clientID))
			return
		}
		ordered[clientID] = payload
	}
	state.submitting = true
	s.batches.mu.Unlock()

	payloadsJSON, err := json.Marshal(ordered)
	if err == nil {
		_, err = s.client.Submit(
			"SubmitPrototypeBatch",
			strconv.Itoa(submission.RoundID),
			string(payloadsJSON),
		)
	}

	s.batches.mu.Lock()
	state.submitting = false
	if err == nil {
		state.submitted = true
	}
	status := prototypeBatchStatus(submission.RoundID, state)
	s.batches.mu.Unlock()
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Sprintf("submit prototype batch: %v", err))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"result": status})
}

func (s *Server) prototypeBatchStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	roundID, err := strconv.Atoi(r.URL.Query().Get("round_id"))
	if err != nil || roundID < 1 {
		writeError(w, http.StatusBadRequest, "round_id must be a positive integer")
		return
	}

	s.batches.mu.Lock()
	state, exists := s.batches.rounds[roundID]
	if !exists {
		s.batches.mu.Unlock()
		writeError(w, http.StatusNotFound, "prototype batch was not found")
		return
	}
	status := prototypeBatchStatus(roundID, state)
	s.batches.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{"result": status})
}

func prototypeBatchStatus(roundID int, state *prototypeBatchState) prototypeBatchStatusResponse {
	status := "COLLECTING"
	if state.submitted {
		status = "SUBMITTED"
	} else if state.submitting {
		status = "SUBMITTING"
	} else if len(state.payloads) == state.expected {
		status = "READY"
	}
	return prototypeBatchStatusResponse{
		RoundID:         roundID,
		ExpectedClients: state.expected,
		ReceivedClients: len(state.payloads),
		Status:          status,
	}
}

func validatePrototypeSubmission(submission prototypeSubmission) error {
	if submission.Encoding != prototypeEncoding {
		return fmt.Errorf("encoding must be %q", prototypeEncoding)
	}
	if submission.RoundID < 1 || submission.ClientID < 0 {
		return errors.New("round_id must be positive and client_id must be non-negative")
	}
	if len(submission.Shape) != 2 || submission.Shape[0] < 1 || submission.Shape[1] < 1 {
		return errors.New("shape must contain two positive dimensions")
	}
	if submission.Scale < 1 {
		return errors.New("scale must be positive")
	}
	if len(submission.Values) != submission.Shape[0]*submission.Shape[1] {
		return errors.New("prototype value count does not match shape")
	}
	if len(submission.Counts) != submission.Shape[0] {
		return errors.New("prototype counts do not match shape")
	}
	for _, count := range submission.Counts {
		if count < 0 {
			return errors.New("prototype counts must be non-negative")
		}
	}
	return nil
}

func decodeJSONRequest(w http.ResponseWriter, r *http.Request, target any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBody)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return errors.New("request body could not be read")
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return errors.New("request body must be valid JSON")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("request body must contain one JSON object")
	}
	return nil
}
