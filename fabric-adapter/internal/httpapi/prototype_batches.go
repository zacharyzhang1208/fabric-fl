package httpapi

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
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
const prototypeSignatureDomain = "fabric-fl-prototype-v1\n"

type prototypeBatchCollector struct {
	mu     sync.Mutex
	rounds map[int]*prototypeBatchState
}

type prototypeBatchState struct {
	config     openPrototypeBatchRequest
	payloads   map[int]prototypeSubmission
	submitting bool
	processed  bool
	result     json.RawMessage
}

type openPrototypeBatchRequest struct {
	RoundID          int      `json:"round_id"`
	ExperimentID     int      `json:"experiment_id"`
	Sequence         int      `json:"sequence"`
	ExpectedClients  int      `json:"expected_clients"`
	NumClasses       int      `json:"num_classes"`
	Dimension        int      `json:"dimension"`
	Scale            int64    `json:"scale"`
	ClientPublicKeys []string `json:"client_public_keys"`
}

type prototypeSubmission struct {
	Encoding  string  `json:"encoding"`
	RoundID   int     `json:"round_id"`
	ClientID  int     `json:"client_id"`
	Shape     []int   `json:"shape"`
	Scale     int64   `json:"scale"`
	Values    []int64 `json:"values"`
	Counts    []int64 `json:"counts"`
	PublicKey string  `json:"public_key"`
	Signature string  `json:"signature"`
}

type unsignedPrototypeSubmission struct {
	Encoding string  `json:"encoding"`
	RoundID  int     `json:"round_id"`
	ClientID int     `json:"client_id"`
	Shape    []int   `json:"shape"`
	Scale    int64   `json:"scale"`
	Values   []int64 `json:"values"`
	Counts   []int64 `json:"counts"`
}

type prototypeBatchStatusResponse struct {
	RoundID         int             `json:"round_id"`
	ExpectedClients int             `json:"expected_clients"`
	ReceivedClients int             `json:"received_clients"`
	Status          string          `json:"status"`
	RoundResult     json.RawMessage `json:"round_result,omitempty"`
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
	if err := validateBatchConfig(request); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	s.batches.mu.Lock()
	state, exists := s.batches.rounds[request.RoundID]
	if exists && !reflect.DeepEqual(state.config, request) {
		s.batches.mu.Unlock()
		writeError(w, http.StatusConflict, "prototype batch already exists with different configuration")
		return
	}
	if !exists {
		state = &prototypeBatchState{
			config:   request,
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
	if submission.ClientID >= state.config.ExpectedClients {
		s.batches.mu.Unlock()
		writeError(w, http.StatusBadRequest, fmt.Sprintf(
			"client_id %d is outside [0, %d]",
			submission.ClientID,
			state.config.ExpectedClients-1,
		))
		return
	}
	if submission.Shape[0] != state.config.NumClasses ||
		submission.Shape[1] != state.config.Dimension || submission.Scale != state.config.Scale {
		s.batches.mu.Unlock()
		writeError(w, http.StatusBadRequest, "prototype metadata does not match batch configuration")
		return
	}
	if submission.PublicKey != state.config.ClientPublicKeys[submission.ClientID] {
		s.batches.mu.Unlock()
		writeError(w, http.StatusUnauthorized, "prototype public key does not match registered client key")
		return
	}
	if err := verifyPrototypeSubmission(submission); err != nil {
		s.batches.mu.Unlock()
		writeError(w, http.StatusUnauthorized, err.Error())
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

	if state.processed || state.submitting || len(state.payloads) < state.config.ExpectedClients {
		status := prototypeBatchStatus(submission.RoundID, state)
		s.batches.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{"result": status})
		return
	}

	ordered := make([]prototypeSubmission, state.config.ExpectedClients)
	for clientID := 0; clientID < state.config.ExpectedClients; clientID++ {
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
		var result []byte
		result, err = s.client.Submit(
			"ProcessRound",
			strconv.Itoa(submission.RoundID),
			strconv.Itoa(state.config.ExperimentID),
			strconv.Itoa(state.config.Sequence),
			strconv.Itoa(state.config.ExpectedClients),
			strconv.Itoa(state.config.NumClasses),
			strconv.Itoa(state.config.Dimension),
			strconv.FormatInt(state.config.Scale, 10),
			string(payloadsJSON),
		)
		if err == nil {
			if !json.Valid(result) {
				err = errors.New("ProcessRound returned invalid JSON")
			} else {
				stateResult := append(json.RawMessage(nil), result...)
				s.batches.mu.Lock()
				state.result = stateResult
				s.batches.mu.Unlock()
			}
		}
	}

	s.batches.mu.Lock()
	state.submitting = false
	if err == nil {
		state.processed = true
	}
	status := prototypeBatchStatus(submission.RoundID, state)
	s.batches.mu.Unlock()
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Sprintf("process prototype round: %v", err))
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
	if state.processed {
		status = "PROCESSED"
	} else if state.submitting {
		status = "SUBMITTING"
	} else if len(state.payloads) == state.config.ExpectedClients {
		status = "READY"
	}
	return prototypeBatchStatusResponse{
		RoundID:         roundID,
		ExpectedClients: state.config.ExpectedClients,
		ReceivedClients: len(state.payloads),
		Status:          status,
		RoundResult:     state.result,
	}
}

func validateBatchConfig(config openPrototypeBatchRequest) error {
	if config.RoundID < 1 || config.ExperimentID < 1 || config.Sequence < 1 ||
		config.ExpectedClients < 1 || config.NumClasses < 1 || config.Dimension < 1 || config.Scale < 1 {
		return errors.New("all prototype batch configuration values must be positive")
	}
	if len(config.ClientPublicKeys) != config.ExpectedClients {
		return errors.New("client_public_keys must contain one key per client")
	}
	seen := make(map[string]bool, len(config.ClientPublicKeys))
	for clientID, encoded := range config.ClientPublicKeys {
		key, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(key) != ed25519.PublicKeySize {
			return fmt.Errorf("client_public_keys[%d] must be a base64 Ed25519 public key", clientID)
		}
		if seen[encoded] {
			return errors.New("client_public_keys must be unique")
		}
		seen[encoded] = true
	}
	return nil
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
	if submission.PublicKey == "" || submission.Signature == "" {
		return errors.New("public_key and signature are required")
	}
	return nil
}

func verifyPrototypeSubmission(submission prototypeSubmission) error {
	publicKey, err := base64.StdEncoding.DecodeString(submission.PublicKey)
	if err != nil || len(publicKey) != ed25519.PublicKeySize {
		return errors.New("prototype public_key must be a base64 Ed25519 public key")
	}
	signature, err := base64.StdEncoding.DecodeString(submission.Signature)
	if err != nil || len(signature) != ed25519.SignatureSize {
		return errors.New("prototype signature must be a base64 Ed25519 signature")
	}
	unsigned := unsignedPrototypeSubmission{
		Encoding: submission.Encoding,
		RoundID:  submission.RoundID,
		ClientID: submission.ClientID,
		Shape:    submission.Shape,
		Scale:    submission.Scale,
		Values:   submission.Values,
		Counts:   submission.Counts,
	}
	encoded, err := json.Marshal(unsigned)
	if err != nil {
		return errors.New("prototype signing message could not be encoded")
	}
	message := append([]byte(prototypeSignatureDomain), encoded...)
	if !ed25519.Verify(ed25519.PublicKey(publicKey), message, signature) {
		return errors.New("prototype signature verification failed")
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
