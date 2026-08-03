package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strconv"
	"strings"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const (
	roundObjectType            = "round"
	prototypeObjectType        = "prototype"
	globalPrototypeObjectType  = "globalPrototype"
	reputationObjectType       = "clientReputation"
	assessmentObjectType       = "clientAssessment"
	reputationReportObjectType = "reputationReport"
	experimentRoundObjectType  = "experimentRound"
	prototypeEncoding          = "fixed-point-int64"
	statusOpen                 = "OPEN"
	statusFinalized            = "FINALIZED"
	maxPrototypeValues         = 1_000_000
)

type SmartContract struct {
	contractapi.Contract
}

type Round struct {
	DocType         string `json:"doc_type"`
	RoundID         int    `json:"round_id"`
	ExperimentID    int    `json:"experiment_id"`
	Sequence        int    `json:"sequence"`
	ExpectedClients int    `json:"expected_clients"`
	NumClasses      int    `json:"num_classes"`
	Dimension       int    `json:"dimension"`
	Scale           int64  `json:"scale"`
	Status          string `json:"status"`
	CreatorMSP      string `json:"creator_msp"`
	FinalizedTxID   string `json:"finalized_tx_id,omitempty"`
}

type PrototypePayload struct {
	Encoding string  `json:"encoding"`
	RoundID  int     `json:"round_id"`
	ClientID int     `json:"client_id"`
	Shape    []int   `json:"shape"`
	Scale    int64   `json:"scale"`
	Values   []int64 `json:"values"`
	Counts   []int64 `json:"counts"`
}

type PrototypeRecord struct {
	PrototypePayload
	DocType        string `json:"doc_type"`
	SubmittedByMSP string `json:"submitted_by_msp"`
	TransactionID  string `json:"transaction_id"`
}

type GlobalPrototype struct {
	DocType  string  `json:"doc_type"`
	RoundID  int     `json:"round_id"`
	Encoding string  `json:"encoding"`
	Shape    []int   `json:"shape"`
	Scale    int64   `json:"scale"`
	Values   []int64 `json:"values"`
	Counts   []int64 `json:"counts"`
}

func (s *SmartContract) Set(ctx contractapi.TransactionContextInterface, key string, value string) error {
	if isReservedKey(key) {
		return fmt.Errorf("key %q uses a reserved prototype namespace", key)
	}
	return ctx.GetStub().PutState(key, []byte(value))
}

func (s *SmartContract) Get(ctx contractapi.TransactionContextInterface, key string) (string, error) {
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return "", fmt.Errorf("failed to read key %q: %w", key, err)
	}
	if value == nil {
		return "", fmt.Errorf("key %q does not exist", key)
	}
	return string(value), nil
}

func (s *SmartContract) CreateRound(
	ctx contractapi.TransactionContextInterface,
	roundID int,
	experimentID int,
	sequence int,
	expectedClients int,
	numClasses int,
	dimension int,
	scale int64,
) error {
	if err := validateRoundConfig(roundID, experimentID, sequence, expectedClients, numClasses, dimension, scale); err != nil {
		return err
	}

	key, err := roundKey(ctx, roundID)
	if err != nil {
		return err
	}
	existing, err := ctx.GetStub().GetState(key)
	if err != nil {
		return fmt.Errorf("read round %d: %w", roundID, err)
	}
	if existing != nil {
		var current Round
		if err := json.Unmarshal(existing, &current); err != nil {
			return fmt.Errorf("decode existing round %d: %w", roundID, err)
		}
		if current.ExpectedClients == expectedClients &&
			current.ExperimentID == experimentID &&
			current.Sequence == sequence &&
			current.NumClasses == numClasses &&
			current.Dimension == dimension &&
			current.Scale == scale {
			return nil
		}
		return fmt.Errorf("round %d already exists with different configuration", roundID)
	}

	sequenceKey, err := experimentRoundKey(ctx, experimentID, sequence)
	if err != nil {
		return err
	}
	sequenceState, err := ctx.GetStub().GetState(sequenceKey)
	if err != nil {
		return fmt.Errorf("read experiment %d sequence %d: %w", experimentID, sequence, err)
	}
	if sequenceState != nil {
		return fmt.Errorf("experiment %d sequence %d already belongs to round %s", experimentID, sequence, sequenceState)
	}
	if sequence > 1 {
		previousKey, err := experimentRoundKey(ctx, experimentID, sequence-1)
		if err != nil {
			return err
		}
		previousState, err := ctx.GetStub().GetState(previousKey)
		if err != nil {
			return fmt.Errorf("read experiment %d sequence %d: %w", experimentID, sequence-1, err)
		}
		if previousState == nil {
			return fmt.Errorf("experiment %d sequence %d must be created first", experimentID, sequence-1)
		}
		previousRoundID, err := strconv.Atoi(string(previousState))
		if err != nil {
			return fmt.Errorf("experiment %d sequence %d has invalid round id: %w", experimentID, sequence-1, err)
		}
		previousRound, err := getRound(ctx, previousRoundID)
		if err != nil {
			return err
		}
		if previousRound.Status != statusFinalized {
			return fmt.Errorf("experiment %d sequence %d is not finalized", experimentID, sequence-1)
		}
		if previousRound.ExpectedClients != expectedClients ||
			previousRound.NumClasses != numClasses ||
			previousRound.Dimension != dimension ||
			previousRound.Scale != scale {
			return fmt.Errorf("experiment %d configuration cannot change after sequence 1", experimentID)
		}
	}

	mspID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("get creator MSP: %w", err)
	}
	round := Round{
		DocType:         roundObjectType,
		RoundID:         roundID,
		ExperimentID:    experimentID,
		Sequence:        sequence,
		ExpectedClients: expectedClients,
		NumClasses:      numClasses,
		Dimension:       dimension,
		Scale:           scale,
		Status:          statusOpen,
		CreatorMSP:      mspID,
	}
	if err := ctx.GetStub().PutState(sequenceKey, []byte(strconv.Itoa(roundID))); err != nil {
		return fmt.Errorf("write experiment %d sequence %d: %w", experimentID, sequence, err)
	}
	return putJSON(ctx, key, round)
}

func (s *SmartContract) SubmitPrototype(
	ctx contractapi.TransactionContextInterface,
	roundID int,
	clientID int,
	payloadJSON string,
) error {
	round, err := getRound(ctx, roundID)
	if err != nil {
		return err
	}
	if round.Status != statusOpen {
		return fmt.Errorf("round %d is %s; prototype submissions are closed", roundID, round.Status)
	}
	if clientID < 0 || clientID >= round.ExpectedClients {
		return fmt.Errorf("client_id %d is outside [0, %d]", clientID, round.ExpectedClients-1)
	}

	payload, err := decodePrototypePayload(payloadJSON)
	if err != nil {
		return fmt.Errorf("invalid prototype payload: %w", err)
	}
	if err := validatePrototypePayload(payload, round, clientID); err != nil {
		return err
	}

	key, err := prototypeKey(ctx, roundID, clientID)
	if err != nil {
		return err
	}
	existing, err := ctx.GetStub().GetState(key)
	if err != nil {
		return fmt.Errorf("read prototype for round %d client %d: %w", roundID, clientID, err)
	}
	if existing != nil {
		return fmt.Errorf("prototype for round %d client %d already exists", roundID, clientID)
	}

	mspID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("get submitter MSP: %w", err)
	}
	record := PrototypeRecord{
		PrototypePayload: payload,
		DocType:          prototypeObjectType,
		SubmittedByMSP:   mspID,
		TransactionID:    ctx.GetStub().GetTxID(),
	}
	return putJSON(ctx, key, record)
}

func (s *SmartContract) SubmitPrototypeBatch(
	ctx contractapi.TransactionContextInterface,
	roundID int,
	payloadsJSON string,
) error {
	round, err := getRound(ctx, roundID)
	if err != nil {
		return err
	}
	if round.Status != statusOpen {
		return fmt.Errorf("round %d is %s; prototype submissions are closed", roundID, round.Status)
	}

	payloads, err := decodePrototypeBatch(payloadsJSON)
	if err != nil {
		return fmt.Errorf("invalid prototype batch: %w", err)
	}
	ordered, err := orderPrototypeBatch(payloads, round)
	if err != nil {
		return err
	}

	keys := make([]string, len(ordered))
	existingRecords := 0
	for clientID := range ordered {
		key, err := prototypeKey(ctx, roundID, clientID)
		if err != nil {
			return err
		}
		existing, err := ctx.GetStub().GetState(key)
		if err != nil {
			return fmt.Errorf("read prototype for round %d client %d: %w", roundID, clientID, err)
		}
		if existing != nil {
			var record PrototypeRecord
			if err := json.Unmarshal(existing, &record); err != nil {
				return fmt.Errorf("decode existing prototype for round %d client %d: %w", roundID, clientID, err)
			}
			if !reflect.DeepEqual(record.PrototypePayload, ordered[clientID]) {
				return fmt.Errorf("prototype for round %d client %d already exists with different content", roundID, clientID)
			}
			existingRecords++
		}
		keys[clientID] = key
	}
	if existingRecords == len(ordered) {
		return nil
	}
	if existingRecords != 0 {
		return fmt.Errorf(
			"round %d has a partial prototype batch with %d of %d clients",
			roundID,
			existingRecords,
			len(ordered),
		)
	}

	mspID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("get batch submitter MSP: %w", err)
	}
	transactionID := ctx.GetStub().GetTxID()
	for clientID, payload := range ordered {
		record := PrototypeRecord{
			PrototypePayload: payload,
			DocType:          prototypeObjectType,
			SubmittedByMSP:   mspID,
			TransactionID:    transactionID,
		}
		if err := putJSON(ctx, keys[clientID], record); err != nil {
			return err
		}
	}
	return nil
}

func (s *SmartContract) FinalizeRound(ctx contractapi.TransactionContextInterface, roundID int) error {
	round, err := getRound(ctx, roundID)
	if err != nil {
		return err
	}
	if round.Status == statusFinalized {
		return nil
	}
	if round.Status != statusOpen {
		return fmt.Errorf("round %d has unsupported status %q", roundID, round.Status)
	}

	records := make([]PrototypeRecord, 0, round.ExpectedClients)
	for clientID := 0; clientID < round.ExpectedClients; clientID++ {
		key, err := prototypeKey(ctx, roundID, clientID)
		if err != nil {
			return err
		}
		value, err := ctx.GetStub().GetState(key)
		if err != nil {
			return fmt.Errorf("read prototype for round %d client %d: %w", roundID, clientID, err)
		}
		if value == nil {
			return fmt.Errorf("round %d is missing prototype for client %d", roundID, clientID)
		}

		var record PrototypeRecord
		if err := json.Unmarshal(value, &record); err != nil {
			return fmt.Errorf("decode prototype for round %d client %d: %w", roundID, clientID, err)
		}
		if err := validatePrototypePayload(record.PrototypePayload, round, clientID); err != nil {
			return fmt.Errorf("stored prototype for client %d is invalid: %w", clientID, err)
		}
		records = append(records, record)
	}

	assessments, reputations, report, err := assessPrototypeReputations(ctx, round, records)
	if err != nil {
		return fmt.Errorf("assess round %d: %w", roundID, err)
	}
	included := make(map[int]bool, len(assessments))
	for index, assessment := range assessments {
		included[assessment.ClientID] = assessment.Included
		assessmentKey, err := clientAssessmentKey(ctx, roundID, assessment.ClientID)
		if err != nil {
			return err
		}
		if err := putJSON(ctx, assessmentKey, assessment); err != nil {
			return err
		}

		reputationKey, err := clientReputationKey(ctx, round.ExperimentID, assessment.ClientID)
		if err != nil {
			return err
		}
		if err := putJSON(ctx, reputationKey, reputations[index]); err != nil {
			return err
		}
	}
	reportKey, err := reputationReportKey(ctx, roundID)
	if err != nil {
		return err
	}
	if err := putJSON(ctx, reportKey, report); err != nil {
		return err
	}

	global, err := aggregateSelectedPrototypes(round, records, included)
	if err != nil {
		return fmt.Errorf("aggregate round %d: %w", roundID, err)
	}
	globalKey, err := globalPrototypeKey(ctx, roundID)
	if err != nil {
		return err
	}
	if err := putJSON(ctx, globalKey, global); err != nil {
		return err
	}

	round.Status = statusFinalized
	round.FinalizedTxID = ctx.GetStub().GetTxID()
	roundStateKey, err := roundKey(ctx, roundID)
	if err != nil {
		return err
	}
	return putJSON(ctx, roundStateKey, round)
}

func (s *SmartContract) GetGlobalPrototype(
	ctx contractapi.TransactionContextInterface,
	roundID int,
) (*GlobalPrototype, error) {
	key, err := globalPrototypeKey(ctx, roundID)
	if err != nil {
		return nil, err
	}
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, fmt.Errorf("read global prototype for round %d: %w", roundID, err)
	}
	if value == nil {
		return nil, fmt.Errorf("global prototype for round %d does not exist", roundID)
	}

	var global GlobalPrototype
	if err := json.Unmarshal(value, &global); err != nil {
		return nil, fmt.Errorf("decode global prototype for round %d: %w", roundID, err)
	}
	return &global, nil
}

func (s *SmartContract) GetClientReputation(
	ctx contractapi.TransactionContextInterface,
	experimentID int,
	clientID int,
) (*ClientReputation, error) {
	if experimentID < 1 || clientID < 0 {
		return nil, errors.New("experiment_id must be positive and client_id must be non-negative")
	}
	key, err := clientReputationKey(ctx, experimentID, clientID)
	if err != nil {
		return nil, err
	}
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, fmt.Errorf("read reputation for experiment %d client %d: %w", experimentID, clientID, err)
	}
	if value == nil {
		return nil, fmt.Errorf("reputation for experiment %d client %d does not exist", experimentID, clientID)
	}
	var reputation ClientReputation
	if err := json.Unmarshal(value, &reputation); err != nil {
		return nil, fmt.Errorf("decode reputation for experiment %d client %d: %w", experimentID, clientID, err)
	}
	return &reputation, nil
}

func (s *SmartContract) GetRoundReputationReport(
	ctx contractapi.TransactionContextInterface,
	roundID int,
) (*ReputationReport, error) {
	key, err := reputationReportKey(ctx, roundID)
	if err != nil {
		return nil, err
	}
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, fmt.Errorf("read reputation report for round %d: %w", roundID, err)
	}
	if value == nil {
		return nil, fmt.Errorf("reputation report for round %d does not exist", roundID)
	}
	var report ReputationReport
	if err := json.Unmarshal(value, &report); err != nil {
		return nil, fmt.Errorf("decode reputation report for round %d: %w", roundID, err)
	}
	return &report, nil
}

func validateRoundConfig(roundID int, experimentID int, sequence int, expectedClients int, numClasses int, dimension int, scale int64) error {
	if roundID < 1 || experimentID < 1 || sequence < 1 {
		return errors.New("round_id, experiment_id, and sequence must be positive")
	}
	if expectedClients < 1 {
		return errors.New("expected_clients must be positive")
	}
	if numClasses < 1 || dimension < 1 {
		return errors.New("num_classes and dimension must be positive")
	}
	if scale < 1 {
		return errors.New("scale must be positive")
	}
	if numClasses > maxPrototypeValues/dimension {
		return fmt.Errorf("prototype shape exceeds %d values", maxPrototypeValues)
	}
	return nil
}

func decodePrototypePayload(payloadJSON string) (PrototypePayload, error) {
	decoder := json.NewDecoder(bytes.NewBufferString(payloadJSON))
	decoder.DisallowUnknownFields()

	var payload PrototypePayload
	if err := decoder.Decode(&payload); err != nil {
		return PrototypePayload{}, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return PrototypePayload{}, errors.New("payload must contain one JSON object")
	}
	return payload, nil
}

func decodePrototypeBatch(payloadsJSON string) ([]PrototypePayload, error) {
	decoder := json.NewDecoder(bytes.NewBufferString(payloadsJSON))
	decoder.DisallowUnknownFields()

	var payloads []PrototypePayload
	if err := decoder.Decode(&payloads); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("batch must contain one JSON array")
	}
	if payloads == nil {
		return nil, errors.New("batch must be a JSON array")
	}
	return payloads, nil
}

func orderPrototypeBatch(payloads []PrototypePayload, round *Round) ([]PrototypePayload, error) {
	if len(payloads) != round.ExpectedClients {
		return nil, fmt.Errorf(
			"prototype batch contains %d clients; expected %d",
			len(payloads),
			round.ExpectedClients,
		)
	}

	ordered := make([]PrototypePayload, round.ExpectedClients)
	seen := make([]bool, round.ExpectedClients)
	for _, payload := range payloads {
		clientID := payload.ClientID
		if clientID < 0 || clientID >= round.ExpectedClients {
			return nil, fmt.Errorf("client_id %d is outside [0, %d]", clientID, round.ExpectedClients-1)
		}
		if seen[clientID] {
			return nil, fmt.Errorf("prototype batch contains duplicate client_id %d", clientID)
		}
		if err := validatePrototypePayload(payload, round, clientID); err != nil {
			return nil, fmt.Errorf("prototype for client %d is invalid: %w", clientID, err)
		}
		seen[clientID] = true
		ordered[clientID] = payload
	}
	return ordered, nil
}

func validatePrototypePayload(payload PrototypePayload, round *Round, clientID int) error {
	if payload.Encoding != prototypeEncoding {
		return fmt.Errorf("encoding must be %q", prototypeEncoding)
	}
	if payload.RoundID != round.RoundID {
		return fmt.Errorf("payload round_id %d does not match round %d", payload.RoundID, round.RoundID)
	}
	if payload.ClientID != clientID {
		return fmt.Errorf("payload client_id %d does not match client %d", payload.ClientID, clientID)
	}
	if len(payload.Shape) != 2 || payload.Shape[0] != round.NumClasses || payload.Shape[1] != round.Dimension {
		return fmt.Errorf("shape must be [%d,%d]", round.NumClasses, round.Dimension)
	}
	if payload.Scale != round.Scale {
		return fmt.Errorf("scale %d does not match round scale %d", payload.Scale, round.Scale)
	}
	expectedValues := round.NumClasses * round.Dimension
	if len(payload.Values) != expectedValues {
		return fmt.Errorf("values contains %d items; expected %d", len(payload.Values), expectedValues)
	}
	if len(payload.Counts) != round.NumClasses {
		return fmt.Errorf("counts contains %d items; expected %d", len(payload.Counts), round.NumClasses)
	}
	for classID, count := range payload.Counts {
		if count < 0 {
			return fmt.Errorf("count for class %d must be non-negative", classID)
		}
	}
	return nil
}

func aggregatePrototypes(round *Round, records []PrototypeRecord) (*GlobalPrototype, error) {
	included := make(map[int]bool, len(records))
	for _, record := range records {
		included[record.ClientID] = true
	}
	return aggregateSelectedPrototypes(round, records, included)
}

func aggregateSelectedPrototypes(round *Round, records []PrototypeRecord, included map[int]bool) (*GlobalPrototype, error) {
	if len(records) != round.ExpectedClients {
		return nil, fmt.Errorf("received %d prototypes; expected %d", len(records), round.ExpectedClients)
	}

	valueCount := round.NumClasses * round.Dimension
	sums := make([]int64, valueCount)
	counts := make([]int64, round.NumClasses)
	for _, record := range records {
		if !included[record.ClientID] {
			continue
		}
		for classID, sampleCount := range record.Counts {
			if sampleCount == 0 {
				continue
			}
			newCount, err := checkedAdd(counts[classID], sampleCount)
			if err != nil {
				return nil, fmt.Errorf("class %d sample count: %w", classID, err)
			}
			counts[classID] = newCount
			offset := classID * round.Dimension
			for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
				index := offset + dimensionID
				weightedValue, err := checkedMul(record.Values[index], sampleCount)
				if err != nil {
					return nil, fmt.Errorf("class %d dimension %d: %w", classID, dimensionID, err)
				}
				sum, err := checkedAdd(sums[index], weightedValue)
				if err != nil {
					return nil, fmt.Errorf("class %d dimension %d: %w", classID, dimensionID, err)
				}
				sums[index] = sum
			}
		}
	}

	values := make([]int64, valueCount)
	for classID, sampleCount := range counts {
		if sampleCount == 0 {
			fallbackValues := make([][]int64, round.Dimension)
			for _, record := range records {
				if record.Counts[classID] == 0 {
					continue
				}
				offset := classID * round.Dimension
				for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
					fallbackValues[dimensionID] = append(fallbackValues[dimensionID], record.Values[offset+dimensionID])
				}
			}
			if len(fallbackValues[0]) > 0 {
				offset := classID * round.Dimension
				for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
					values[offset+dimensionID] = medianInt64(fallbackValues[dimensionID])
				}
				counts[classID] = 1
			}
			continue
		}
		offset := classID * round.Dimension
		for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
			values[offset+dimensionID] = divideRoundNearest(sums[offset+dimensionID], sampleCount)
		}
	}

	return &GlobalPrototype{
		DocType:  globalPrototypeObjectType,
		RoundID:  round.RoundID,
		Encoding: prototypeEncoding,
		Shape:    []int{round.NumClasses, round.Dimension},
		Scale:    round.Scale,
		Values:   values,
		Counts:   counts,
	}, nil
}

func checkedAdd(left int64, right int64) (int64, error) {
	if right > 0 && left > math.MaxInt64-right {
		return 0, errors.New("int64 overflow")
	}
	if right < 0 && left < math.MinInt64-right {
		return 0, errors.New("int64 underflow")
	}
	return left + right, nil
}

func checkedMul(left int64, right int64) (int64, error) {
	if left == 0 || right == 0 {
		return 0, nil
	}
	if left == math.MinInt64 && right == -1 {
		return 0, errors.New("int64 overflow")
	}
	if right == math.MinInt64 && left == -1 {
		return 0, errors.New("int64 overflow")
	}
	result := left * right
	if result/right != left {
		if (left < 0) != (right < 0) {
			return 0, errors.New("int64 underflow")
		}
		return 0, errors.New("int64 overflow")
	}
	return result, nil
}

func divideRoundNearest(value int64, divisor int64) int64 {
	quotient := value / divisor
	remainder := value % divisor
	absRemainder := remainder
	if absRemainder < 0 {
		absRemainder = -absRemainder
	}
	if absRemainder*2 >= divisor {
		if value < 0 {
			return quotient - 1
		}
		return quotient + 1
	}
	return quotient
}

func getRound(ctx contractapi.TransactionContextInterface, roundID int) (*Round, error) {
	if roundID < 1 {
		return nil, errors.New("round_id must be positive")
	}
	key, err := roundKey(ctx, roundID)
	if err != nil {
		return nil, err
	}
	value, err := ctx.GetStub().GetState(key)
	if err != nil {
		return nil, fmt.Errorf("read round %d: %w", roundID, err)
	}
	if value == nil {
		return nil, fmt.Errorf("round %d does not exist", roundID)
	}

	var round Round
	if err := json.Unmarshal(value, &round); err != nil {
		return nil, fmt.Errorf("decode round %d: %w", roundID, err)
	}
	return &round, nil
}

func putJSON(ctx contractapi.TransactionContextInterface, key string, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode state: %w", err)
	}
	if err := ctx.GetStub().PutState(key, encoded); err != nil {
		return fmt.Errorf("write state: %w", err)
	}
	return nil
}

func roundKey(ctx contractapi.TransactionContextInterface, roundID int) (string, error) {
	return compositeKey(ctx, roundObjectType, strconv.Itoa(roundID))
}

func prototypeKey(ctx contractapi.TransactionContextInterface, roundID int, clientID int) (string, error) {
	return compositeKey(ctx, prototypeObjectType, strconv.Itoa(roundID), strconv.Itoa(clientID))
}

func globalPrototypeKey(ctx contractapi.TransactionContextInterface, roundID int) (string, error) {
	return compositeKey(ctx, globalPrototypeObjectType, strconv.Itoa(roundID))
}

func clientReputationKey(ctx contractapi.TransactionContextInterface, experimentID int, clientID int) (string, error) {
	return compositeKey(ctx, reputationObjectType, strconv.Itoa(experimentID), strconv.Itoa(clientID))
}

func clientAssessmentKey(ctx contractapi.TransactionContextInterface, roundID int, clientID int) (string, error) {
	return compositeKey(ctx, assessmentObjectType, strconv.Itoa(roundID), strconv.Itoa(clientID))
}

func reputationReportKey(ctx contractapi.TransactionContextInterface, roundID int) (string, error) {
	return compositeKey(ctx, reputationReportObjectType, strconv.Itoa(roundID))
}

func experimentRoundKey(ctx contractapi.TransactionContextInterface, experimentID int, sequence int) (string, error) {
	return compositeKey(ctx, experimentRoundObjectType, strconv.Itoa(experimentID), strconv.Itoa(sequence))
}

func compositeKey(ctx contractapi.TransactionContextInterface, objectType string, attributes ...string) (string, error) {
	key, err := ctx.GetStub().CreateCompositeKey(objectType, attributes)
	if err != nil {
		return "", fmt.Errorf("create %s key: %w", objectType, err)
	}
	return key, nil
}

func isReservedKey(key string) bool {
	if strings.ContainsRune(key, rune(0)) {
		return true
	}
	for _, prefix := range []string{"round:", "prototype:", "global-prototype:", "reputation:", "assessment:", "experiment:"} {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}
