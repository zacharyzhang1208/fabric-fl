package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const (
	roundObjectType           = "round"
	prototypeObjectType       = "prototype"
	globalPrototypeObjectType = "globalPrototype"
	prototypeEncoding         = "fixed-point-int64"
	statusOpen                = "OPEN"
	statusFinalized           = "FINALIZED"
	maxPrototypeValues        = 1_000_000
)

type SmartContract struct {
	contractapi.Contract
}

type Round struct {
	DocType         string `json:"doc_type"`
	RoundID         int    `json:"round_id"`
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
	expectedClients int,
	numClasses int,
	dimension int,
	scale int64,
) error {
	if err := validateRoundConfig(roundID, expectedClients, numClasses, dimension, scale); err != nil {
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
			current.NumClasses == numClasses &&
			current.Dimension == dimension &&
			current.Scale == scale {
			return nil
		}
		return fmt.Errorf("round %d already exists with different configuration", roundID)
	}

	mspID, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return fmt.Errorf("get creator MSP: %w", err)
	}
	round := Round{
		DocType:         roundObjectType,
		RoundID:         roundID,
		ExpectedClients: expectedClients,
		NumClasses:      numClasses,
		Dimension:       dimension,
		Scale:           scale,
		Status:          statusOpen,
		CreatorMSP:      mspID,
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

	global, err := aggregatePrototypes(round, records)
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

func validateRoundConfig(roundID int, expectedClients int, numClasses int, dimension int, scale int64) error {
	if roundID < 1 {
		return errors.New("round_id must be positive")
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
	if len(records) != round.ExpectedClients {
		return nil, fmt.Errorf("received %d prototypes; expected %d", len(records), round.ExpectedClients)
	}

	valueCount := round.NumClasses * round.Dimension
	sums := make([]int64, valueCount)
	contributors := make([]int64, round.NumClasses)
	for _, record := range records {
		for classID, sampleCount := range record.Counts {
			if sampleCount == 0 {
				continue
			}
			contributors[classID]++
			offset := classID * round.Dimension
			for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
				index := offset + dimensionID
				sum, err := checkedAdd(sums[index], record.Values[index])
				if err != nil {
					return nil, fmt.Errorf("class %d dimension %d: %w", classID, dimensionID, err)
				}
				sums[index] = sum
			}
		}
	}

	values := make([]int64, valueCount)
	for classID, contributorCount := range contributors {
		if contributorCount == 0 {
			continue
		}
		offset := classID * round.Dimension
		for dimensionID := 0; dimensionID < round.Dimension; dimensionID++ {
			values[offset+dimensionID] = divideRoundNearest(sums[offset+dimensionID], contributorCount)
		}
	}

	return &GlobalPrototype{
		DocType:  globalPrototypeObjectType,
		RoundID:  round.RoundID,
		Encoding: prototypeEncoding,
		Shape:    []int{round.NumClasses, round.Dimension},
		Scale:    round.Scale,
		Values:   values,
		Counts:   contributors,
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
	for _, prefix := range []string{"round:", "prototype:", "global-prototype:"} {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}
