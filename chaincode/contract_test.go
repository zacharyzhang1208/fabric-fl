package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"math"
	"testing"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

func TestContractMetadataBuilds(t *testing.T) {
	chaincode, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		t.Fatalf("NewChaincode() error = %v", err)
	}
	if chaincode == nil {
		t.Fatal("NewChaincode() returned nil")
	}
}

func TestAggregatePrototypesUsesSampleCountWeightPerClass(t *testing.T) {
	round := &Round{
		RoundID:         1,
		ExpectedClients: 2,
		NumClasses:      2,
		Dimension:       2,
		Scale:           100,
	}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{
			Values: []int64{100, 200, 300, 400},
			Counts: []int64{5, 0},
		}},
		{PrototypePayload: PrototypePayload{
			Values: []int64{200, 400, 500, 700},
			Counts: []int64{2, 1},
		}},
	}

	global, err := aggregatePrototypes(round, records)
	if err != nil {
		t.Fatalf("aggregatePrototypes() error = %v", err)
	}
	wantValues := []int64{129, 257, 500, 700}
	wantCounts := []int64{7, 1}
	assertInt64Slice(t, global.Values, wantValues)
	assertInt64Slice(t, global.Counts, wantCounts)
}

func TestAggregatePrototypesRoundsHalfAwayFromZero(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 2, NumClasses: 1, Dimension: 2, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{Values: []int64{0, 0}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{1, -1}, Counts: []int64{1}}},
	}

	global, err := aggregatePrototypes(round, records)
	if err != nil {
		t.Fatalf("aggregatePrototypes() error = %v", err)
	}
	assertInt64Slice(t, global.Values, []int64{1, -1})
}

func TestAggregatePrototypesRejectsOverflow(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 1, NumClasses: 1, Dimension: 1, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{Values: []int64{math.MaxInt64}, Counts: []int64{2}}},
	}

	if _, err := aggregatePrototypes(round, records); err == nil {
		t.Fatal("aggregatePrototypes() error = nil, want overflow error")
	}
}

func TestRobustReferencesIgnoreSingleExtremeClient(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 5, NumClasses: 1, Dimension: 2, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{Values: []int64{10, 20}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{11, 19}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{9, 21}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{10, 20}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{Values: []int64{10000, -10000}, Counts: []int64{1}}},
	}

	references, eligible := robustReferences(round, records)
	assertInt64Slice(t, references, []int64{10, 20})
	if !eligible[0] {
		t.Fatal("class should be eligible for reputation assessment")
	}
}

func TestAggregateSelectedPrototypesExcludesRejectedClient(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 3, NumClasses: 1, Dimension: 1, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{ClientID: 0, Values: []int64{10}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{ClientID: 1, Values: []int64{20}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{ClientID: 2, Values: []int64{1000}, Counts: []int64{1}}},
	}

	global, err := aggregateSelectedPrototypes(round, records, map[int]bool{0: true, 1: true, 2: false})
	if err != nil {
		t.Fatalf("aggregateSelectedPrototypes() error = %v", err)
	}
	assertInt64Slice(t, global.Values, []int64{15})
	assertInt64Slice(t, global.Counts, []int64{2})
}

func TestAggregateSelectedPrototypesFallsBackToMedianForEmptyClass(t *testing.T) {
	round := &Round{RoundID: 1, ExpectedClients: 3, NumClasses: 1, Dimension: 1, Scale: 1}
	records := []PrototypeRecord{
		{PrototypePayload: PrototypePayload{ClientID: 0, Values: []int64{10}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{ClientID: 1, Values: []int64{12}, Counts: []int64{1}}},
		{PrototypePayload: PrototypePayload{ClientID: 2, Values: []int64{1000}, Counts: []int64{1}}},
	}

	global, err := aggregateSelectedPrototypes(round, records, map[int]bool{})
	if err != nil {
		t.Fatalf("aggregateSelectedPrototypes() error = %v", err)
	}
	assertInt64Slice(t, global.Values, []int64{12})
	assertInt64Slice(t, global.Counts, []int64{1})
}

func TestReputationScoringAndStatus(t *testing.T) {
	if score := weightedReputation(8000, 0); score != 6400 {
		t.Fatalf("first anomaly score = %d, want 6400", score)
	}
	if status := reputationStatus(6400); status != statusWatch {
		t.Fatalf("reputationStatus(6400) = %s, want %s", status, statusWatch)
	}
	if score := weightedReputation(6400, 0); score != 5120 {
		t.Fatalf("second anomaly score = %d, want 5120", score)
	}
}

func TestValidatePrototypePayload(t *testing.T) {
	round := &Round{RoundID: 3, ExpectedClients: 2, NumClasses: 2, Dimension: 2, Scale: 1000}
	payload := PrototypePayload{
		Encoding: prototypeEncoding,
		RoundID:  3,
		ClientID: 1,
		Shape:    []int{2, 2},
		Scale:    1000,
		Values:   []int64{1, 2, 3, 4},
		Counts:   []int64{2, 0},
	}

	if err := validatePrototypePayload(payload, round, 1); err != nil {
		t.Fatalf("validatePrototypePayload() error = %v", err)
	}
	payload.Shape = []int{2, 3}
	if err := validatePrototypePayload(payload, round, 1); err == nil {
		t.Fatal("validatePrototypePayload() error = nil, want shape error")
	}
}

func TestOrderPrototypeBatchSortsAndValidatesAllClients(t *testing.T) {
	round := &Round{RoundID: 7, ExpectedClients: 2, NumClasses: 1, Dimension: 1, Scale: 100}
	payloads := []PrototypePayload{
		validPrototypePayload(7, 1, 200),
		validPrototypePayload(7, 0, 100),
	}

	ordered, err := orderPrototypeBatch(payloads, round)
	if err != nil {
		t.Fatalf("orderPrototypeBatch() error = %v", err)
	}
	if ordered[0].ClientID != 0 || ordered[1].ClientID != 1 {
		t.Fatalf("ordered client ids = [%d, %d]", ordered[0].ClientID, ordered[1].ClientID)
	}
}

func TestOrderPrototypeBatchRejectsDuplicateOrIncompleteClients(t *testing.T) {
	round := &Round{RoundID: 7, ExpectedClients: 2, NumClasses: 1, Dimension: 1, Scale: 100}
	duplicate := []PrototypePayload{
		validPrototypePayload(7, 0, 100),
		validPrototypePayload(7, 0, 200),
	}
	if _, err := orderPrototypeBatch(duplicate, round); err == nil {
		t.Fatal("orderPrototypeBatch() accepted duplicate client ids")
	}

	incomplete := []PrototypePayload{validPrototypePayload(7, 0, 100)}
	if _, err := orderPrototypeBatch(incomplete, round); err == nil {
		t.Fatal("orderPrototypeBatch() accepted an incomplete batch")
	}
}

func TestPrototypeSignatureRejectsTamperedValues(t *testing.T) {
	payload := validPrototypePayload(7, 0, 100)
	if err := verifyPrototypeSignature(payload); err != nil {
		t.Fatalf("valid signature rejected: %v", err)
	}
	payload.Values[0]++
	if err := verifyPrototypeSignature(payload); err == nil {
		t.Fatal("tampered prototype signature was accepted")
	}
}

func TestPrototypeSignatureMatchesPythonTestVector(t *testing.T) {
	payload := PrototypePayload{
		Encoding:  prototypeEncoding,
		RoundID:   17,
		ClientID:  3,
		Shape:     []int{2, 2},
		Scale:     1_000_000,
		Values:    []int64{1, -2, 3, 4},
		Counts:    []int64{5, 6},
		PublicKey: "A6EHv/POEL4dcN0Y50vAmWfk1jCbpQ1fHdyGZBJVMbg=",
		Signature: "COgFLegKcwX7O+BOVG+zeoYP4tVsF+gar4+1COfM+Z/7Rp3+4LNB+yup2Da23sMVajN8j4/vsMCOSro5SDzlAQ==",
	}
	if err := verifyPrototypeSignature(payload); err != nil {
		t.Fatalf("Python Ed25519 test vector rejected: %v", err)
	}
}

func TestDecodePrototypeBatchRejectsUnknownFields(t *testing.T) {
	_, err := decodePrototypeBatch(`[{"encoding":"fixed-point-int64","round_id":1,"client_id":0,"shape":[1,1],"scale":1,"values":[1],"counts":[1],"extra":true}]`)
	if err == nil {
		t.Fatal("decodePrototypeBatch() accepted an unknown field")
	}
}

func TestPrototypeBatchHashIsStableAfterClientOrdering(t *testing.T) {
	round := &Round{RoundID: 7, ExpectedClients: 2, NumClasses: 1, Dimension: 1, Scale: 100}
	first, err := orderPrototypeBatch([]PrototypePayload{
		validPrototypePayload(7, 0, 100),
		validPrototypePayload(7, 1, 200),
	}, round)
	if err != nil {
		t.Fatalf("order first batch: %v", err)
	}
	second, err := orderPrototypeBatch([]PrototypePayload{
		validPrototypePayload(7, 1, 200),
		validPrototypePayload(7, 0, 100),
	}, round)
	if err != nil {
		t.Fatalf("order second batch: %v", err)
	}
	firstHash, err := prototypeBatchHash(first)
	if err != nil {
		t.Fatalf("hash first batch: %v", err)
	}
	secondHash, err := prototypeBatchHash(second)
	if err != nil {
		t.Fatalf("hash second batch: %v", err)
	}
	if firstHash != secondHash || len(firstHash) != 64 {
		t.Fatalf("hashes = %q and %q", firstHash, secondHash)
	}
}

func TestExistingProcessRoundUsesBatchHashForIdempotency(t *testing.T) {
	existing := &Round{
		RoundID: 7, ExperimentID: 6, Sequence: 1, ExpectedClients: 2,
		NumClasses: 1, Dimension: 1, Scale: 100, Status: statusFinalized,
		PrototypeBatchHash: "same-hash",
	}
	requested := *existing
	if _, err := existingProcessRoundResult(existing, &requested); err != nil {
		t.Fatalf("identical retry rejected: %v", err)
	}
	requested.PrototypeBatchHash = "different-hash"
	if _, err := existingProcessRoundResult(existing, &requested); err == nil {
		t.Fatal("conflicting retry was accepted")
	}
}

func validPrototypePayload(roundID int, clientID int, value int64) PrototypePayload {
	payload := PrototypePayload{
		Encoding: prototypeEncoding,
		RoundID:  roundID,
		ClientID: clientID,
		Shape:    []int{1, 1},
		Scale:    100,
		Values:   []int64{value},
		Counts:   []int64{1},
	}
	seed := bytes.Repeat([]byte{byte(clientID + 1)}, ed25519.SeedSize)
	privateKey := ed25519.NewKeyFromSeed(seed)
	payload.PublicKey = base64.StdEncoding.EncodeToString(privateKey.Public().(ed25519.PublicKey))
	unsigned := unsignedPrototypePayload{
		Encoding: payload.Encoding, RoundID: payload.RoundID, ClientID: payload.ClientID,
		Shape: payload.Shape, Scale: payload.Scale, Values: payload.Values, Counts: payload.Counts,
	}
	encoded, err := json.Marshal(unsigned)
	if err != nil {
		panic(err)
	}
	payload.Signature = base64.StdEncoding.EncodeToString(
		ed25519.Sign(privateKey, append([]byte(prototypeSignatureDomain), encoded...)),
	)
	return payload
}

func assertInt64Slice(t *testing.T, got []int64, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("got[%d] = %d, want %d", index, got[index], want[index])
		}
	}
}
